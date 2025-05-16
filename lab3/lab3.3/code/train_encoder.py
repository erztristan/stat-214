import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import random
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
#import from  files
from encoder import Encoder
from data import TextDataset, DataLoader, BertTokenizerFast

# --- MLM MASKING ---
def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15, seed = 42):
    '''
    Implement MLM masking
    Args:
        input_ids: Input IDs
        vocab_size: Vocabulary size
        mask_token_id: Mask token ID
        pad_token_id: Pad token ID
        mlm_prob: Probability of masking
    '''
    # Create a copy of the original inputs to modify
    labels = input_ids.clone()

    # Set seeds for deciding where to mask
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    probability_matrix = torch.full(labels.shape, mlm_prob, device=labels.device)
    padding_mask = input_ids.eq(pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Replace labels of non-masked tokens with -100 so they're ignored in the loss
    labels[~masked_indices] = -100
    
    # masked tokens:
    # - 80% will  replaced with [MASK]
    # - 10% will  replaced with random token
    # - 10% will remained unchanged
    
    # Set up indices for each case
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
    
    # Make a copy of input_ids to modify
    masked_input_ids = input_ids.clone()
    
    # Replace with [MASK] token (80% of masked tokens)
    masked_input_ids[indices_replaced] = mask_token_id
    
    # Replace with random token (10% of masked tokens)
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=labels.device)
    masked_input_ids[indices_random] = random_words[indices_random]
    
    # The remaining 10% will be unchanged
    return masked_input_ids, labels

def evaluate_model(model, dataloader, tokenizer, device):
    """
    Evaluate the model on the validation set
    Args:
        model: BERT model
        dataloader: Validation data loader
        tokenizer: Tokenizer
        device: Device to run the model on
    Returns:
        avg_loss: Average loss on validation set
    """
    model.eval()  
    total_loss = 0
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    
    with torch.no_grad():  
        for batch in dataloader:
            # Get input batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            attention_mask = attention_mask.squeeze()
            
            # Apply the masking
            masked_input_ids, labels = mask_tokens(
                input_ids=input_ids,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id
            )
            
            # Forward 
            outputs = model(
                input_ids=masked_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )[0]
            
            # Calculating losses
            loss_fct = nn.CrossEntropyLoss()
            active_loss = labels != -100
            active_logits = outputs[active_loss].view(-1, vocab_size)
            active_labels = labels[active_loss].view(-1)
            loss = loss_fct(active_logits, active_labels)
            
            total_loss += loss.item()
    
    # by Calculating average loss
    avg_loss = total_loss / len(dataloader)
    model.train()  # Set model back to training mode
    return avg_loss

def train_bert(model, train_dataloader, val_dataloader, tokenizer, epochs=3, lr=5e-4, 
               mask_prob = 0.15, device='cuda'):
    '''
    Implement training loop for BERT with validation
    Args:
        model: BERT model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        tokenizer: Tokenizer
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run the model on
    '''
    # Move model to the device (either cpu or gpu)
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    
    # loss arrays used for plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            attention_mask = attention_mask.squeeze()
            
            # Apply masking
            masked_input_ids, labels = mask_tokens(
                input_ids=input_ids,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                mlm_prob = mask_prob
            )
            
            # Forward
            model.zero_grad()
            outputs = model(
                input_ids=masked_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )[0]
            
            # Calculate loss - outputs is prediction scores from encoder
            loss_fct = nn.CrossEntropyLoss()
            # Reshape outputs for loss calculation 
            active_loss = labels != -100
            active_logits = outputs[active_loss].view(-1, vocab_size)
            active_labels = labels[active_loss].view(-1)
            loss = loss_fct(active_logits, active_labels)
            
            # optimization that is in backward pass
            loss.backward()
            optimizer.step()
            # Updating the progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})
        
        # Calculate average trainings loss for the each epoch
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # testing phase
        avg_val_loss = evaluate_model(model, val_dataloader, tokenizer, device)
        val_losses.append(avg_val_loss)
        
        # Reporting the epoch results
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Ploting loss curves
    plot_loss_curves(train_losses, val_losses)
    
    return model, train_losses, val_losses

def plot_loss_curves(train_losses, val_losses):
    """
    Plot and save the training and validation loss curves
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('bert_loss_curves.png')
    print("Loss curves saved as bert_loss_curves.png")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BERT model with MLM')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=32, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set split ratio')
    args = parser.parse_args()
    
    # Loading tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Example dataset - you can replace with your real dataset
    texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are powerful models for NLP tasks.",
    "Masked language modeling trains BERT to understand context.",
    "Pretraining is followed by task-specific fine-tuning.",
    "Natural language processing enables machines to interpret human language.",
    "Attention mechanisms help models focus on important parts of a sentence.",
    "The cat sat on the mat while the sun was shining.",
    "Transfer learning allows models to reuse knowledge from one task to another.",
    "Deep learning has revolutionized computer vision and NLP fields.",
    "BERT stands for Bidirectional Encoder Representations from Transformers.",
    "Tokenization splits text into smaller units like words or subwords.",
    "GPT models generate human-like text based on input prompts.",
    "Training a language model requires large datasets and compute resources.",
    "Sequence-to-sequence models are used for translation tasks.",
    "The weather today is sunny with a chance of showers.",
    "Machine learning algorithms improve through experience with data.",
    "Early stopping prevents overfitting during model training.",
    "Optimization techniques like Adam enhance gradient descent.",
    "Fine-tuning adjusts pretrained models to perform specific tasks.",
    "Evaluation metrics measure the quality of model predictions."
    ]
    
    # Spliting of data
    random.seed(42)  
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    split_idx = int(len(texts) * args.val_split)
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]
    
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, max_len=args.max_len)
    val_dataset = TextDataset(val_texts, tokenizer, max_len=args.max_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initializing model
    vocab_size = len(tokenizer)
    print("vocab size is = ", vocab_size)
    model = Encoder(
        vocab_size=vocab_size,
        hidden_size=256,
        num_heads=4,
        num_layers=4,
        intermediate_size=512,
        max_len=args.max_len
    )
    
    # Training of model
    print(f"Starting training on {args.device}...")
    model, train_losses, val_losses = train_bert(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )
    
    # Save the trained model
    torch.save(model.state_dict(), "bert_encoder.pt")
    print("Model saved as bert_encoder.pt")
    
    # Save loss values
    np.savez('bert_training_history.npz', 
             train_losses=np.array(train_losses), 
             val_losses=np.array(val_losses))
    print("Training history saved as bert_training_history.npz")