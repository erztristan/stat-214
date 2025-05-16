import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import random
import numpy as np


def split_based_on_flags(data, stories, p, min_len, max_len):
    sentences = []
    for story in stories:
        words = data[story].data
        flags = np.full(len(words), False)
        pos = 0

        for i in range(len(words) - 1):
            # If the sentence exceeds max_len, cut the sentence
            if i - pos > max_len:
                flags[i] = True
                pos = i

            # If the pause between words is long and the current sentence exceeds min_len,
            # Cut the sentence
            elif np.diff(data[story].data_times)[i] >= p and i - pos > min_len:
                flags[i] = True
                pos = i

        current_sentence = []

        for word, flag in zip(words, flags):
            current_sentence.append(word)  # Add words to the current sentence
            if flag:
                sentences.append(current_sentence)  # Add current sentence to the list
                current_sentence = []  # Create new sentence

        if current_sentence: # Last sentence
            sentences.append(current_sentence)

            # Make sure that last sentence should have enough length
            while len(sentences[-1]) < min_len: 
                sentences[-2] = sentences[-2] + sentences[-1]
                sentences.pop()

    return sentences

# --- DATASET ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        '''
        Args:
            texts: List of stories
            tokenizer: Tokenizer
            max_len: Maximum length of the story # THIS IS JUST AN EXAMPLE
        '''
        self.encodings = tokenizer(
            texts,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx].unsqueeze(0).unsqueeze(0)
        }

# --- MAIN EXECUTION ---
def load_data(texts = None, n_batch = 2, max_length = 128, seed = 42):
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")  # or your trained tokenizer path

    # Example dataset
    if texts is None:
        texts = [
            "The quick brown dfox jumps over the lazy dog.",
            "Transformers are powerful models for NLP tasks.",
            "Masked language modeling trains BERT to understand context.",
            "Pretraining is followed by task-specific fine-tuning."
        ]

    # Set seeds for shuffling the data
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = TextDataset(texts, tokenizer, max_len = max_length)
    dataloader = DataLoader(dataset, batch_size = n_batch, shuffle = True, drop_last = True)

    # Check the imput
    print("length of batch: ", len(dataloader))
    for i, batch in enumerate(dataloader):
        if i < 2:
            for j in range(batch["input_ids"].size(0)):
                print(f"Sample {i * dataloader.batch_size + j}:")
                print("  Text:", tokenizer.decode(batch["input_ids"][j], skip_special_tokens=False))
                #print("  Token Type:", tokenizer.decode(batch["token_type_ids"][j], skip_special_tokens=False))
                print("  Attention mask:", batch["attention_mask"][j].tolist())
                print(batch["input_ids"][j])
                print()

    return dataloader
