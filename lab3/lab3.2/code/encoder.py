import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BERT-STYLE MODEL ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        '''
        TODO: Implement multi-head self-attention
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
        '''
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # projection for the output
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, mask=None):
        '''
        TODO: Implement forward pass for multi-head self-attention
        Args:
            x: Input
            mask: Attention mask
        '''
        batch_size = x.size(0)
        
        # by computing projections
        q = self.query(x)  
        k = self.key(x)    
        v = self.value(x)  
        
        # Reshaping for attention of multi head
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        
        # by Computing attention scores
        scores = torch.matmul(q, k.transpose(-1, -2))  
        scores = scores / (self.head_dim ** 0.5)  # Scaling attention scores
        
        # Apply mask if and only is provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # by applying softmax
        attn_weights = F.softmax(scores, dim=-1)  
        
        # Apply attention to the values
        context = torch.matmul(attn_weights, v)  
        
        # Reshaping and combining heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)  
        
        # projection of output and it is final
        output = self.output(context)  
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        '''
        TODO: Implement feed-forward network
        Args:
            hidden_size: Hidden size of the model
            intermediate_size: Intermediate size of the model
        '''
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        '''
        TODO: Implement forward pass for transformer block
        Args:
            x: Input
            mask: Attention mask
        '''
        # Self-attention with the residuals connections and normalization at layer level
        attn_output = self.attn(x, mask)
        assert x.shape == attn_output.shape, f"Mismatch: x {x.shape}, attn_output {attn_output.shape}"
        attn_output = self.attn(x, mask)
        x = self.ln1(x + attn_output)
        
        # Feed-forward network with residual connection and normalization at layer level
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, num_layers=4,
                 intermediate_size=512, max_len=128):
        '''
        TODO: Implement encoder
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            num_layers: Number of layers
            intermediate_size: Intermediate size of the model
            max_len: Maximum length of the input
        '''
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.type_emb = nn.Embedding(2, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        assert input_ids.shape == token_type_ids.shape == attention_mask.shape, "Shape mismatch in inputs!"
        seq_length = input_ids.size(1)
        
        # Create position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Geting embeddings
        token_embeddings = self.token_emb(input_ids)
        position_embeddings = self.pos_emb(position_ids)
        token_type_embeddings = self.type_emb(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        
        # Pass through layers of transformer
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer of  normalization
        hidden_states = self.norm(hidden_states)
        
        # MLM prediction head
        prediction_scores = self.mlm_head(hidden_states)
        
        return prediction_scores, hidden_states