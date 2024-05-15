# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, withMask=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.applyMask = withMask
        if(withMask):
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        if(self.applyMask):
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out, wei

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, isDecoder):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size, n_embd=n_embd, block_size= block_size, withMask=isDecoder) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
  

    def forward(self, x):
        out,attension_map = zip(*[h(x) for h in self.heads])
        out = self.proj(torch.cat(out, dim=-1))
        return out, attension_map

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, isDecoder):
        super().__init__()
        hidden_size = 100
        if(isDecoder == False):
            hidden_size = 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_embd),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, isDecoder):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, block_size=block_size, isDecoder=isDecoder)
        self.ffwd = FeedFoward(n_embd, isDecoder = isDecoder)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x_after_head, attension_map =self.sa(x)
        x = x + self.ln1(x_after_head)
        x = x + self.ln2(self.ffwd(x))
        return x, attension_map

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, device, isDecoder = False):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding= nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, isDecoder) for _ in range(n_layer)])
        self.block_size = block_size
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.isDecoder = isDecoder

   
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        attention_maps = []
        
        for block in self.blocks:
            x, att_map = block(x)
            attention_maps = attention_maps + list(att_map)
        
        x = self.ln_f(x)
        if(self.isDecoder):
            x = self.lm_head(x)
            return x, attention_maps
        else:
            return torch.mean(x, dim=-2), attention_maps
    
    # def generate(self, idx, max_new_tokens):
    #     # idx is (B, T) array of indices in the current context
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = idx[:, -self.block_size:]
    #         # get the predictions
    #         logits, loss = self(idx_cond)
    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         # append sampled index to the running sequence
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx   

