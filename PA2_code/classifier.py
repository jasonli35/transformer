from transformer import Encoder
import torch.nn as nn

class FeedForwardClassifier(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, device, ff_input, ff_n_hidden, ff_n_output):
        super().__init__()
        self.encoder = Encoder(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_layer=n_layer, n_head=n_head, device=device)
        self.net = nn.Sequential(
            nn.Linear(ff_input, ff_n_hidden),
            nn.ReLU(),
            nn.Linear(ff_n_hidden, ff_n_output),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        
    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.net(x)
        return self.log_softmax(x)


