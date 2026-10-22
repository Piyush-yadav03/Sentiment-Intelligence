import torch
import torch.nn as nn

class TinyTransformerStudent(nn.Module):
    def __init__(self, vocab_size, pad_id, max_length=96, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 256, dropout=0.1,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask):
        n, length = input_ids.shape
        if length > self.max_length:
            input_ids, attention_mask = input_ids[:, :self.max_length], attention_mask[:, :self.max_length]
            length = self.max_length
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(self.dropout(self.norm(pooled)))
