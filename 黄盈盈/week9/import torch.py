import torch
import torch.nn as nn
import math
from torch.nn import Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_embedding', pos_embedding)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, 
                 nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.transformer = Transformer(
            d_model=emb_size, nhead=nhead, 
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
    
    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)
    
    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)
    
    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


        from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

class Seq2SeqDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = self.build_vocab(src_data, ["PAD", "EOS", "UNK"])
        self.tgt_vocab = self.build_vocab(tgt_data, ["PAD", "BOS", "EOS", "UNK"])
    
    def __getitem__(self, idx):
        src_seq = list(self.src_data[idx]) + ["EOS"]
        tgt_seq = ["BOS"] + list(self.tgt_data[idx]) + ["EOS"]
        src_ids = [self.src_vocab.get(tk, self.src_vocab["UNK"]) for tk in src_seq]
        tgt_ids = [self.tgt_vocab.get(tk, self.tgt_vocab["UNK"]) for tk in tgt_seq]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)
    
    def __len__(self):
        return len(self.src_data)
    
    def build_vocab(self, data, special_tokens):
        vocab = OrderedDict({token: idx for idx, token in enumerate(special_tokens)})
        for seq in data:
            for token in set(seq):
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_batch, tgt_batch

# 超参数配置
NUM_EPOCHS = 10
BATCH_SIZE = 32
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设已有训练数据 src_data 和 tgt_data（需替换为实际数据）
dataset = Seq2SeqDataset(src_data, tgt_data)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

model = Seq2SeqTransformer(
    num_encoder_layers=3, num_decoder_layers=3, emb_size=EMB_SIZE, 
    nhead=NHEAD, src_vocab_size=len(dataset.src_vocab), 
    tgt_vocab_size=len(dataset.tgt_vocab)
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD的损失

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        src_mask, tgt_mask = generate_square_subsequent_mask(src.size(0)), generate_square_subsequent_mask(tgt_input.size(0))
        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt_input == 0).transpose(0, 1)
        
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# 保存模型
torch.save(model.state_dict(), "transformer_model.pth")