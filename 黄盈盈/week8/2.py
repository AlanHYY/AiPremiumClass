import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, mode='add'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=True)
        self.mode = mode  # 'add'或'concat'

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # 处理双向hidden state
        if self.mode == 'add':
            hidden = hidden[0] + hidden[1]
        elif self.mode == 'concat':
            hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        # 计算attention分数
        energy = torch.tanh(self.W(decoder_hidden + encoder_outputs))
        attn_scores = torch.sum(energy, dim=2)
        return torch.softmax(attn_scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_dim + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size*2, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        # 计算attention权重
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1))
        # 拼接输入和上下文
        gru_input = torch.cat([embedded, context.squeeze(1)], dim=1)
        output, hidden = self.gru(gru_input.unsqueeze(0), hidden)
        output = self.out(torch.cat([output.squeeze(0), context.squeeze(1)], dim=1))
        return output, hidden
    
    