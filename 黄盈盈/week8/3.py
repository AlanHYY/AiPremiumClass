from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# 初始化模型
encoder = Encoder(len(vocab), 256, 512, mode='add')
decoder = Decoder(len(vocab), 256, 512)
optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 训练循环
for epoch in range(10):
    total_loss = 0
    for src, tgt in dataloader:
        encoder_outputs, hidden = encoder(src, src_lengths)
        loss = 0
        decoder_input = tgt[0]  # 初始输入为<SOS>
        for t in range(1, tgt.size(0)):
            output, hidden = decoder(decoder_input, hidden, encoder_outputs)
            loss += criterion(output, tgt[t])
            decoder_input = tgt[t] if random.random()<0.5 else output.argmax(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
        def inference(encoder, decoder, input_seq, max_len=20):
    with torch.no_grad():
        input_seq = [word2idx.get(word, 3) for word in tokenize(input_seq)]  # 3是<UNK>
        input_tensor = torch.LongTensor(input_seq).unsqueeze(1)
        lengths = [len(input_seq)]
        encoder_outputs, hidden = encoder(input_tensor, lengths)
        decoded_words = []
        decoder_input = torch.LongTensor([[1]])  # <SOS>
        for _ in range(max_len):
            output, hidden = decoder(decoder_input, hidden, encoder_outputs)
            topi = output.argmax(1)
            if topi.item() == 2:  # <EOS>
                break
            decoded_words.append(vocab[topi.item()])
            decoder_input = topi.unsqueeze(0)
        return ''.join(decoded_words)

# 测试
print(inference(encoder, decoder, "春风绿柳迎风舞"))
# 输出示例：丽日红花映日开