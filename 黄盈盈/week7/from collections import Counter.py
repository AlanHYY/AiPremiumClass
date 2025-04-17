from collections import Counter

# 统计所有词汇
all_tokens = [token for tokens in train_df["tokens"] for token in tokens]
vocab_counter = Counter(all_tokens)

# 保留前N个高频词
vocab_size = 10000
vocab = ["PAD", "UNK"] + [word for word, _ in vocab_counter.most_common(vocab_size)]

# 词汇表映射
word2idx = {word: idx for idx, word in enumerate(vocab)}

def text_to_indices(tokens):
    return [word2idx.get(token, 1) for token in tokens]  # UNK索引为1

train_df["indices"] = train_df["tokens"].apply(text_to_indices)
test_df["indices"] = test_df["tokens"].apply(text_to_indices)