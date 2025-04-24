import jieba
import pandas as pd
from collections import defaultdict

# 加载数据集
df = pd.read_csv("chinese-couplets/train.csv", header=None, names=["上联", "下联"])

# 分词函数
def tokenize(text):
    return list(jieba.cut(text.strip()) )

# 构建词汇表
word_freq = defaultdict(int)
for line in df["上联"] + df["下联"]:
    for word in tokenize(line):
        word_freq[word] += 1

# 过滤低频词并生成词表
min_freq = 3
vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + \
        [word for word, freq in word_freq.items() if freq >= min_freq]
word2idx = {word: idx for idx, word in enumerate(vocab)}

# 保存词表
import pickle
pickle.dump(word2idx, open("vocab.pkl", "wb"))