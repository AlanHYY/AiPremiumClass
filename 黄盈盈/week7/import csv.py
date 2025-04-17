import csv
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv("data/comments.csv", encoding="utf-8")

# 根据评分划分标签
df["label"] = df["votes"].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else -1))
df = df[df["label"] != -1]  # 删除评分3的样本

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def tokenize(text):
    # 使用jieba分词
    return jieba.lcut(text)

# 示例分词
train_df["tokens"] = train_df["content"].apply(tokenize)
test_df["tokens"] = test_df["content"].apply(tokenize)

