from datasets import load_dataset

# 从Kaggle下载并加载数据集
dataset = load_dataset('csv', data_files='jd_comments.csv')

# 数据预处理示例
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_data = dataset.map(preprocess_function, batched=True)


from transformers import AutoModelForSequenceClassification

# 加载基础模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

# 冻结BERT参数（仅训练分类头）
# for param in model.bert.parameters():
#     param.requires_grad = False

from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

# 定义评估指标
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",  # TensorBoard日志目录
    report_to="tensorboard"
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    compute_metrics=compute_metrics
)

# 启动训练
trainer.train()

# 启动TensorBoard
%load_ext tensorboard
%tensorboard --logdir ./logs


# 保存模型
model.save_pretrained("./jd_classifier")

# 加载模型进行预测
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./jd_classifier",
    tokenizer=tokenizer
)

sample_text = "商品质量非常好，快递速度很快"
result = classifier(sample_text)
print(f"预测结果：{result[0]['label']} (置信度：{result[0]['score']:.4f})")