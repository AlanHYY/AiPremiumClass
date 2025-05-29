from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    pipeline
)
from datasets import load_dataset
import torch
import numpy as np
from seqeval.metrics import classification_report

# 1. 加载数据集
dataset = load_dataset("doushabao4766/msra_ner_k_V3")

# 2. 定义标签映射
label_list = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}

# 3. 加载预训练模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 4. 数据预处理函数
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 5. 处理数据集
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="./ner_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 7. 数据收集器
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# 8. 定义评估函数
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions)
    return {
        "precision": report.split("\n")[-4].split()[1],
        "recall": report.split("\n")[-4].split()[2],
        "f1": report.split("\n")[-4].split()[3],
        "accuracy": report.split("\n")[-3].split()[1]
    }

# 9. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 10. 训练模型
trainer.train()

# 11. 保存模型
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

# 12. 实体预测函数
def predict_entities(text):
    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    
    entities = []
    results = nlp(text)
    
    for entity in results:
        if entity["entity_group"] != "O":
            entities.append({
                "entity": entity["entity_group"].split("-")[-1],
                "content": entity["word"]
            })
    
    return entities

# 13. 测试预测
test_text = "双方确定了今后发展中美关系的指导方针。"
print(predict_entities(test_text))


[
  {"entity": "ORG", "content": "中"},
  {"entity": "ORG", "content": "美"}
]





