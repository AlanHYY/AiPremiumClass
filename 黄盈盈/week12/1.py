%%writefile ner_ddp_train.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 实体映射
entites = ['O'] + list({'movie', 'name', 'game', 'address', 'position', 
                        'company', 'scene', 'book', 'organization', 'government'})
tags = ['O']
for entity in entites[1:]:
    tags.append('B-' + entity.upper())
    tags.append('I-' + entity.upper())

entity_index = {entity: i for i, entity in enumerate(entites)}
id2lbl = {i: tag for i, tag in enumerate(tags)}
lbl2id = {tag: i for i, tag in enumerate(tags)}

# 数据处理函数
def process_dataset(ds):
    def entity_tags_proc(item):
        text_len = len(item['text'])
        tags = [0] * text_len
        for ent in item['ents']:
            indices = ent['indices']
            label = ent['label']
            tags[indices[0]] = entity_index[label] * 2 - 1
            for idx in indices[1:]:
                tags[idx] = entity_index[label] * 2
        return {'ent_tag': tags}
    
    ds1 = ds.map(entity_tags_proc)
    
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    
    def data_input_proc(item):
        batch_texts = [list(text) for text in item['text']]
        input_data = tokenizer(
            batch_texts, 
            truncation=True, 
            add_special_tokens=False, 
            max_length=512,
            is_split_into_words=True,
            padding='max_length'
        )
        input_data['labels'] = [tag + [0] * (512 - len(tag)) for tag in item['ent_tag']]
        return input_data
    
    return ds1.map(data_input_proc, batched=True), tokenizer

# 训练函数
def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 加载和处理数据
    ds = load_dataset('nlhappy/CLUE-NER')
    ds2, tokenizer = process_dataset(ds)
    
    # 创建模型
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese',
        num_labels=len(tags),
        id2label=id2lbl,
        label2id=lbl2id
    )
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 数据加载器
    train_sampler = DistributedSampler(
        ds2['train'], 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    train_dl = DataLoader(
        ds2['train'], 
        batch_size=16,
        sampler=train_sampler,
        pin_memory=True
    )
    
    # 优化器和学习率调度器
    param_optimizer = list(ddp_model.named_parameters())
    bert_params, classifier_params = [], []
    
    for name, param in param_optimizer:
        if 'bert' in name:
            bert_params.append(param)
        else:
            classifier_params.append(param)
    
    param_groups = [
        {'params': bert_params, 'lr': 1e-5},
        {'params': classifier_params, 'weight_decay': 0.1, 'lr': 1e-3}
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    train_steps = len(train_dl) * 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100,
        num_training_steps=train_steps
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练循环
    for epoch in range(5):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        
        tpbar = tqdm(train_dl, desc=f"Rank {rank} Epoch {epoch+1}", disable=rank != 0)
        
        for batch in tpbar:
            batch = {k: v.to(rank) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = ddp_model(**batch)
                loss = outputs.loss
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if rank == 0:
                tpbar.set_postfix({
                    'loss': loss.item(),
                    'bert_lr': scheduler.get_lr()[0],
                    'classifier_lr': scheduler.get_lr()[1]
                })
    
    # 保存模型（只在主进程）
    if rank == 0:
        torch.save(model.state_dict(), "ner_model.pt")
        tokenizer.save_pretrained("ner_tokenizer")
        print("Model saved")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Starting DDP training with {world_size} GPUs")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


    # ner_inference.py
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# 加载保存的模型和tokenizer
model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-chinese",
    num_labels=21
)
model.load_state_dict(torch.load("ner_model.pt"))
tokenizer = AutoTokenizer.from_pretrained("ner_tokenizer")

# 实体标签映射
tags = [
    'O', 'B-POSITION', 'I-POSITION', 'B-NAME', 'I-NAME', 
    'B-GOVERNMENT', 'I-GOVERNMENT', 'B-MOVIE', 'I-MOVIE', 
    'B-BOOK', 'I-BOOK', 'B-ORGANIZATION', 'I-ORGANIZATION', 
    'B-COMPANY', 'I-COMPANY', 'B-ADDRESS', 'I-ADDRESS', 
    'B-SCENE', 'I-SCENE', 'B-GAME', 'I-GAME'
]

def predict_entities(text):
    # 预处理输入
    inputs = tokenizer(
        list(text),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # 模型预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    
    # 对齐预测结果
    word_ids = inputs.word_ids()
    aligned_predictions = []
    current_word = None
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None:  # 特殊token
            continue
        if word_id != current_word:
            current_word = word_id
            aligned_predictions.append(predictions[idx])
    
    # 提取实体
    entities = []
    current_entity = None
    current_start = None
    
    for i, tag_idx in enumerate(aligned_predictions):
        tag = tags[tag_idx]
        
        if tag.startswith('B-'):
            if current_entity:
                entities.append({
                    'entity': current_entity,
                    'text': text[current_start:i],
                    'start': current_start,
                    'end': i
                })
            current_entity = tag[2:]
            current_start = i
        
        elif tag.startswith('I-') and current_entity == tag[2:]:
            continue  # 继续当前实体
        
        else:
            if current_entity:
                entities.append({
                    'entity': current_entity,
                    'text': text[current_start:i],
                    'start': current_start,
                    'end': i
                })
                current_entity = None
                current_start = None
    
    # 处理最后一个实体
    if current_entity:
        entities.append({
            'entity': current_entity,
            'text': text[current_start:],
            'start': current_start,
            'end': len(text)
        })
    
    return entities

# 测试推理
if __name__ == "__main__":
    test_text = "张艺谋导演的红高粱在山东高密拍摄"
    results = predict_entities(test_text)
    
    print("文本:", test_text)
    print("识别结果:")
    for entity in results:
        print(f"- {entity['text']} ({entity['entity']})")