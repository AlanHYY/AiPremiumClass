from sklearn.datasets import fetch_olivetti_faces
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# 加载数据
faces = fetch_olivetti_faces()
X = faces.data.reshape(-1, 64, 64, 1)  # 64x64图像
y = faces.target

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建Dataset类
class FaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建DataLoader
train_loader = DataLoader(FaceDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(FaceDataset(X_test, y_test), batch_size=32, shuffle=False)

import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, rnn_type='RNN', input_size=64, hidden_size=128, num_layers=2, num_classes=40):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 选择RNN类型
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'BiRNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            hidden_size *= 2  # 双向RNN输出维度翻倍
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 输入形状: (batch_size, 64, 64, 1) -> 调整为 (batch_size, 64, 64)
        x = x.squeeze(-1)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * (2 if 'Bi' in self.rnn_type else 1), x.size(0), self.hidden_size).to(x.device)
        
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros_like(h0)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

def train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001, model_name='RNN'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # TensorBoard记录
    writer = SummaryWriter(f'runs/face_classification_{model_name}')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 9:
                writer.add_scalar('training_loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0
        
        # 每个epoch记录训练准确率
        train_acc = 100 * correct / total
        writer.add_scalar('training_accuracy', train_acc, epoch)
        
        # 测试集评估
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        writer.add_scalar('test_accuracy', test_acc, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    writer.close()
    return model

# 训练不同模型
models = {
    'RNN': RNNModel(rnn_type='RNN'),
    'LSTM': RNNModel(rnn_type='LSTM'),
    'GRU': RNNModel(rnn_type='GRU'),
    'BiRNN': RNNModel(rnn_type='BiRNN')
}

for name, model in models.items():
    print(f"Training {name} model...")
    train_model(model, train_loader, test_loader, num_epochs=20, model_name=name)