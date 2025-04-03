import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter  # 这是关键导入
import numpy as np

import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# 加载数据
weather_data = pd.read_csv('weather_history.csv')  # 假设已经从Kaggle下载
weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data = weather_data.sort_values('date')

# 选择最高气温作为预测目标
temps = weather_data[['max_temp']].values

# 数据标准化
scaler = MinMaxScaler()
temps_scaled = scaler.fit_transform(temps)

# 创建时间序列数据集
def create_dataset(data, look_back=30, look_forward=1):
    X, y = [], []
    for i in range(len(data)-look_back-look_forward+1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[(i+look_back):(i+look_back+look_forward), 0])
    return np.array(X), np.array(y)

# 创建1天和5天预测的数据集
X_1, y_1 = create_dataset(temps_scaled, look_back=30, look_forward=1)
X_5, y_5 = create_dataset(temps_scaled, look_back=30, look_forward=5)

# 划分训练集和测试集
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, shuffle=False)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5, y_5, test_size=0.2, shuffle=False)

# 创建Dataset类
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(-1)  # 添加特征维度
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建DataLoader
train_loader_1 = DataLoader(WeatherDataset(X_train_1, y_train_1), batch_size=32, shuffle=True)
test_loader_1 = DataLoader(WeatherDataset(X_test_1, y_test_1), batch_size=32, shuffle=False)
train_loader_5 = DataLoader(WeatherDataset(X_train_5, y_train_5), batch_size=32, shuffle=True)
test_loader_5 = DataLoader(WeatherDataset(X_test_5, y_test_5), batch_size=32, shuffle=False)


class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(WeatherRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 对于多步预测，我们只取最后output_size个时间步的输出
        out = out[:, -self.fc.out_features:, :]
        out = self.fc(out)
        return out.squeeze(-1) if output_size == 1 else out
    
    def train_weather_model(model, train_loader, test_loader, num_epochs=50, lr=0.001, model_name='weather_1day'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # TensorBoard记录
    writer = SummaryWriter(f'runs/{model_name}')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                writer.add_scalar('training_loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0
        
        # 测试集评估
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        
        test_loss /= len(test_loader)
        writer.add_scalar('test_loss', test_loss, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.6f}')
    
    writer.close()
    return model

# 1天预测模型
model_1day = WeatherRNN(output_size=1)
train_weather_model(model_1day, train_loader_1, test_loader_1, model_name='weather_1day')

# 5天预测模型
model_5day = WeatherRNN(output_size=5)
train_weather_model(model_5day, train_loader_5, test_loader_5, model_name='weather_5day')