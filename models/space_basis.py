import torch.nn as nn
import torch
import numpy as np
from models.utils import activation_func

def fourier_basis_output(x, num_basis,base):
    # x 是输入，num_basis 是基函数的数量
    outputs = []
    # 生成傅里叶基函数的 sin 和 cos 项
    for k in range(base, num_basis + base):
        sin_kx = torch.sin(np.pi*k * x)  # 计算 sin(kx)
        cos_kx = torch.cos(np.pi*k * x)  # 计算 cos(kx)
        outputs.append(sin_kx)
        outputs.append(cos_kx)
    
    # 堆叠基函数时沿着第一个维度 (dim=1) 堆叠，得到形状 (50, 20)
    return torch.cat(outputs, dim=1)

class FNN(nn.Module):       
    def __init__(self, 
                 device,            # Device
                 data_dim,          # Dimension of the problem
                 dim_u,             # Dimension of the R^{du} space
                 hidden_dim,        # Dimension of the hidden layers
                 hidden_layers,     # Number of hidden layers
                 non_linearity,     # activation function type
                 option=None): # Basis function type
        super(FNN, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.dim_u = dim_u
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.non_linearity = activation_func(non_linearity)
        self.option = option
    
        # Define the neural network
        layers = []
        prev_size = data_dim
        for i in range(hidden_layers-1):
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(self.non_linearity)
            prev_size = hidden_dim
        layers.append(nn.Linear(prev_size, dim_u, bias=False))  # Final output layer
        self.fc_NN = nn.Sequential(*layers).to(device)

        # Initialize Xavier weights and zero biases
        for block in self.fc_NN:
            if isinstance(block, nn.Linear):
                nn.init.xavier_normal_(block.weight)
                if block.bias is not None:
                    nn.init.zeros_(block.bias)
    
    def forward(self, x):
        output = self.fc_NN(x)

        if self.option == 'Fourier':
            fourier_features = fourier_basis_output(x, num_basis=int(self.dim_u/2),base=1)
            output = output + fourier_features

        return output

class FNN_Time(nn.Module):
    def __init__(self,
                 device,            # Device
                 data_dim,          # Dimension of the problem
                 dim_u,             # Dimension of the R^{du} space
                 hidden_dim,        # Dimension of the hidden layers
                 hidden_layers,     # Number of hidden layers
                 non_linearity,     # activation function type
                 time_steps):       # Number of time steps
            super(FNN_Time, self).__init__()
            self.device = device
            self.data_dim = data_dim
            self.dim_u = dim_u
            self.hidden_dim = hidden_dim
            self.hidden_layers = hidden_layers
            self.non_linearity = activation_func(non_linearity)
            self.time_steps = time_steps
        
            # Define the neural network
            blocks = [FNN(device, data_dim, dim_u, hidden_dim, hidden_layers, non_linearity)
                for i in range(time_steps)]
            self.fnn_time = nn.Sequential(*blocks).to(device)
        
    def forward(self, k, x):
        return self.fnn_time[k](x)

class CNN(nn.Module):
    def __init__(self, device, Nx, dim_u, hidden_layers, non_linearity):
        """
        参数说明：
         - Nx: 输入矩阵的边长 Nx, 输入尺寸为 Nx x Nx
         - dim_u: 每个空间位置最终输出的向量维度
         - hidden_layers: 卷积层的层数（不改变空间尺寸）
        """
        super(CNN, self).__init__()
        self.device = device
        self.Nx = Nx
        self.dim_u = dim_u
        self.hidden_layers = hidden_layers
        self.non_linearity = activation_func(non_linearity)
        
        layers = []
        # 注意：由于每个位置有 2 个特征，输入通道应设为 2
        in_channels = 2  
        # 初始隐藏通道数设为 32
        hidden_channels = 16  
        for i in range(hidden_layers):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=1))
            layers.append(self.non_linearity)
            # 每一层后更新 in_channels，并将通道数翻倍
            in_channels = hidden_channels
            hidden_channels *= 2  
        # 最后用1×1卷积将最后一层的通道映射到目标 dim_u
        layers.append(nn.Conv2d(in_channels, dim_u, kernel_size=1))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        假设输入 x 的尺寸为 [Nx*Nx, 2]（单个样本）
        最终希望输出尺寸为 [Nx*Nx, dim_u]
        """
        
        x = x.view(self.Nx, self.Nx, 2).permute(2, 0, 1).unsqueeze(0)
        
        # 现在 x 的尺寸为 [batch, 2, Nx, Nx]
        x = self.conv(x)  
        # 将 x 重排为 [Nx*Nx, dim_u]
        x = x.permute(0, 2, 3, 1).reshape(self.Nx * self.Nx, self.dim_u)

        return x