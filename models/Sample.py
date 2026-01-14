import torch 
import torch.nn as nn
import torch.nn.functional as F 

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = config.channels
        self.patch_length = config.period_len
        
        self.patch_num = self.seq_len // self.patch_length
        self.output_patch_num = self.pred_len // self.patch_length

        self.decomposition = series_decomp(kernel_size=25)
        self.seasonal_fc1 = nn.Linear(self.patch_num, self.output_patch_num)
        self.trend_fc1 = nn.Linear(self.patch_num, self.output_patch_num)

        
    def forward(self, x): # x: [batch_size, seq_len, features]
        B, L, C = x.shape
        
        # 정규화 
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        
        # 분해
        seasonal, trend = self.decomposition(x)
        
        # Seasonal
        # 주기(계절성)은 주기가 확실한 경우 구간마다 비슷한 패턴이 반복된다고 가정. 
        # 그렇다면 시계열 데이터 모두를 쓸 필요없이 하나의 구간으로 요약하면 어떨까?
        seasonal = seasonal.reshape(-1, self.patch_length, self.patch_num) # [batch_size * features, num_patches, patch_length]
        seasonal = self.seasonal_fc1(seasonal) # [batch_size * features, patch_length, output_patch_num]
        
        # Trend
        trend = trend.reshape(-1, self.patch_length, self.patch_num)
        trend = self.trend_fc1(trend) # [batch_size * features, patch_length, output_patch_num]
        
        # Combine
        output = seasonal + trend
        
        # Reshape back to [B, Pred_Len, C]
        output = output.permute(0, 2, 1).reshape(B, C, -1) # [B, C, Pred_Len]
        output = output.permute(0, 2, 1) # [B, Pred_Len, C]
        
        output = output + seq_mean
        
        return output