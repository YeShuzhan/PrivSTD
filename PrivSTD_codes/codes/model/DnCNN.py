import torch
import torch.nn as nn
import os
import json
# class DnCNN(nn.Module):
#     def __init__(self, config, channels, num_of_layers=17):
#         super(DnCNN, self).__init__()
#         self.config = config
#         kernel_size = 3
#         padding = 1
#         features = 64
#         layers = []
#         layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(num_of_layers-2):
#             layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
#             layers.append(nn.BatchNorm2d(features))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
#         self.dncnn = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.dncnn(x)
#         return x-out

#     def save_model(self, res_str, save_path):
#         save_name = save_path + '/{}_img{}_w{}_h{}_e{}_d{}'.format(self.config['train']['model'],
#             self.config['net']['img_size'], self.config['net']['window_size'],
#             self.config['net']['num_heads'], self.config['net']['embed_dim'],
#             self.config['net']['drop_rate']
#         )
#         os.makedirs(os.path.dirname(save_name), exist_ok=True)
#         torch.save(self.dncnn.state_dict(), save_name + '.pth')

#         res = self.config
#         res['result'] = res_str
#         # save config
#         with open(save_name + '.json', 'w') as f:
#             json.dump(res, f)



class DnCNN(nn.Module):
    def __init__(self, config, channels, num_of_layers=17,
                 dropout_p=None,        # 若为 None，则从 config['net']['drop_rate'] 读取；否则以此为准
                 dropout_start=4,       # 从第几个中间卷积块开始加（以卷积层编号计，见下文）
                 dropout_end=None,      # 到哪个卷积块结束（含），默认到倒数第二个卷积块
                 spatial=True):         # True=Dropout2d（推荐），False=普通 Dropout
        super(DnCNN, self).__init__()
        self.config = config
        
        # 读取/确定 p
        if dropout_p is None:
            self.dropout_p = float(config.get('net', {}).get('drop_rate', 0.0))
        else:
            self.dropout_p = float(dropout_p)

        kernel_size = 3
        padding = 1
        features = 64

        # 末端之前一共有 num_of_layers-1 个卷积块（第1块是首层卷积）
        # 我们给“中间块”的 ReLU 后插入 Dropout
        if dropout_end is None:
            dropout_end = num_of_layers - 1  # 不包含最后输出卷积层

        DropCls = nn.Dropout2d if spatial else nn.Dropout
        use_do = self.dropout_p > 0.0

        layers = []
        # conv1 + relu（不放 Dropout）
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间块：conv_i + BN + ReLU (+ Dropout)
        # 这里 i 表示“第 i 个卷积层”，i=2..(num_of_layers-1)
        for i in range(2, num_of_layers):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            if use_do and (dropout_start <= i <= dropout_end):
                layers.append(DropCls(p=self.dropout_p))

        # 最后一层输出卷积（不放 Dropout/BN/激活）
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # 残差学习：预测噪声

    def save_model(self, res_str, save_path):
        save_name = save_path + '/{}_img{}_w{}_h{}_e{}_d{}'.format(
            self.config['train']['model'],
            self.config['net']['img_size'], self.config['net']['window_size'],
            self.config['net']['num_heads'], self.config['net']['embed_dim'],
            self.config['net'].get('drop_rate', self.dropout_p)  # 兼容外部配置/本类参数
        )
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        torch.save(self.dncnn.state_dict(), save_name + '.pth')

        res = dict(self.config)
        res['result'] = res_str
        with open(save_name + '.json', 'w') as f:
            json.dump(res, f)


# 手动实现简单 CBAM (Channel + Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Channel attention
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x, std_map=None):
        # x: [B, F, H, W] (features)
        B, F, H, W = x.shape
        
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).squeeze(-1).squeeze(-1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).squeeze(-1).squeeze(-1))))
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)  # [B, F, 1, 1]
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]  # [B,1,H,W]
        spatial_in = torch.cat([avg_spatial, max_spatial], dim=1)  # [B,2,H,W]
        spatial_att = self.sigmoid(self.conv_spatial(spatial_in))  # [B,1,H,W]
        
        # 显式利用 std_map (var_map sqrt) 做 weighting：高 std (高噪声) 降低 attention
        if std_map is not None:
            # std_map: [B,1,H,W]，假设已 resize 到当前 H,W
            weight = 1.0 / (std_map + 1e-6)  # 高噪声区权重低
            spatial_att = spatial_att * weight  # multiply 到 att map
        
        x = x * spatial_att
        return x

class DnCNN_two(nn.Module):
    def __init__(self, config, channels=2, num_of_layers=17,
                 dropout_p=None,        # 若为 None，则从 config['net']['drop_rate'] 读取；否则以此为准
                 dropout_start=8,       # 从第几个中间卷积块开始加（以卷积层编号计，见下文）
                 dropout_end=None,      # 到哪个卷积块结束（含），默认到倒数第二个卷积块
                 spatial=True,          # True=Dropout2d（推荐），False=普通 Dropout
                 attention_interval=4): # 每隔多少层插入一个 CBAM
        super(DnCNN_two, self).__init__()
        self.config = config
        
        # 读取/确定 p（减少到0.05作为默认建议值，如果config有则覆盖）
        if dropout_p is None:
            self.dropout_p = float(config.get('net', {}).get('drop_rate', 0.05))
        else:
            self.dropout_p = float(dropout_p)

        kernel_size = 3
        padding = 1
        features = 64

        # 末端之前一共有 num_of_layers-1 个卷积块（第1块是首层卷积）
        # 我们给“中间块”的 ReLU 后插入 Dropout
        if dropout_end is None:
            dropout_end = num_of_layers - 1  # 不包含最后输出卷积层

        DropCls = nn.Dropout2d if spatial else nn.Dropout
        use_do = self.dropout_p > 0.0

        layers = []
        # conv1 + relu（不放 Dropout）
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间块：conv_i + BN + ReLU (+ Dropout) (+ CBAM 每隔 interval)
        # 这里 i 表示“第 i 个卷积层”，i=2..(num_of_layers-1)
        for i in range(2, num_of_layers):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            if use_do and (dropout_start <= i <= dropout_end):
                layers.append(DropCls(p=self.dropout_p))
            
            # 插入 CBAM，每隔 attention_interval 层（e.g., 4）
            if (i - 1) % attention_interval == 0 and i > 2:
                layers.append(CBAM(features))

        # 最后一层输出卷积（不放 Dropout/BN/激活），输出通道改为1（只预测噪声 for noisy 通道）
        layers.append(nn.Conv2d(in_channels=features, out_channels=1,
                                kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B,2,H,W]，第0通道=noisy，第1通道=std_map
        noisy = x[:, 0:1, :, :]  # 提取 noisy 通道作为残差基底
        std_map = x[:, 1:2, :, :]  # 提取 std_map，用于 CBAM weighting
        
        # 在 forward 中运行 sequential，但 CBAM 需要 std_map
        # 注意：由于 sequential 是 list of modules，我们需在 forward 中手动传递 std_map 到 CBAM
        out = x  # 输入全 [B,2,H,W]，但后续层从 features 开始
        for module in self.dncnn:
            if isinstance(module, CBAM):
                out = module(out, std_map)  # 传递 std_map 到 CBAM
            else:
                out = module(out)
        
        return noisy - out       # 残差学习：去噪 = noisy - 噪声估计

    def save_model(self, res_str, save_path):
        # 原 save_name 有无关参数如 num_heads, embed_dim（像是 Transformer 的），这里修正为 DnCNN 相关
        # 假设这些是 config 中的遗留，保留但注释潜在问题
        save_name = save_path + '/{}_img{}_w{}_h{}_e{}_d{}'.format(
            self.config['train']['model'],
            self.config['net']['img_size'], self.config['net']['window_size'],
            self.config['net'].get('num_heads', 'N/A'),  # 如果无，设 N/A
            self.config['net'].get('embed_dim', 'N/A'),  # 同上
            self.config['net'].get('drop_rate', self.dropout_p)  # 兼容外部配置/本类参数
        )
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        torch.save(self.dncnn.state_dict(), save_name + '.pth')

        res = dict(self.config)
        res['result'] = res_str
        with open(save_name + '.json', 'w') as f:
            json.dump(res, f)