import torch
from torch import nn

from Model.ConvBlock import ConvBlock


class M2SCA(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.1, sequenceLengthdim=99):
        super(M2SCA, self).__init__()

        self.conv0 = ConvBlock(in_channel, out_channel, kernel_size=1, stride_size=1, padding_size=0, dropout=dropout,
                               attn=False)

        self.conv1 = nn.Sequential(
            ConvBlock(in_channel, out_channel, kernel_size=1, stride_size=1, padding_size=0, dropout=dropout,
                      attn=False),
            ConvBlock(out_channel, out_channel, kernel_size=3, stride_size=1, padding_size=0, dropout=dropout,
                      attn=False)
        )

        self.conv2 = nn.Sequential(
            ConvBlock(in_channel, out_channel, kernel_size=1, stride_size=1, padding_size=0, dropout=dropout,
                      attn=False),
            ConvBlock(out_channel, out_channel, kernel_size=5, stride_size=1, padding_size=0, dropout=dropout,
                      attn=False),
            ConvBlock(out_channel, out_channel, kernel_size=5, stride_size=1, padding_size=0, dropout=dropout,
                      attn=False)
        )

        self.fc0 = nn.Linear(99, 99)
        self.fc1 = nn.Linear(97, 99)
        self.fc2 = nn.Linear(91, 99)

        self.multihead_attn13 = nn.MultiheadAttention(embed_dim=out_channel, num_heads=8, dropout=0.1, batch_first=True)
        self.multihead_attn31 = nn.MultiheadAttention(embed_dim=out_channel, num_heads=8, dropout=0.1, batch_first=True)
        self.multihead_attn15 = nn.MultiheadAttention(embed_dim=out_channel, num_heads=8, dropout=0.1, batch_first=True)
        self.multihead_attn51 = nn.MultiheadAttention(embed_dim=out_channel, num_heads=8, dropout=0.1, batch_first=True)
        self.multihead_attn35 = nn.MultiheadAttention(embed_dim=out_channel, num_heads=8, dropout=0.1, batch_first=True)
        self.multihead_attn53 = nn.MultiheadAttention(embed_dim=out_channel, num_heads=8, dropout=0.1, batch_first=True)

        self.bn = nn.BatchNorm1d(out_channel*6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x0 = self.fc0(x0)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        x0 = x0.permute(2, 0, 1)
        x1 = x1.permute(2, 0, 1)
        x2 = x2.permute(2, 0, 1)
        x01, _ = self.multihead_attn13(x0, x1, x1)
        x10, _ = self.multihead_attn31(x1, x0, x0)
        x12, _ = self.multihead_attn35(x1, x2, x2)
        x21, _ = self.multihead_attn53(x2, x1, x1)
        x02, _ = self.multihead_attn15(x0, x2, x2)
        x20, _ = self.multihead_attn51(x2, x0, x0)
        x01 = x01.permute(1, 2, 0)
        x12 = x12.permute(1, 2, 0)
        x02 = x02.permute(1, 2, 0)
        x10 = x10.permute(1, 2, 0)
        x21 = x21.permute(1, 2, 0)
        x20 = x20.permute(1, 2, 0)

        # x = torch.cat([x0, x1, x2], dim=1)
        x = torch.cat([x01, x10, x02, x20, x12, x21], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x
