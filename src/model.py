
from util import *
import torch
import torch.nn as nn

import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from KAN import *
from attention import *
from add_features import *
from torchvision.models import resnet18,ResNet18_Weights
import os





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool  # 导入全局池化


class GaussianFuzzyLayer(nn.Module):
    """
    高斯隶属度函数层，用于模糊化处理。
    """
    def __init__(self, input_dim, output_dim):
        super(GaussianFuzzyLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # 定义高斯隶属度函数的中心和宽度（这些参数可以训练）
        self.centers = nn.Parameter(torch.randn(output_dim))  # 高斯函数的中心
        self.widths = nn.Parameter(torch.ones(output_dim))    # 高斯函数的宽度

    def forward(self, x):
        x = self.fc(x)  # 先经过线性层
        # 使用高斯隶属度函数进行模糊化
        x = torch.exp(-((x - self.centers) ** 2) / (2 * (self.widths ** 2)))
        return x




class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GNN, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=0.6)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x, edge_index):
        x = self.dropout(F.elu(self.gat1(x, edge_index)))
        x = self.dropout(self.gat2(x, edge_index))
        return x




class MFFtPC(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, gnn_hidden_dim=128, gnn_output_dim=256, max_pool=5):
        super(MFFtPC, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout_value = dropout
        self.fan_epoch = fan_epoch
        self.num_heads = num_heads
        self.max_pool = max_pool

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if max_pool == 2:
            shape = 6016
        elif max_pool == 3:
            shape = 3968
        elif max_pool == 4:
            shape = 2944
        elif max_pool == 5:
            shape = 2304
        else:
            shape = 1920

        self.embed = nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
        self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)

        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
        self.conv2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
        self.conv3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
        self.conv4 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)

        self.MaxPool1d = nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)

        self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
        self.fan = FAN_encode(self.dropout_value, 1664).to(self.device)

        self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
        self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
        self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
        self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
        self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)

        self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1,batch_first=True, bidirectional=True).to(self.device)

        self.dropout = nn.Dropout(self.dropout_value).to(self.device)
        self.fuzzy_layer = GaussianFuzzyLayer(256, 256).to(self.device)

        self.gnn = GNN(input_dim=256, hidden_dim=gnn_hidden_dim, output_dim=gnn_output_dim, num_heads=num_heads).to(self.device)
        self.fusion_model = GateFeatureFusion(256, 256, 8,self.dropout_value).to(self.device)
        #self.fusion_model = FeatureFusion(256, 256, 8).to(self.device)

        self.full3 = nn.Linear(1664, 1024).to(self.device)
        self.full4 = nn.Linear(1024, 512).to(self.device)
        self.full5 = nn.Linear(512, 256).to(self.device)
        self.Flatten = nn.Linear(256, 64).to(self.device)
        self.out = nn.Linear(64, self.output_size).to(self.device)

    def TextCNN(self, x):
        x = x.to(self.device)
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.MaxPool1d(x1)

        x2 = self.conv2(x)
        x2 = F.relu(x2)
        x2 = self.MaxPool1d(x2)

        x3 = self.conv3(x)
        x3 = F.relu(x3)
        x3 = self.MaxPool1d(x3)

        x4 = self.conv4(x)
        x4 = F.relu(x4)
        x4 = self.MaxPool1d(x4)

        y = torch.cat([x4, x2, x3], dim=-1)
        x = self.dropout(y)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, train_data, valid_lens, features, edge_index, gnn_features, batch):
        # 将所有输入数据移动到设备
        train_data = train_data.to(self.device)

        # 序列模型部分
        embed_output = self.embed(train_data)
        pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))

        # 注意力编码
        attention_output = self.attention_encode(pos_output)

        # 特征嵌入
        aai_features = self.fc_aai(features['aai'])
        paac_features = self.fc_paac(features['paac'])
        pc6_features = self.fc_pc6(features['pc6'])
        blosum62_features = self.fc_blosum62(features['blosum62'])
        aac_features = self.fc_aac(features['aac'])


        fuzzy_aai_output = self.fuzzy_layer(aai_features)
        fuzzy_paac_output = self.fuzzy_layer(paac_features)
        fuzzy_pc6_output = self.fuzzy_layer(pc6_features)
        fuzzy_blosum62_output = self.fuzzy_layer(blosum62_features)
        fuzzy_aac_output = self.fuzzy_layer(aac_features)

        gnn_out = self.gnn(gnn_features, edge_index)  # [num_nodes, 256]
        n,c = gnn_out.shape
        a = n/50
        a = int(a)
        gnn_out = gnn_out.reshape(a,50,256)


        # 融合所有特征

        combined_features = self.fusion_model([embed_output,attention_output ,att_aai_output, att_paac_output,
                                          att_pc6_output, att_blosum62_output, att_aac_output,gnn_out])


        # combined_features = self.fusion_model([embed_output,attention_output ,att_aai_output, att_paac_output,
        #                                   att_pc6_output, att_blosum62_output, att_aac_output,gnn_out])

        #combined_features = (embed_output+attention_output+fuzzy_aai_output+fuzzy_paac_output+fuzzy_pc6_output+fuzzy_blosum62_output+fuzzy_aac_output+gnn_out)/8

        lstm_output, _ = self.bilstm(combined_features)  # 根据 LSTM 的输入维度调整参数


        #lstm_output = combined_features  # 根据 LSTM 的输入维度调整参数


        # 对 LSTM 的输出进行处理，作为 CNN 的输入
        cnn_input = lstm_output.permute(0, 2, 1)  # [batch_size, 256, seq_len]


        # 经过 CNN
        cnn_output = self.TextCNN(cnn_input)

        # 经过 FAN 编码
        fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
        for i in range(self.fan_epoch):
            fan_encode = self.fan(fan_encode)

        # 融合序列特征和 GNN 特征
        combined = fan_encode.squeeze()

        #print(combined.shape)
        combined = self.full3(combined)
        combined = F.relu(combined)
        combined = self.full4(combined)
        combined = F.relu(combined)
        combined = self.full5(combined)
        combined = F.relu(combined)
        combined = self.Flatten(combined)
        combined = F.relu(combined)
        out_label = self.out(combined)

        return out_label






