#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/26 21:04
# @Author : fhh
# @FileName: model.py
# @Software: PyCharm

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



# bert_path = 'bert-base-uncased'
# xlnet_path = 'xlnet-base-uncased'
# sbert_path = 'KR-SBERT-V40K-klueNLI-augSTS'
# feature_extration_bertpath = 'princeton-nlpunsup-simcse-bert-base-uncased'
# #tcr_bert_mlm_only_path = 'tcr-bert-mlm-only'
#
#
# """加载bert模型"""
# def load_plm(path ):
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     model = AutoModel.from_pretrained(path)
#     return tokenizer,model

"""用预训练的word2vec替代embedding"""
# def load_pretrained_embedding(vocab_size, embedding_dim):
#     """加载预训练的Word2Vec模型并返回嵌入层"""
#     # 加载模型
#     model = Word2Vec.load('amino_acid_word2vec.model')
#     weights = model.wv.vectors
#     num_embeddings, embedding_dim = weights.shape
#
#     # 创建嵌入层并加载权重
#     embedding = nn.Embedding(num_embeddings, embedding_dim)
#     embedding.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
#     embedding.weight.requires_grad = False  # 不训练这层
#
#     return embedding


# class StudentModel(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads):
#         super(StudentModel, self).__init__()
#
#         self.vocab_size = vocab_size#设置词汇表大小
#         self.embedding_size = embedding_size#设置嵌入大小
#         self.output_size = output_size# 设置输出大小
#         self.dropout = dropout# 设置dropout率
#         self.fan_epoch = fan_epoch# 设置注意力机制的迭代次数
#         self.num_heads = num_heads# 设置注意力头数
#         # 创建词嵌入层
#         self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
#         # 创建位置编码器
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
#         # 创建双向LSTM层
#         self.BiRNN = nn.LSTM(input_size=self.embedding_size,
#                              hidden_size=self.embedding_size // 2,
#                              num_layers=2,
#                              bidirectional=True,
#                              batch_first=True)
#         # 创建MASK注意力编码层
#         self.attention_encode = MASK_AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         # 创建transformer编码层
#         self.transformer = transformer_encode(self.dropout, self.embedding_size, self.num_heads)
#         # 创建加性注意力层
#         self.Attention = AdditiveAttention(input_size=self.embedding_size,
#                                            value_size=21,
#                                            num_hiddens=self.embedding_size,
#                                            dropout=0.5)
#         # 创建全连接层
#         self.full6 = nn.Linear(2100, 1000)
#         self.full1 = nn.Linear(9600, 4032)
#         self.bn1 = nn.BatchNorm1d(4032)
#         self.full2 = nn.Linear(4032, 2304)
#         self.bn = nn.BatchNorm1d(2304)
#         self.full3 = nn.Linear(2304, 1000)
#         self.full4 = nn.Linear(1000, 500)
#         self.full5 = nn.Linear(500, 256)
#
#         # 创建将多维张量展平为一维的层
#         self.Flatten = nn.Linear(256, 64)
#         # 输出层
#         self.out = nn.Linear(64, self.output_size)
#         # dropout层
#         self.dropout = torch.nn.Dropout(self.dropout)
#
#     def forward(self, train_data, valid_lens, in_feat=False):
#         """---------------------氨基酸编码------------------------"""
#         # 嵌入层处理输入数据
#         embed_output = self.embedding(train_data)
#         '''-----------------------------------------------------'''
#
#         vectors = embed_output
#
#         '''----------------------attention----------------------'''
#         # 使用MASK注意力编码层进行多次迭代处理
#         attention_encode = vectors
#         for i in range(self.fan_epoch):
#             attention_encode = self.attention_encode(attention_encode, valid_lens)
#
#         # 使用加性注意力层处理数据
#         attention_output, weights = self.Attention(attention_encode, attention_encode, attention_encode, valid_lens)
#         '''-----------------------------------------------------'''
#         # 展平处理后的数据
#         out = attention_output.contiguous().view(attention_output.size()[0], -1)
#
#         # 全连接层
#         label = self.full2(out)
#         label = self.bn(label)
#         label = torch.nn.ReLU()(label)
#
#         label = self.full3(label)
#         label = torch.nn.ReLU()(label)
#
#         label2 = self.full4(label)
#         label = torch.nn.ReLU()(label2)
#
#         label3 = self.full5(label)
#         label = torch.nn.ReLU()(label3)
#
#         label4 = self.Flatten(label)
#         label = torch.nn.ReLU()(label4)
#         out_label = self.out(label)
#
#         # 如果需要输出中间特征，则返回对应的中间特征，否则返回输出结果
#         if in_feat:
#             return label2, label3, label4, out_label
#         else:
#             return out_label



"""原代码"""

# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#         # 根据最大池化大小设置特定形状参数
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         # 词嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
#
#         # 位置编码层
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
#
#         # 一维卷积层
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=2,
#                                      stride=1
#                                      )
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=3,
#                                      stride=1
#                                      )
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=4,
#                                      stride=1
#                                      )
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=5,
#                                      stride=1
#                                      )
#         # 序列最短为5，故将卷积核分别设为：2、3、4、5
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
#
#         # 注意力编码层
#         self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         # Fan编码层
#         self.fan = FAN_encode(self.dropout, shape)
#
#         # 全连接层
#         self.full3 = nn.Linear(shape, 1000)
#         self.full4 = nn.Linear(1000, 500)
#         self.full5 = nn.Linear(500, 256)
#         # self.full6 = nn.Linear(4608, 2304)
#         self.Flatten = nn.Linear(256, 64)
#         self.out = nn.Linear(64, self.output_size)
#         self.dropout = torch.nn.Dropout(self.dropout)
#
#
#
#     def TextCNN(self, x):
#         # 定义文本卷积神经网络的前向传播过程
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         y = torch.cat([x1, x2, x3, x4], dim=-1)
#
#         x = self.dropout(y)
#
#         x = x.view(x.size(0), -1)
#
#         return x
#
#     def forward(self, train_data, valid_lens = None, in_feat=False):
#
#         # 进行词嵌入
#         embed_output = self.embed(train_data)
#
#         '''----------------------位置编码------------------------'''
#         pos_output = self.pos_encoding(self.embed(train_data) * math.sqrt(self.embedding_size))
#         '''-----------------------------------------------------'''
#
#         '''----------------------attention----------------------'''
#         # 进行注意力编码
#         attention_output = self.attention_encode(pos_output)
#         '''-----------------------------------------------------'''
#
#         '''----------------------特征相加-------------------------'''
#         vectors = embed_output + attention_output
#         '''------------------------------------------------------'''
#
#         '''---------------------data_cnn-----------------------'''
#         # 调整维度以适应卷积层输入格式
#
#
#
#         cnn_input = vectors.permute(0, 2, 1)
#
#         #文本卷积神经网络处理
#         cnn_output = self.TextCNN(cnn_input)
#         '''-----------------------------------------------------'''
#
#         '''---------------------fan_encode----------------------'''
#         # 调整维度以适应Fan编码层输入格式
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         # Fan编码处理
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#         '''-----------------------------------------------------'''
#
#         # 去除多余的维度
#         out = fan_encode.squeeze()
#         # 全连接层
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label1 = self.full4(label)
#         label = torch.nn.ReLU()(label1)
#         label2 = self.full5(label)
#         label = torch.nn.ReLU()(label2)
#         label3 = self.Flatten(label)
#         label = torch.nn.ReLU()(label3)
#         out_label = self.out(label)
#
#         if in_feat:
#             return label1, label2, label3, out_label
#         else:
#             return out_label



"""这个是把四个特征提取方法放在一起的版本"""
# import torch
# import torch.nn as nn
# from util import PositionalEncoding, AttentionEncode, FAN_encode
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
#         #self.embed = load_pretrained_embedding(vocab_size, embedding_size)
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
#         # self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=2,
#         #                              stride=1
#         #                              )
#         # self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=4,
#         #                              stride=1
#         #                              )
#         # self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=6,
#         #                              stride=1
#         #                              )
#         # self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=8,
#         #                              stride=1
#         #                              )
#         # self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
#         # self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         # self.fan = FAN_encode(self.dropout, 1664)
#         # self.fc_aai = nn.Linear(531, self.embedding_size)
#         # self.fc_paac = nn.Linear(3, self.embedding_size)
#         # self.fc_pc6 = nn.Linear(6, self.embedding_size)
#         # self.fc_blosum62 = nn.Linear(23, self.embedding_size)
#         # self.fc_aac = nn.Linear(20,self.embedding_size)
#         # self.full3 = nn.Linear(1664, 1000)
#         # self.full4 = nn.Linear(1000, 500)
#         # self.full5 = nn.Linear(500, 256)
#         # self.Flatten = nn.Linear(256, 64)
#         # self.out = nn.Linear(64, self.output_size)
#
#
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=3,
#                                      stride=1
#                                      )
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=6,
#                                      stride=1
#                                      )
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=9,
#                                      stride=1
#                                      )
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=12,
#                                      stride=1
#                                      )
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
#         self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         self.fan = FAN_encode(self.dropout, 1536)
#         self.fc_aai = nn.Linear(531, self.embedding_size)
#         self.fc_paac = nn.Linear(3, self.embedding_size)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size)
#         self.fc_aac = nn.Linear(20,self.embedding_size)
#         self.full3 = nn.Linear(1536, 1000)
#         self.full4 = nn.Linear(1000, 500)
#         self.full5 = nn.Linear(500, 256)
#         self.Flatten = nn.Linear(256, 64)
#         self.out = nn.Linear(64, self.output_size)
#
#
#
#         self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,batch_first=True, bidirectional=True)
#
#         #卷积下面加用这个# class ETFC(nn.Module):
# #     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
# #         super(ETFC, self).__init__()
# #         self.vocab_size = vocab_size
# #         self.embedding_size = embedding_size
# #         self.output_size = output_size
# #         self.dropout = dropout
# #         self.fan_epoch = fan_epoch
# #         self.num_heads = num_heads
# #         self.max_pool = max_pool
# #
# #         # 设备
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# #         # 根据 max_pool 设置形状
# #         if max_pool == 2:
# #             shape = 6016
# #         elif max_pool == 3:
# #             shape = 3968
# #         elif max_pool == 4:
# #             shape = 2944
# #         elif max_pool == 5:
# #             shape = 2304
# #         else:
# #             shape = 1920
# #
# #         # 定义嵌入层
# #         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
# #
# #         # 定位编码
# #         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout).to(self.device)
# #
# #         # 卷积层定义
# #         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
# #         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
# #         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
# #         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
# #
# #         # 最大池化层
# #         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
# #
# #         # 注意力编码
# #         self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads).to(self.device)
# #
# #         # FAN 编码
# #         self.fan = FAN_encode(self.dropout, 1664).to(self.device)
# #
# #         # 特征嵌入
# #         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
# #         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
# #         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
# #         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
# #         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
# #
# #         # 全连接层
# #         self.full3 = nn.Linear(1664, 1000).to(self.device)
# #         self.full4 = nn.Linear(1000, 500).to(self.device)
# #         self.full5 = nn.Linear(500, 256).to(self.device)
# #         self.Flatten = nn.Linear(256, 64).to(self.device)
# #         self.out = nn.Linear(64, self.output_size).to(self.device)
# #
# #         # 双向 LSTM 层
# #         self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
# #
# #         # Dropout 层
# #         self.dropout = torch.nn.Dropout(self.dropout).to(self.device)
# #
# #
# #
# #     def TextCNN(self, x):
# #         x = x.to(self.device)
# #         x1 = self.conv1(x)
# #         x1 = torch.nn.ReLU()(x1)
# #         x1 = self.MaxPool1d(x1)
# #
# #         x2 = self.conv2(x)
# #         x2 = torch.nn.ReLU()(x2)
# #         x2 = self.MaxPool1d(x2)
# #
# #         x3 = self.conv3(x)
# #         x3 = torch.nn.ReLU()(x3)
# #         x3 = self.MaxPool1d(x3)
# #
# #         x4 = self.conv4(x)
# #         x4 = torch.nn.ReLU()(x4)
# #         x4 = self.MaxPool1d(x4)
# #
# #         # 拼接卷积层的输出
# #         y = torch.cat([x4, x2, x3], dim=-1)
# #         x = self.dropout(y)
# #         x = x.view(x.size(0), -1)
# #         return x
# #
# #     def forward(self, train_data, valid_lens, features):
# #         # 将所有输入数据移动到设备
# #         train_data = train_data.to(self.device)
# #
# #         # n, c = train_data.shape
# #         # hidden_slice_row = torch.zeros(n, 50).to(self.device)
# #         # hidden_slice_col = torch.zeros(n, 50).to(self.device)
# #         # slice_len = 10
# #         # train_data = self.hvgsu(hidden_slice_row, hidden_slice_col, train_data, n, 0, slice_len)
# #
# #         embed_output = self.embed(train_data)
# #         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
# #
# #         # 注意力编码
# #         attention_output = self.attention_encode(pos_output)
# #
# #         # 特征嵌入
# #         aai_features = self.fc_aai(features['aai'])
# #         paac_features = self.fc_paac(features['paac'])
# #         pc6_features = self.fc_pc6(features['pc6'])
# #         blosum62_features = self.fc_blosum62(features['blosum62'])
# #         aac_features = self.fc_aac(features['aac'])
# #
# #         # 特征注意力编码
# #         attention_aai_output = self.attention_encode(aai_features)
# #         attention_paac_output = self.attention_encode(paac_features)
# #         attention_pc6_output = self.attention_encode(pc6_features)
# #         attention_blosum62_output = self.attention_encode(blosum62_features)
# #         attention_aac_output = self.attention_encode(aac_features)
# #
# #
# #
# #         # 融合所有特征
# #         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
# #         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
# #                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
# #
# #
# #         # 经过双向 LSTM 层
# #         #vectors, _ = self.bilstm(combined_features)
# #         vectors= combined_features
# #         cnn_input = vectors.permute(0, 2, 1)
# #
# #         # 经过卷积层
# #         cnn_output = self.TextCNN(cnn_input)
# #
# #         # 经过 FAN 编码
# #         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
# #         for i in range(self.fan_epoch):
# #             fan_encode = self.fan(fan_encode)
# #
# #         # 经过全连接层
# #         out = fan_encode.squeeze()
# #
# #         label = self.full3(out)
# #         label = torch.nn.ReLU()(label)
# #         label = self.full4(label)
# #         label = torch.nn.ReLU()(label)
# #         label = self.full5(label)
# #         label = torch.nn.ReLU()(label)
# #         label = self.Flatten(label)
# #         label = torch.nn.ReLU()(label)
# #         out_label = self.out(label)
# #
# #         return out_label
#         #self.bilstm = nn.LSTM(input_size=2304, hidden_size=1152, num_layers=1, batch_first=True, bidirectional=True)
#
#
#
#         self.dropout = torch.nn.Dropout(self.dropout)
#
#
#     def TextCNN(self, x):
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#         #y = torch.cat([x1, x2, x3, x4], dim=-1)
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(self.embed(train_data) * math.sqrt(self.embedding_size))
#         '''-----------------------------------------------------'''
#
#         '''----------------------attention----------------------'''
#         # 进行注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#         '''-----------------------------------------------------'''
#
#         '''----------------------特征相加-------------------------'''
#
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         attention_aai_output = self.attention_encode(aai_features)
#         attention_paac_output = self.attention_encode(paac_features)
#         attention_pc6_output = self.attention_encode(pc6_features)
#         attention_blosum62_output = self.attention_encode(blosum62_features)
#         attention_aac_output = self.attention_encode(aac_features)
#
#
#
#         #combined_features = aai_features + embed_output
#
#         #combined_features = embed_output + paac_features + blosum62_features + pc6_features + attention_output + aai_features
#         #combined_features = embed_output +attention_output +attention_aai_output+attention_paac_output+attention_pc6_output+attention_blosum62_output
#
#         fusion_model = FeatureFusion(256, 256, 7).to(DEVICE)
#
#         # 融合特征
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                       attention_pc6_output, attention_blosum62_output,attention_aac_output])
#
#
#         vectors = combined_features
#         vectors, _ = self.bilstm(vectors)  # x 的形状：[200, 50, 256]，因为双向每方向128
#         cnn_input = vectors.permute(0, 2, 1)
#         cnn_output = self.TextCNN(cnn_input)
#
#
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#         out = fan_encode.squeeze()
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label1 = self.full4(label)
#         label = torch.nn.ReLU()(label1)
#         label2 = self.full5(label)
#         label = torch.nn.ReLU()(label2)
#         label3 = self.Flatten(label)
#         label = torch.nn.ReLU()(label3)
#         out_label = self.out(label)
#         return out_label



#魔改


import torch
import torch.nn as nn
import math

# 请确保导入或定义了以下类和函数
# from positional_encoding import PositionalEncoding
# from attention_encode import AttentionEncode
# from fan_encode import FAN_encode
# from feature_fusion import FeatureFusion



# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size  # 动态调整的 embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout  # 使用 dropout_value 避免与 nn.Dropout 冲突
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 根据 max_pool 设置形状（您可以根据实际需要调整）
#         shape = 1664  # 需要根据模型的输出调整
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 卷积层定义，使用动态的 embedding_size
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # FAN 编码
#         self.fan = FAN_encode(self.dropout_value, shape).to(self.device)
#
#         # 特征嵌入，使用动态的 embedding_size
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 全连接层，输入大小需要根据前一层的输出调整
#         self.full3 = nn.Linear(shape, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         # 双向 LSTM 层，input_size 使用动态的 embedding_size
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         # 拼接卷积层的输出
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])  # 访问 'aai' 特征
#
#         paac_features = self.fc_paac(features['paac'])  # 访问 'paac' 特征
#         pc6_features = self.fc_pc6(features['pc6'])  # 访问 'pc6' 特征
#         blosum62_features = self.fc_blosum62(features['blosum62'])  # 访问 'blosum62' 特征
#         aac_features = self.fc_aac(features['aac'])  # 访问 'aac' 特征
#
#         # 特征注意力编码
#         attention_aai_output = self.attention_encode(aai_features)
#         attention_paac_output = self.attention_encode(paac_features)
#         attention_pc6_output = self.attention_encode(pc6_features)
#         attention_blosum62_output = self.attention_encode(blosum62_features)
#         attention_aac_output = self.attention_encode(aac_features)
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(self.embedding_size, self.embedding_size, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         # 经过双向 LSTM 层
#         vectors = combined_features
#         cnn_input = vectors.permute(0, 2, 1)
#
#         # 经过卷积层
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过全连接层
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label  # 输出形状应为 [batch_size, num_classes]


#PSO训练用
# import torch
# import torch.nn as nn
# import math
#
# # 请确保导入或定义了以下类和函数
# # from positional_encoding import PositionalEncoding
# # from attention_encode import AttentionEncode
# # from fan_encode import FAN_encode
# # from feature_fusion import FeatureFusion
#
# class GaussianFuzzyLayer(nn.Module):
#     """
#     高斯隶属度函数层，用于模糊化处理。
#     """
#     def __init__(self, input_dim, output_dim):
#         super(GaussianFuzzyLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         # 定义高斯隶属度函数的中心和宽度（这些参数可以训练）
#         self.centers = nn.Parameter(torch.randn(output_dim))  # 高斯函数的中心
#         self.widths = nn.Parameter(torch.ones(output_dim))    # 高斯函数的宽度
#
#     def forward(self, x):
#         x = self.fc(x)  # 先经过线性层
#         # 使用高斯隶属度函数进行模糊化
#         x = torch.exp(-((x - self.centers) ** 2) / (2 * (self.widths ** 2)))
#         return x
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size  # 动态调整的 embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout  # 使用 dropout_value 避免与 nn.Dropout 冲突
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 根据 max_pool 设置形状（您可以根据实际需要调整）
#         shape = 1664  # 需要根据模型的输出调整
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 卷积层定义，使用动态的 embedding_size
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # FAN 编码
#         self.fan = FAN_encode(self.dropout_value, shape).to(self.device)
#
#
#         # 特征嵌入，使用动态的 embedding_size
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 全连接层，输入大小需要根据前一层的输出调整
#         self.full3 = nn.Linear(shape, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         # 双向 LSTM 层，input_size 使用动态的 embedding_size
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#         # 使用高斯隶属函数的模糊化层
#         self.fuzzy_layer = GaussianFuzzyLayer(256,256).to(self.device)
#
#
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         # 拼接卷积层的输出
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])  # 访问 'aai' 特征
#
#         paac_features = self.fc_paac(features['paac'])  # 访问 'paac' 特征
#         pc6_features = self.fc_pc6(features['pc6'])  # 访问 'pc6' 特征
#         blosum62_features = self.fc_blosum62(features['blosum62'])  # 访问 'blosum62' 特征
#         aac_features = self.fc_aac(features['aac'])  # 访问 'aac' 特征
#
#         # 特征注意力编码
#         attention_aai_output = self.attention_encode(aai_features)
#         attention_paac_output = self.attention_encode(paac_features)
#         attention_pc6_output = self.attention_encode(pc6_features)
#         attention_blosum62_output = self.attention_encode(blosum62_features)
#         attention_aac_output = self.attention_encode(aac_features)
#
#         attention_aai_output = self.fuzzy_layer(attention_aai_output)
#         attention_paac_output = self.fuzzy_layer(attention_paac_output)
#         attention_pc6_output = self.fuzzy_layer(attention_pc6_output)
#         attention_blosum62_output = self.fuzzy_layer(attention_blosum62_output)
#         attention_aac_output = self.fuzzy_layer(attention_aac_output)
#
#
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(self.embedding_size, self.embedding_size, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         # 经过双向 LSTM 层
#         vectors = combined_features
#         cnn_input = vectors.permute(0, 2, 1)
#
#         # 经过卷积层
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过全连接层
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label  # 输出形状应为 [batch_size, num_classes]


# class GaussianFuzzyLayer(nn.Module):
#     """
#     高斯隶属度函数层，用于模糊化处理。
#     """
#     def __init__(self, input_dim, output_dim):
#         super(GaussianFuzzyLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         # 定义高斯隶属度函数的中心和宽度（这些参数可以训练）
#         self.centers = nn.Parameter(torch.randn(output_dim))  # 高斯函数的中心
#         self.widths = nn.Parameter(torch.ones(output_dim))    # 高斯函数的宽度
#
#     def forward(self, x):
#         x = self.fc(x)  # 先经过线性层
#         # 使用高斯隶属度函数进行模糊化
#         x = torch.exp(-((x - self.centers) ** 2) / (2 * (self.widths ** 2)))
#         return x
#
#
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 根据 max_pool 设置形状
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 卷积层定义
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # FAN 编码
#         self.fan = FAN_encode(self.dropout_value,1664).to(self.device)
#
#         # 特征嵌入
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 全连接层
#         self.full3 = nn.Linear(1664, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         # 双向 LSTM 层
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#         # 使用高斯隶属函数的模糊化层
#         self.fuzzy_layer = GaussianFuzzyLayer(256,256).to(self.device)
#
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         # 拼接卷积层的输出
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         #x = self.fuzzy_layer(x)
#         return x
#
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         # 特征注意力编码
#         # attention_aai_output = self.attention_encode(aai_features)
#         # attention_paac_output = self.attention_encode(paac_features)
#         # attention_pc6_output = self.attention_encode(pc6_features)
#         # attention_blosum62_output = self.attention_encode(blosum62_features)
#         # attention_aac_output = self.attention_encode(aac_features)
#
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#
#         lstm_output, _ = self.bilstm(combined_features)  # 根据 LSTM 的输入维度调整参数
#
#
#         # 对 LSTM 的输出进行处理，作为 CNN 的输入
#         cnn_input = lstm_output.permute(0, 2, 1)  # [batch_size, 256, seq_len]
#
#         # 经过 CNN
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过全连接层
#
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#
#         return out_label


#GAN

# import torch
# import torch.nn as nn
# import math
#
# # 确保以下模块已正确导入或定义
# # from your_module import PositionalEncoding, AttentionEncode, FAN_encode, GaussianFuzzyLayer, FeatureFusion
#
# class Generator(nn.Module):
#     def __init__(self, noise_dim, seq_length, vocab_size, temperature=1.0):
#         super(Generator, self).__init__()
#         self.seq_length = seq_length
#         self.vocab_size = vocab_size
#         self.temperature = temperature
#         self.model = nn.Sequential(
#             nn.Linear(noise_dim, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.Linear(512, seq_length * vocab_size)
#         )
#
#     def forward(self, z):
#         output = self.model(z)
#         output = output.view(-1, self.seq_length, self.vocab_size)
#         # 使用 Gumbel-Softmax 采样
#         gen_data = F.gumbel_softmax(output, tau=self.temperature, hard=True, dim=-1)
#         return gen_data  # 返回 one-hot 向量序列
#
# class Discriminator(nn.Module):
#     def __init__(self, seq_length, vocab_size):
#         super(Discriminator, self).__init__()
#         self.seq_length = seq_length
#         self.vocab_size = vocab_size
#
#         self.model = nn.Sequential(
#             nn.Linear(seq_length * vocab_size, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         # 输入的数据形状为 (batch_size, seq_length, vocab_size)
#         data = data.view(data.size(0), -1)  # 展平
#         validity = self.model(data)
#         return validity
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         self.fan = FAN_encode(self.dropout_value,1664).to(self.device)
#
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         self.full3 = nn.Linear(1664, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#         self.fuzzy_layer = GaussianFuzzyLayer(256,256).to(self.device)
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         attention_output = self.attention_encode(pos_output)
#
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#         # attention_aai_output = self.attention_encode(aai_features)
#         # attention_paac_output = self.attention_encode(paac_features)
#         # attention_pc6_output = self.attention_encode(pc6_features)
#         # attention_blosum62_output = self.attention_encode(blosum62_features)
#         # attention_aac_output = self.attention_encode(aac_features)
#
#
#         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         lstm_output, _ = self.bilstm(combined_features)
#
#         cnn_input = lstm_output.permute(0, 2, 1)
#
#
#         cnn_output = self.TextCNN(cnn_input)
#
#
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label





#用BAS（Beetle Antennae Search）进行超参优化，还有CNN,LSTM的超参也一起优化了

# import torch
# import torch.nn as nn
# import math
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads,
#                  conv_channels=64, kernel_sizes=[2, 4, 6, 8], max_pool=2,
#                  lstm_hidden_size=128, lstm_num_layers=1, lstm_dropout=0.0):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#         self.conv_channels = conv_channels
#         self.kernel_sizes = kernel_sizes  # 支持多种卷积核大小
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # 特征嵌入
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 使用高斯隶属函数的模糊化层
#         self.fuzzy_layer = GaussianFuzzyLayer(256, 256).to(self.device)
#
#         # 融合所有特征
#         self.fusion_model = FeatureFusion(self.embedding_size, self.embedding_size, 7).to(self.device)
#
#         # 双向 LSTM 层
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
#                               dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
#                               batch_first=True, bidirectional=True).to(self.device)
#
#         # 卷积层定义，支持多种卷积核大小
#         # 修改 in_channels 为 lstm_hidden_size * 2
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=lstm_hidden_size * 2, out_channels=self.conv_channels, kernel_size=ks).to(self.device)
#             for ks in self.kernel_sizes
#         ])
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#         # 全连接层将在 forward 中动态定义
#         self.full3 = None
#         self.full4 = None
#         self.full5 = None
#         self.Flatten = None
#         self.out = None
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         conv_outputs = []
#         for conv in self.convs:
#             x_conv = conv(x)
#             x_conv = torch.nn.ReLU()(x_conv)
#             x_conv = self.MaxPool1d(x_conv)
#             conv_outputs.append(x_conv)
#         # 拼接卷积层的输出
#         y = torch.cat(conv_outputs, dim = -1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         # 特征模糊化处理
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#         # 融合所有特征
#         combined_features = self.fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                               attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         lstm_output, _ = self.bilstm(combined_features)  # [batch, seq_len, lstm_hidden_size * 2]
#
#         # 对 LSTM 的输出进行处理，作为 CNN 的输入
#         cnn_input = lstm_output.permute(0, 2, 1)  # [batch_size, lstm_hidden_size*2, seq_len]
#
#         # 经过 CNN
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 这里根据 cnn_output 的形状动态定义全连接层
#         if self.full3 is None:
#             fc_input_dim = cnn_output.size(1)
#             self.full3 = nn.Linear(fc_input_dim, 1000).to(self.device)
#             self.full4 = nn.Linear(1000, 500).to(self.device)
#             self.full5 = nn.Linear(500, 256).to(self.device)
#             self.Flatten = nn.Linear(256, 64).to(self.device)
#             self.out = nn.Linear(64, self.output_size).to(self.device)
#             # 初始化权重
#             nn.init.xavier_uniform_(self.full3.weight)
#             nn.init.xavier_uniform_(self.full4.weight)
#             nn.init.xavier_uniform_(self.full5.weight)
#             nn.init.xavier_uniform_(self.Flatten.weight)
#             nn.init.xavier_uniform_(self.out.weight)
#
#         # 经过全连接层
#         label = self.full3(cnn_output)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label





#转成图卷积
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
# from torch_geometric.utils import add_self_loops
#
# class GNNModel(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, num_layers=3):
#         super(GNNModel, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#
#         # 词嵌入层
#         self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
#
#         # GCN层
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         for i in range(num_layers):
#             self.convs.append(GCNConv(self.embedding_size, self.embedding_size))
#             self.bns.append(BatchNorm(self.embedding_size))
#
#         # 全连接层
#         self.fc1 = nn.Linear(self.embedding_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, self.output_size)
#
#         self.dropout_layer = nn.Dropout(self.dropout)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         # 词嵌入
#         x = self.embedding(x)
#         #x = x.float()  # 转换为浮点型
#
#         # GCN层
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             #x = bn(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 全局池化
#         x = global_mean_pool(x, batch)  # [batch_size, embedding_size]
#
#         # 全连接层
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout_layer(x)
#
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout_layer(x)
#         #x = F.relu(x)
#         x = self.fc3(x)
#         return x


# model.py

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from torch_geometric.nn import GCNConv, GATConv, global_mean_pool  # 导入全局池化
#
# class GNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
#         super(GNN, self).__init__()
#         self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6)
#         self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=0.6)
#         self.dropout = nn.Dropout(0.6)
#
#     def forward(self, x, edge_index):
#         x = self.dropout(F.elu(self.gat1(x, edge_index)))
#         x = self.dropout(self.gat2(x, edge_index))
#         return x
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, gnn_hidden_dim=128, gnn_output_dim=256, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         self.embed = nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         self.MaxPool1d = nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#         self.fan = FAN_encode(self.dropout_value, 1664).to(self.device)
#
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # self.full3 = nn.Linear(1664, 1000).to(self.device)
#         # self.full4 = nn.Linear(1000, 500).to(self.device)
#         # self.full5 = nn.Linear(500, 256).to(self.device)
#         # self.Flatten = nn.Linear(256, 64).to(self.device)
#         # self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         self.dropout = nn.Dropout(self.dropout_value).to(self.device)
#         self.fuzzy_layer = GaussianFuzzyLayer(256, 256).to(self.device)
#
#         self.gnn = GNN(input_dim=256, hidden_dim=gnn_hidden_dim, output_dim=gnn_output_dim, num_heads=num_heads).to(self.device)
#
#         self.full3 = nn.Linear(1664 + gnn_output_dim, 1024).to(self.device)
#         self.full4 = nn.Linear(1024, 512).to(self.device)
#         self.full5 = nn.Linear(512, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = F.relu(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = F.relu(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = F.relu(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = F.relu(x4)
#         x4 = self.MaxPool1d(x4)
#
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features, edge_index, gnn_features, batch):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         # 序列模型部分
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         lstm_output, _ = self.bilstm(combined_features)  # 根据 LSTM 的输入维度调整参数
#
#         # 对 LSTM 的输出进行处理，作为 CNN 的输入
#         cnn_input = lstm_output.permute(0, 2, 1)  # [batch_size, 256, seq_len]
#
#
#         # 经过 CNN
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过 GNN
#         gnn_out = self.gnn(gnn_features, edge_index)  # [num_nodes, 256]
#         #print(gnn_out.shape)
#
#         # 对 GNN 输出进行全局平均池化，获得每个图的图级特征
#         gnn_pooled = global_mean_pool(gnn_out, batch)  # [batch_size, output_dim]
#         #print(gnn_pooled.shape)
#
#
#         # 融合序列特征和 GNN 特征
#         combined = torch.cat([fan_encode.squeeze(), gnn_pooled], dim=1)
#         #print(combined.shape)
#         combined = self.full3(combined)
#         combined = F.relu(combined)
#         combined = self.full4(combined)
#         combined = F.relu(combined)
#         combined = self.full5(combined)
#         combined = F.relu(combined)
#         combined = self.Flatten(combined)
#         combined = F.relu(combined)
#         out_label = self.out(combined)
#
#         return out_label



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


# import torch
# import torch.nn as nn
#
# class GaussianFuzzyLayer(nn.Module):
#     """
#     高斯隶属度函数层，用于模糊化处理。
#     """
#     def __init__(self, input_dim, output_dim, dropout_value = 0.6):
#         super(GaussianFuzzyLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.centers = nn.Parameter(torch.randn(output_dim))  # 高斯函数的中心
#         self.widths = nn.Parameter(torch.ones(output_dim))    # 高斯函数的宽度
#         self.dropout = nn.Dropout(dropout_value)  # 添加 Dropout 层
#
#     def forward(self, x):
#         x = self.fc(x)  # 先经过线性层
#         x = torch.exp(-((x - self.centers) ** 2) / (2 * (self.widths ** 2)))  # 使用高斯隶属度函数进行模糊化
#         x = self.dropout(x)  # 应用 Dropout
#         return x


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




class MMEN_MTPP(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, gnn_hidden_dim=128, gnn_output_dim=256, max_pool=5):
        super(MMEN_MTPP, self).__init__()
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

        att_aai_output = self.attention_encode(aai_features)
        att_paac_output = self.attention_encode(paac_features)
        att_pc6_output = self.attention_encode(pc6_features)
        att_blosum62_output = self.attention_encode(blosum62_features)
        att_aac_output = self.attention_encode(aac_features)


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

        # combined_features = self.fusion_model([embed_output,attention_output ,fuzzy_aai_output, fuzzy_paac_output,
        #                                   fuzzy_pc6_output, fuzzy_blosum62_output, fuzzy_aac_output,gnn_out])


        combined_features = self.fusion_model([embed_output,attention_output ,att_aai_output, att_paac_output,
                                          att_pc6_output, att_blosum62_output, att_aac_output,gnn_out])

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/26 21:04
# @Author : fhh
# @FileName: model.py
# @Software: PyCharm

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



# bert_path = 'bert-base-uncased'
# xlnet_path = 'xlnet-base-uncased'
# sbert_path = 'KR-SBERT-V40K-klueNLI-augSTS'
# feature_extration_bertpath = 'princeton-nlpunsup-simcse-bert-base-uncased'
# #tcr_bert_mlm_only_path = 'tcr-bert-mlm-only'
#
#
# """加载bert模型"""
# def load_plm(path ):
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     model = AutoModel.from_pretrained(path)
#     return tokenizer,model

"""用预训练的word2vec替代embedding"""
# def load_pretrained_embedding(vocab_size, embedding_dim):
#     """加载预训练的Word2Vec模型并返回嵌入层"""
#     # 加载模型
#     model = Word2Vec.load('amino_acid_word2vec.model')
#     weights = model.wv.vectors
#     num_embeddings, embedding_dim = weights.shape
#
#     # 创建嵌入层并加载权重
#     embedding = nn.Embedding(num_embeddings, embedding_dim)
#     embedding.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
#     embedding.weight.requires_grad = False  # 不训练这层
#
#     return embedding


# class StudentModel(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads):
#         super(StudentModel, self).__init__()
#
#         self.vocab_size = vocab_size#设置词汇表大小
#         self.embedding_size = embedding_size#设置嵌入大小
#         self.output_size = output_size# 设置输出大小
#         self.dropout = dropout# 设置dropout率
#         self.fan_epoch = fan_epoch# 设置注意力机制的迭代次数
#         self.num_heads = num_heads# 设置注意力头数
#         # 创建词嵌入层
#         self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
#         # 创建位置编码器
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
#         # 创建双向LSTM层
#         self.BiRNN = nn.LSTM(input_size=self.embedding_size,
#                              hidden_size=self.embedding_size // 2,
#                              num_layers=2,
#                              bidirectional=True,
#                              batch_first=True)
#         # 创建MASK注意力编码层
#         self.attention_encode = MASK_AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         # 创建transformer编码层
#         self.transformer = transformer_encode(self.dropout, self.embedding_size, self.num_heads)
#         # 创建加性注意力层
#         self.Attention = AdditiveAttention(input_size=self.embedding_size,
#                                            value_size=21,
#                                            num_hiddens=self.embedding_size,
#                                            dropout=0.5)
#         # 创建全连接层
#         self.full6 = nn.Linear(2100, 1000)
#         self.full1 = nn.Linear(9600, 4032)
#         self.bn1 = nn.BatchNorm1d(4032)
#         self.full2 = nn.Linear(4032, 2304)
#         self.bn = nn.BatchNorm1d(2304)
#         self.full3 = nn.Linear(2304, 1000)
#         self.full4 = nn.Linear(1000, 500)
#         self.full5 = nn.Linear(500, 256)
#
#         # 创建将多维张量展平为一维的层
#         self.Flatten = nn.Linear(256, 64)
#         # 输出层
#         self.out = nn.Linear(64, self.output_size)
#         # dropout层
#         self.dropout = torch.nn.Dropout(self.dropout)
#
#     def forward(self, train_data, valid_lens, in_feat=False):
#         """---------------------氨基酸编码------------------------"""
#         # 嵌入层处理输入数据
#         embed_output = self.embedding(train_data)
#         '''-----------------------------------------------------'''
#
#         vectors = embed_output
#
#         '''----------------------attention----------------------'''
#         # 使用MASK注意力编码层进行多次迭代处理
#         attention_encode = vectors
#         for i in range(self.fan_epoch):
#             attention_encode = self.attention_encode(attention_encode, valid_lens)
#
#         # 使用加性注意力层处理数据
#         attention_output, weights = self.Attention(attention_encode, attention_encode, attention_encode, valid_lens)
#         '''-----------------------------------------------------'''
#         # 展平处理后的数据
#         out = attention_output.contiguous().view(attention_output.size()[0], -1)
#
#         # 全连接层
#         label = self.full2(out)
#         label = self.bn(label)
#         label = torch.nn.ReLU()(label)
#
#         label = self.full3(label)
#         label = torch.nn.ReLU()(label)
#
#         label2 = self.full4(label)
#         label = torch.nn.ReLU()(label2)
#
#         label3 = self.full5(label)
#         label = torch.nn.ReLU()(label3)
#
#         label4 = self.Flatten(label)
#         label = torch.nn.ReLU()(label4)
#         out_label = self.out(label)
#
#         # 如果需要输出中间特征，则返回对应的中间特征，否则返回输出结果
#         if in_feat:
#             return label2, label3, label4, out_label
#         else:
#             return out_label



"""原代码"""

# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#         # 根据最大池化大小设置特定形状参数
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         # 词嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
#
#         # 位置编码层
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
#
#         # 一维卷积层
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=2,
#                                      stride=1
#                                      )
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=3,
#                                      stride=1
#                                      )
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=4,
#                                      stride=1
#                                      )
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=5,
#                                      stride=1
#                                      )
#         # 序列最短为5，故将卷积核分别设为：2、3、4、5
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
#
#         # 注意力编码层
#         self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         # Fan编码层
#         self.fan = FAN_encode(self.dropout, shape)
#
#         # 全连接层
#         self.full3 = nn.Linear(shape, 1000)
#         self.full4 = nn.Linear(1000, 500)
#         self.full5 = nn.Linear(500, 256)
#         # self.full6 = nn.Linear(4608, 2304)
#         self.Flatten = nn.Linear(256, 64)
#         self.out = nn.Linear(64, self.output_size)
#         self.dropout = torch.nn.Dropout(self.dropout)
#
#
#
#     def TextCNN(self, x):
#         # 定义文本卷积神经网络的前向传播过程
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         y = torch.cat([x1, x2, x3, x4], dim=-1)
#
#         x = self.dropout(y)
#
#         x = x.view(x.size(0), -1)
#
#         return x
#
#     def forward(self, train_data, valid_lens = None, in_feat=False):
#
#         # 进行词嵌入
#         embed_output = self.embed(train_data)
#
#         '''----------------------位置编码------------------------'''
#         pos_output = self.pos_encoding(self.embed(train_data) * math.sqrt(self.embedding_size))
#         '''-----------------------------------------------------'''
#
#         '''----------------------attention----------------------'''
#         # 进行注意力编码
#         attention_output = self.attention_encode(pos_output)
#         '''-----------------------------------------------------'''
#
#         '''----------------------特征相加-------------------------'''
#         vectors = embed_output + attention_output
#         '''------------------------------------------------------'''
#
#         '''---------------------data_cnn-----------------------'''
#         # 调整维度以适应卷积层输入格式
#
#
#
#         cnn_input = vectors.permute(0, 2, 1)
#
#         #文本卷积神经网络处理
#         cnn_output = self.TextCNN(cnn_input)
#         '''-----------------------------------------------------'''
#
#         '''---------------------fan_encode----------------------'''
#         # 调整维度以适应Fan编码层输入格式
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         # Fan编码处理
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#         '''-----------------------------------------------------'''
#
#         # 去除多余的维度
#         out = fan_encode.squeeze()
#         # 全连接层
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label1 = self.full4(label)
#         label = torch.nn.ReLU()(label1)
#         label2 = self.full5(label)
#         label = torch.nn.ReLU()(label2)
#         label3 = self.Flatten(label)
#         label = torch.nn.ReLU()(label3)
#         out_label = self.out(label)
#
#         if in_feat:
#             return label1, label2, label3, out_label
#         else:
#             return out_label



"""这个是把四个特征提取方法放在一起的版本"""
# import torch
# import torch.nn as nn
# from util import PositionalEncoding, AttentionEncode, FAN_encode
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
#         #self.embed = load_pretrained_embedding(vocab_size, embedding_size)
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
#         # self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=2,
#         #                              stride=1
#         #                              )
#         # self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=4,
#         #                              stride=1
#         #                              )
#         # self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=6,
#         #                              stride=1
#         #                              )
#         # self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
#         #                              out_channels=64,
#         #                              kernel_size=8,
#         #                              stride=1
#         #                              )
#         # self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
#         # self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         # self.fan = FAN_encode(self.dropout, 1664)
#         # self.fc_aai = nn.Linear(531, self.embedding_size)
#         # self.fc_paac = nn.Linear(3, self.embedding_size)
#         # self.fc_pc6 = nn.Linear(6, self.embedding_size)
#         # self.fc_blosum62 = nn.Linear(23, self.embedding_size)
#         # self.fc_aac = nn.Linear(20,self.embedding_size)
#         # self.full3 = nn.Linear(1664, 1000)
#         # self.full4 = nn.Linear(1000, 500)
#         # self.full5 = nn.Linear(500, 256)
#         # self.Flatten = nn.Linear(256, 64)
#         # self.out = nn.Linear(64, self.output_size)
#
#
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=3,
#                                      stride=1
#                                      )
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=6,
#                                      stride=1
#                                      )
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=9,
#                                      stride=1
#                                      )
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
#                                      out_channels=64,
#                                      kernel_size=12,
#                                      stride=1
#                                      )
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
#         self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
#         self.fan = FAN_encode(self.dropout, 1536)
#         self.fc_aai = nn.Linear(531, self.embedding_size)
#         self.fc_paac = nn.Linear(3, self.embedding_size)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size)
#         self.fc_aac = nn.Linear(20,self.embedding_size)
#         self.full3 = nn.Linear(1536, 1000)
#         self.full4 = nn.Linear(1000, 500)
#         self.full5 = nn.Linear(500, 256)
#         self.Flatten = nn.Linear(256, 64)
#         self.out = nn.Linear(64, self.output_size)
#
#
#
#         self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,batch_first=True, bidirectional=True)
#
#         #卷积下面加用这个# class ETFC(nn.Module):
# #     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
# #         super(ETFC, self).__init__()
# #         self.vocab_size = vocab_size
# #         self.embedding_size = embedding_size
# #         self.output_size = output_size
# #         self.dropout = dropout
# #         self.fan_epoch = fan_epoch
# #         self.num_heads = num_heads
# #         self.max_pool = max_pool
# #
# #         # 设备
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# #         # 根据 max_pool 设置形状
# #         if max_pool == 2:
# #             shape = 6016
# #         elif max_pool == 3:
# #             shape = 3968
# #         elif max_pool == 4:
# #             shape = 2944
# #         elif max_pool == 5:
# #             shape = 2304
# #         else:
# #             shape = 1920
# #
# #         # 定义嵌入层
# #         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
# #
# #         # 定位编码
# #         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout).to(self.device)
# #
# #         # 卷积层定义
# #         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
# #         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
# #         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
# #         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
# #
# #         # 最大池化层
# #         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
# #
# #         # 注意力编码
# #         self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads).to(self.device)
# #
# #         # FAN 编码
# #         self.fan = FAN_encode(self.dropout, 1664).to(self.device)
# #
# #         # 特征嵌入
# #         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
# #         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
# #         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
# #         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
# #         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
# #
# #         # 全连接层
# #         self.full3 = nn.Linear(1664, 1000).to(self.device)
# #         self.full4 = nn.Linear(1000, 500).to(self.device)
# #         self.full5 = nn.Linear(500, 256).to(self.device)
# #         self.Flatten = nn.Linear(256, 64).to(self.device)
# #         self.out = nn.Linear(64, self.output_size).to(self.device)
# #
# #         # 双向 LSTM 层
# #         self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
# #
# #         # Dropout 层
# #         self.dropout = torch.nn.Dropout(self.dropout).to(self.device)
# #
# #
# #
# #     def TextCNN(self, x):
# #         x = x.to(self.device)
# #         x1 = self.conv1(x)
# #         x1 = torch.nn.ReLU()(x1)
# #         x1 = self.MaxPool1d(x1)
# #
# #         x2 = self.conv2(x)
# #         x2 = torch.nn.ReLU()(x2)
# #         x2 = self.MaxPool1d(x2)
# #
# #         x3 = self.conv3(x)
# #         x3 = torch.nn.ReLU()(x3)
# #         x3 = self.MaxPool1d(x3)
# #
# #         x4 = self.conv4(x)
# #         x4 = torch.nn.ReLU()(x4)
# #         x4 = self.MaxPool1d(x4)
# #
# #         # 拼接卷积层的输出
# #         y = torch.cat([x4, x2, x3], dim=-1)
# #         x = self.dropout(y)
# #         x = x.view(x.size(0), -1)
# #         return x
# #
# #     def forward(self, train_data, valid_lens, features):
# #         # 将所有输入数据移动到设备
# #         train_data = train_data.to(self.device)
# #
# #         # n, c = train_data.shape
# #         # hidden_slice_row = torch.zeros(n, 50).to(self.device)
# #         # hidden_slice_col = torch.zeros(n, 50).to(self.device)
# #         # slice_len = 10
# #         # train_data = self.hvgsu(hidden_slice_row, hidden_slice_col, train_data, n, 0, slice_len)
# #
# #         embed_output = self.embed(train_data)
# #         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
# #
# #         # 注意力编码
# #         attention_output = self.attention_encode(pos_output)
# #
# #         # 特征嵌入
# #         aai_features = self.fc_aai(features['aai'])
# #         paac_features = self.fc_paac(features['paac'])
# #         pc6_features = self.fc_pc6(features['pc6'])
# #         blosum62_features = self.fc_blosum62(features['blosum62'])
# #         aac_features = self.fc_aac(features['aac'])
# #
# #         # 特征注意力编码
# #         attention_aai_output = self.attention_encode(aai_features)
# #         attention_paac_output = self.attention_encode(paac_features)
# #         attention_pc6_output = self.attention_encode(pc6_features)
# #         attention_blosum62_output = self.attention_encode(blosum62_features)
# #         attention_aac_output = self.attention_encode(aac_features)
# #
# #
# #
# #         # 融合所有特征
# #         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
# #         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
# #                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
# #
# #
# #         # 经过双向 LSTM 层
# #         #vectors, _ = self.bilstm(combined_features)
# #         vectors= combined_features
# #         cnn_input = vectors.permute(0, 2, 1)
# #
# #         # 经过卷积层
# #         cnn_output = self.TextCNN(cnn_input)
# #
# #         # 经过 FAN 编码
# #         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
# #         for i in range(self.fan_epoch):
# #             fan_encode = self.fan(fan_encode)
# #
# #         # 经过全连接层
# #         out = fan_encode.squeeze()
# #
# #         label = self.full3(out)
# #         label = torch.nn.ReLU()(label)
# #         label = self.full4(label)
# #         label = torch.nn.ReLU()(label)
# #         label = self.full5(label)
# #         label = torch.nn.ReLU()(label)
# #         label = self.Flatten(label)
# #         label = torch.nn.ReLU()(label)
# #         out_label = self.out(label)
# #
# #         return out_label
#         #self.bilstm = nn.LSTM(input_size=2304, hidden_size=1152, num_layers=1, batch_first=True, bidirectional=True)
#
#
#
#         self.dropout = torch.nn.Dropout(self.dropout)
#
#
#     def TextCNN(self, x):
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#         #y = torch.cat([x1, x2, x3, x4], dim=-1)
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(self.embed(train_data) * math.sqrt(self.embedding_size))
#         '''-----------------------------------------------------'''
#
#         '''----------------------attention----------------------'''
#         # 进行注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#         '''-----------------------------------------------------'''
#
#         '''----------------------特征相加-------------------------'''
#
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         attention_aai_output = self.attention_encode(aai_features)
#         attention_paac_output = self.attention_encode(paac_features)
#         attention_pc6_output = self.attention_encode(pc6_features)
#         attention_blosum62_output = self.attention_encode(blosum62_features)
#         attention_aac_output = self.attention_encode(aac_features)
#
#
#
#         #combined_features = aai_features + embed_output
#
#         #combined_features = embed_output + paac_features + blosum62_features + pc6_features + attention_output + aai_features
#         #combined_features = embed_output +attention_output +attention_aai_output+attention_paac_output+attention_pc6_output+attention_blosum62_output
#
#         fusion_model = FeatureFusion(256, 256, 7).to(DEVICE)
#
#         # 融合特征
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                       attention_pc6_output, attention_blosum62_output,attention_aac_output])
#
#
#         vectors = combined_features
#         vectors, _ = self.bilstm(vectors)  # x 的形状：[200, 50, 256]，因为双向每方向128
#         cnn_input = vectors.permute(0, 2, 1)
#         cnn_output = self.TextCNN(cnn_input)
#
#
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#         out = fan_encode.squeeze()
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label1 = self.full4(label)
#         label = torch.nn.ReLU()(label1)
#         label2 = self.full5(label)
#         label = torch.nn.ReLU()(label2)
#         label3 = self.Flatten(label)
#         label = torch.nn.ReLU()(label3)
#         out_label = self.out(label)
#         return out_label



#魔改


import torch
import torch.nn as nn
import math

# 请确保导入或定义了以下类和函数
# from positional_encoding import PositionalEncoding
# from attention_encode import AttentionEncode
# from fan_encode import FAN_encode
# from feature_fusion import FeatureFusion



# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size  # 动态调整的 embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout  # 使用 dropout_value 避免与 nn.Dropout 冲突
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 根据 max_pool 设置形状（您可以根据实际需要调整）
#         shape = 1664  # 需要根据模型的输出调整
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 卷积层定义，使用动态的 embedding_size
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # FAN 编码
#         self.fan = FAN_encode(self.dropout_value, shape).to(self.device)
#
#         # 特征嵌入，使用动态的 embedding_size
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 全连接层，输入大小需要根据前一层的输出调整
#         self.full3 = nn.Linear(shape, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         # 双向 LSTM 层，input_size 使用动态的 embedding_size
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         # 拼接卷积层的输出
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])  # 访问 'aai' 特征
#
#         paac_features = self.fc_paac(features['paac'])  # 访问 'paac' 特征
#         pc6_features = self.fc_pc6(features['pc6'])  # 访问 'pc6' 特征
#         blosum62_features = self.fc_blosum62(features['blosum62'])  # 访问 'blosum62' 特征
#         aac_features = self.fc_aac(features['aac'])  # 访问 'aac' 特征
#
#         # 特征注意力编码
#         attention_aai_output = self.attention_encode(aai_features)
#         attention_paac_output = self.attention_encode(paac_features)
#         attention_pc6_output = self.attention_encode(pc6_features)
#         attention_blosum62_output = self.attention_encode(blosum62_features)
#         attention_aac_output = self.attention_encode(aac_features)
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(self.embedding_size, self.embedding_size, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         # 经过双向 LSTM 层
#         vectors = combined_features
#         cnn_input = vectors.permute(0, 2, 1)
#
#         # 经过卷积层
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过全连接层
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label  # 输出形状应为 [batch_size, num_classes]


#PSO训练用
# import torch
# import torch.nn as nn
# import math
#
# # 请确保导入或定义了以下类和函数
# # from positional_encoding import PositionalEncoding
# # from attention_encode import AttentionEncode
# # from fan_encode import FAN_encode
# # from feature_fusion import FeatureFusion
#
# class GaussianFuzzyLayer(nn.Module):
#     """
#     高斯隶属度函数层，用于模糊化处理。
#     """
#     def __init__(self, input_dim, output_dim):
#         super(GaussianFuzzyLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         # 定义高斯隶属度函数的中心和宽度（这些参数可以训练）
#         self.centers = nn.Parameter(torch.randn(output_dim))  # 高斯函数的中心
#         self.widths = nn.Parameter(torch.ones(output_dim))    # 高斯函数的宽度
#
#     def forward(self, x):
#         x = self.fc(x)  # 先经过线性层
#         # 使用高斯隶属度函数进行模糊化
#         x = torch.exp(-((x - self.centers) ** 2) / (2 * (self.widths ** 2)))
#         return x
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size  # 动态调整的 embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout  # 使用 dropout_value 避免与 nn.Dropout 冲突
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 根据 max_pool 设置形状（您可以根据实际需要调整）
#         shape = 1664  # 需要根据模型的输出调整
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 卷积层定义，使用动态的 embedding_size
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # FAN 编码
#         self.fan = FAN_encode(self.dropout_value, shape).to(self.device)
#
#
#         # 特征嵌入，使用动态的 embedding_size
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 全连接层，输入大小需要根据前一层的输出调整
#         self.full3 = nn.Linear(shape, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         # 双向 LSTM 层，input_size 使用动态的 embedding_size
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#         # 使用高斯隶属函数的模糊化层
#         self.fuzzy_layer = GaussianFuzzyLayer(256,256).to(self.device)
#
#
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         # 拼接卷积层的输出
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])  # 访问 'aai' 特征
#
#         paac_features = self.fc_paac(features['paac'])  # 访问 'paac' 特征
#         pc6_features = self.fc_pc6(features['pc6'])  # 访问 'pc6' 特征
#         blosum62_features = self.fc_blosum62(features['blosum62'])  # 访问 'blosum62' 特征
#         aac_features = self.fc_aac(features['aac'])  # 访问 'aac' 特征
#
#         # 特征注意力编码
#         attention_aai_output = self.attention_encode(aai_features)
#         attention_paac_output = self.attention_encode(paac_features)
#         attention_pc6_output = self.attention_encode(pc6_features)
#         attention_blosum62_output = self.attention_encode(blosum62_features)
#         attention_aac_output = self.attention_encode(aac_features)
#
#         attention_aai_output = self.fuzzy_layer(attention_aai_output)
#         attention_paac_output = self.fuzzy_layer(attention_paac_output)
#         attention_pc6_output = self.fuzzy_layer(attention_pc6_output)
#         attention_blosum62_output = self.fuzzy_layer(attention_blosum62_output)
#         attention_aac_output = self.fuzzy_layer(attention_aac_output)
#
#
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(self.embedding_size, self.embedding_size, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         # 经过双向 LSTM 层
#         vectors = combined_features
#         cnn_input = vectors.permute(0, 2, 1)
#
#         # 经过卷积层
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过全连接层
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label  # 输出形状应为 [batch_size, num_classes]


# class GaussianFuzzyLayer(nn.Module):
#     """
#     高斯隶属度函数层，用于模糊化处理。
#     """
#     def __init__(self, input_dim, output_dim):
#         super(GaussianFuzzyLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         # 定义高斯隶属度函数的中心和宽度（这些参数可以训练）
#         self.centers = nn.Parameter(torch.randn(output_dim))  # 高斯函数的中心
#         self.widths = nn.Parameter(torch.ones(output_dim))    # 高斯函数的宽度
#
#     def forward(self, x):
#         x = self.fc(x)  # 先经过线性层
#         # 使用高斯隶属度函数进行模糊化
#         x = torch.exp(-((x - self.centers) ** 2) / (2 * (self.widths ** 2)))
#         return x
#
#
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 根据 max_pool 设置形状
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 卷积层定义
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # FAN 编码
#         self.fan = FAN_encode(self.dropout_value,1664).to(self.device)
#
#         # 特征嵌入
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 全连接层
#         self.full3 = nn.Linear(1664, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         # 双向 LSTM 层
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#         # 使用高斯隶属函数的模糊化层
#         self.fuzzy_layer = GaussianFuzzyLayer(256,256).to(self.device)
#
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         # 拼接卷积层的输出
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         #x = self.fuzzy_layer(x)
#         return x
#
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         # 特征注意力编码
#         # attention_aai_output = self.attention_encode(aai_features)
#         # attention_paac_output = self.attention_encode(paac_features)
#         # attention_pc6_output = self.attention_encode(pc6_features)
#         # attention_blosum62_output = self.attention_encode(blosum62_features)
#         # attention_aac_output = self.attention_encode(aac_features)
#
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#
#         lstm_output, _ = self.bilstm(combined_features)  # 根据 LSTM 的输入维度调整参数
#
#
#         # 对 LSTM 的输出进行处理，作为 CNN 的输入
#         cnn_input = lstm_output.permute(0, 2, 1)  # [batch_size, 256, seq_len]
#
#         # 经过 CNN
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过全连接层
#
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#
#         return out_label


#GAN

# import torch
# import torch.nn as nn
# import math
#
# # 确保以下模块已正确导入或定义
# # from your_module import PositionalEncoding, AttentionEncode, FAN_encode, GaussianFuzzyLayer, FeatureFusion
#
# class Generator(nn.Module):
#     def __init__(self, noise_dim, seq_length, vocab_size, temperature=1.0):
#         super(Generator, self).__init__()
#         self.seq_length = seq_length
#         self.vocab_size = vocab_size
#         self.temperature = temperature
#         self.model = nn.Sequential(
#             nn.Linear(noise_dim, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.Linear(512, seq_length * vocab_size)
#         )
#
#     def forward(self, z):
#         output = self.model(z)
#         output = output.view(-1, self.seq_length, self.vocab_size)
#         # 使用 Gumbel-Softmax 采样
#         gen_data = F.gumbel_softmax(output, tau=self.temperature, hard=True, dim=-1)
#         return gen_data  # 返回 one-hot 向量序列
#
# class Discriminator(nn.Module):
#     def __init__(self, seq_length, vocab_size):
#         super(Discriminator, self).__init__()
#         self.seq_length = seq_length
#         self.vocab_size = vocab_size
#
#         self.model = nn.Sequential(
#             nn.Linear(seq_length * vocab_size, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         # 输入的数据形状为 (batch_size, seq_length, vocab_size)
#         data = data.view(data.size(0), -1)  # 展平
#         validity = self.model(data)
#         return validity
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         self.fan = FAN_encode(self.dropout_value,1664).to(self.device)
#
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         self.full3 = nn.Linear(1664, 1000).to(self.device)
#         self.full4 = nn.Linear(1000, 500).to(self.device)
#         self.full5 = nn.Linear(500, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#         self.fuzzy_layer = GaussianFuzzyLayer(256,256).to(self.device)
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = torch.nn.ReLU()(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = torch.nn.ReLU()(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = torch.nn.ReLU()(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = torch.nn.ReLU()(x4)
#         x4 = self.MaxPool1d(x4)
#
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         attention_output = self.attention_encode(pos_output)
#
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#         # attention_aai_output = self.attention_encode(aai_features)
#         # attention_paac_output = self.attention_encode(paac_features)
#         # attention_pc6_output = self.attention_encode(pc6_features)
#         # attention_blosum62_output = self.attention_encode(blosum62_features)
#         # attention_aac_output = self.attention_encode(aac_features)
#
#
#         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         lstm_output, _ = self.bilstm(combined_features)
#
#         cnn_input = lstm_output.permute(0, 2, 1)
#
#
#         cnn_output = self.TextCNN(cnn_input)
#
#
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         out = fan_encode.squeeze()
#
#         label = self.full3(out)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label





#用BAS（Beetle Antennae Search）进行超参优化，还有CNN,LSTM的超参也一起优化了

# import torch
# import torch.nn as nn
# import math
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads,
#                  conv_channels=64, kernel_sizes=[2, 4, 6, 8], max_pool=2,
#                  lstm_hidden_size=128, lstm_num_layers=1, lstm_dropout=0.0):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#         self.conv_channels = conv_channels
#         self.kernel_sizes = kernel_sizes  # 支持多种卷积核大小
#
#         # 设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 定义嵌入层
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#
#         # 定位编码
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         # 注意力编码
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#
#         # 特征嵌入
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # 使用高斯隶属函数的模糊化层
#         self.fuzzy_layer = GaussianFuzzyLayer(256, 256).to(self.device)
#
#         # 融合所有特征
#         self.fusion_model = FeatureFusion(self.embedding_size, self.embedding_size, 7).to(self.device)
#
#         # 双向 LSTM 层
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
#                               dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
#                               batch_first=True, bidirectional=True).to(self.device)
#
#         # 卷积层定义，支持多种卷积核大小
#         # 修改 in_channels 为 lstm_hidden_size * 2
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=lstm_hidden_size * 2, out_channels=self.conv_channels, kernel_size=ks).to(self.device)
#             for ks in self.kernel_sizes
#         ])
#
#         # 最大池化层
#         self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         # Dropout 层
#         self.dropout = torch.nn.Dropout(self.dropout_value).to(self.device)
#
#         # 全连接层将在 forward 中动态定义
#         self.full3 = None
#         self.full4 = None
#         self.full5 = None
#         self.Flatten = None
#         self.out = None
#
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         conv_outputs = []
#         for conv in self.convs:
#             x_conv = conv(x)
#             x_conv = torch.nn.ReLU()(x_conv)
#             x_conv = self.MaxPool1d(x_conv)
#             conv_outputs.append(x_conv)
#         # 拼接卷积层的输出
#         y = torch.cat(conv_outputs, dim = -1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         # 特征模糊化处理
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#         # 融合所有特征
#         combined_features = self.fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                               attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         lstm_output, _ = self.bilstm(combined_features)  # [batch, seq_len, lstm_hidden_size * 2]
#
#         # 对 LSTM 的输出进行处理，作为 CNN 的输入
#         cnn_input = lstm_output.permute(0, 2, 1)  # [batch_size, lstm_hidden_size*2, seq_len]
#
#         # 经过 CNN
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 这里根据 cnn_output 的形状动态定义全连接层
#         if self.full3 is None:
#             fc_input_dim = cnn_output.size(1)
#             self.full3 = nn.Linear(fc_input_dim, 1000).to(self.device)
#             self.full4 = nn.Linear(1000, 500).to(self.device)
#             self.full5 = nn.Linear(500, 256).to(self.device)
#             self.Flatten = nn.Linear(256, 64).to(self.device)
#             self.out = nn.Linear(64, self.output_size).to(self.device)
#             # 初始化权重
#             nn.init.xavier_uniform_(self.full3.weight)
#             nn.init.xavier_uniform_(self.full4.weight)
#             nn.init.xavier_uniform_(self.full5.weight)
#             nn.init.xavier_uniform_(self.Flatten.weight)
#             nn.init.xavier_uniform_(self.out.weight)
#
#         # 经过全连接层
#         label = self.full3(cnn_output)
#         label = torch.nn.ReLU()(label)
#         label = self.full4(label)
#         label = torch.nn.ReLU()(label)
#         label = self.full5(label)
#         label = torch.nn.ReLU()(label)
#         label = self.Flatten(label)
#         label = torch.nn.ReLU()(label)
#         out_label = self.out(label)
#
#         return out_label





#转成图卷积
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
# from torch_geometric.utils import add_self_loops
#
# class GNNModel(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, num_layers=3):
#         super(GNNModel, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#
#         # 词嵌入层
#         self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
#
#         # GCN层
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         for i in range(num_layers):
#             self.convs.append(GCNConv(self.embedding_size, self.embedding_size))
#             self.bns.append(BatchNorm(self.embedding_size))
#
#         # 全连接层
#         self.fc1 = nn.Linear(self.embedding_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, self.output_size)
#
#         self.dropout_layer = nn.Dropout(self.dropout)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         # 词嵌入
#         x = self.embedding(x)
#         #x = x.float()  # 转换为浮点型
#
#         # GCN层
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             #x = bn(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 全局池化
#         x = global_mean_pool(x, batch)  # [batch_size, embedding_size]
#
#         # 全连接层
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout_layer(x)
#
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout_layer(x)
#         #x = F.relu(x)
#         x = self.fc3(x)
#         return x


# model.py

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from torch_geometric.nn import GCNConv, GATConv, global_mean_pool  # 导入全局池化
#
# class GNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
#         super(GNN, self).__init__()
#         self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6)
#         self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=0.6)
#         self.dropout = nn.Dropout(0.6)
#
#     def forward(self, x, edge_index):
#         x = self.dropout(F.elu(self.gat1(x, edge_index)))
#         x = self.dropout(self.gat2(x, edge_index))
#         return x
#
# class ETFC(nn.Module):
#     def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, gnn_hidden_dim=128, gnn_output_dim=256, max_pool=5):
#         super(ETFC, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.output_size = output_size
#         self.dropout_value = dropout
#         self.fan_epoch = fan_epoch
#         self.num_heads = num_heads
#         self.max_pool = max_pool
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         if max_pool == 2:
#             shape = 6016
#         elif max_pool == 3:
#             shape = 3968
#         elif max_pool == 4:
#             shape = 2944
#         elif max_pool == 5:
#             shape = 2304
#         else:
#             shape = 1920
#
#         self.embed = nn.Embedding(self.vocab_size, self.embedding_size).to(self.device)
#         self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout_value).to(self.device)
#
#         self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=2, stride=1).to(self.device)
#         self.conv2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=4, stride=1).to(self.device)
#         self.conv3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=6, stride=1).to(self.device)
#         self.conv4 = nn.Conv1d(in_channels=self.embedding_size, out_channels=64, kernel_size=8, stride=1).to(self.device)
#
#         self.MaxPool1d = nn.MaxPool1d(kernel_size=self.max_pool).to(self.device)
#
#         self.attention_encode = AttentionEncode(self.dropout_value, self.embedding_size, self.num_heads).to(self.device)
#         self.fan = FAN_encode(self.dropout_value, 1664).to(self.device)
#
#         self.fc_aai = nn.Linear(531, self.embedding_size).to(self.device)
#         self.fc_paac = nn.Linear(3, self.embedding_size).to(self.device)
#         self.fc_pc6 = nn.Linear(6, self.embedding_size).to(self.device)
#         self.fc_blosum62 = nn.Linear(23, self.embedding_size).to(self.device)
#         self.fc_aac = nn.Linear(20, self.embedding_size).to(self.device)
#
#         # self.full3 = nn.Linear(1664, 1000).to(self.device)
#         # self.full4 = nn.Linear(1000, 500).to(self.device)
#         # self.full5 = nn.Linear(500, 256).to(self.device)
#         # self.Flatten = nn.Linear(256, 64).to(self.device)
#         # self.out = nn.Linear(64, self.output_size).to(self.device)
#
#         self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
#
#         self.dropout = nn.Dropout(self.dropout_value).to(self.device)
#         self.fuzzy_layer = GaussianFuzzyLayer(256, 256).to(self.device)
#
#         self.gnn = GNN(input_dim=256, hidden_dim=gnn_hidden_dim, output_dim=gnn_output_dim, num_heads=num_heads).to(self.device)
#
#         self.full3 = nn.Linear(1664 + gnn_output_dim, 1024).to(self.device)
#         self.full4 = nn.Linear(1024, 512).to(self.device)
#         self.full5 = nn.Linear(512, 256).to(self.device)
#         self.Flatten = nn.Linear(256, 64).to(self.device)
#         self.out = nn.Linear(64, self.output_size).to(self.device)
#     def TextCNN(self, x):
#         x = x.to(self.device)
#         x1 = self.conv1(x)
#         x1 = F.relu(x1)
#         x1 = self.MaxPool1d(x1)
#
#         x2 = self.conv2(x)
#         x2 = F.relu(x2)
#         x2 = self.MaxPool1d(x2)
#
#         x3 = self.conv3(x)
#         x3 = F.relu(x3)
#         x3 = self.MaxPool1d(x3)
#
#         x4 = self.conv4(x)
#         x4 = F.relu(x4)
#         x4 = self.MaxPool1d(x4)
#
#         y = torch.cat([x4, x2, x3], dim=-1)
#         x = self.dropout(y)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, train_data, valid_lens, features, edge_index, gnn_features, batch):
#         # 将所有输入数据移动到设备
#         train_data = train_data.to(self.device)
#
#         # 序列模型部分
#         embed_output = self.embed(train_data)
#         pos_output = self.pos_encoding(embed_output * math.sqrt(self.embedding_size))
#
#         # 注意力编码
#         attention_output = self.attention_encode(pos_output)
#
#         # 特征嵌入
#         aai_features = self.fc_aai(features['aai'])
#         paac_features = self.fc_paac(features['paac'])
#         pc6_features = self.fc_pc6(features['pc6'])
#         blosum62_features = self.fc_blosum62(features['blosum62'])
#         aac_features = self.fc_aac(features['aac'])
#
#         attention_aai_output = self.fuzzy_layer(aai_features)
#         attention_paac_output = self.fuzzy_layer(paac_features)
#         attention_pc6_output = self.fuzzy_layer(pc6_features)
#         attention_blosum62_output = self.fuzzy_layer(blosum62_features)
#         attention_aac_output = self.fuzzy_layer(aac_features)
#
#         # 融合所有特征
#         fusion_model = FeatureFusion(256, 256, 7).to(self.device)
#         combined_features = fusion_model([embed_output, attention_output, attention_aai_output, attention_paac_output,
#                                           attention_pc6_output, attention_blosum62_output, attention_aac_output])
#
#         lstm_output, _ = self.bilstm(combined_features)  # 根据 LSTM 的输入维度调整参数
#
#         # 对 LSTM 的输出进行处理，作为 CNN 的输入
#         cnn_input = lstm_output.permute(0, 2, 1)  # [batch_size, 256, seq_len]
#
#
#         # 经过 CNN
#         cnn_output = self.TextCNN(cnn_input)
#
#         # 经过 FAN 编码
#         fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
#         for i in range(self.fan_epoch):
#             fan_encode = self.fan(fan_encode)
#
#         # 经过 GNN
#         gnn_out = self.gnn(gnn_features, edge_index)  # [num_nodes, 256]
#         #print(gnn_out.shape)
#
#         # 对 GNN 输出进行全局平均池化，获得每个图的图级特征
#         gnn_pooled = global_mean_pool(gnn_out, batch)  # [batch_size, output_dim]
#         #print(gnn_pooled.shape)
#
#
#         # 融合序列特征和 GNN 特征
#         combined = torch.cat([fan_encode.squeeze(), gnn_pooled], dim=1)
#         #print(combined.shape)
#         combined = self.full3(combined)
#         combined = F.relu(combined)
#         combined = self.full4(combined)
#         combined = F.relu(combined)
#         combined = self.full5(combined)
#         combined = F.relu(combined)
#         combined = self.Flatten(combined)
#         combined = F.relu(combined)
#         out_label = self.out(combined)
#
#         return out_label



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


# import torch
# import torch.nn as nn
#
# class GaussianFuzzyLayer(nn.Module):
#     """
#     高斯隶属度函数层，用于模糊化处理。
#     """
#     def __init__(self, input_dim, output_dim, dropout_value = 0.6):
#         super(GaussianFuzzyLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.centers = nn.Parameter(torch.randn(output_dim))  # 高斯函数的中心
#         self.widths = nn.Parameter(torch.ones(output_dim))    # 高斯函数的宽度
#         self.dropout = nn.Dropout(dropout_value)  # 添加 Dropout 层
#
#     def forward(self, x):
#         x = self.fc(x)  # 先经过线性层
#         x = torch.exp(-((x - self.centers) ** 2) / (2 * (self.widths ** 2)))  # 使用高斯隶属度函数进行模糊化
#         x = self.dropout(x)  # 应用 Dropout
#         return x


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




class MFFTPC(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, gnn_hidden_dim=128, gnn_output_dim=256, max_pool=5):
        super(MFFTPC, self).__init__()
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

        att_aai_output = self.attention_encode(aai_features)
        att_paac_output = self.attention_encode(paac_features)
        att_pc6_output = self.attention_encode(pc6_features)
        att_blosum62_output = self.attention_encode(blosum62_features)
        att_aac_output = self.attention_encode(aac_features)


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

        combined_features = self.fusion_model([embed_output,attention_output ,fuzzy_aai_output, fuzzy_paac_output,
                                          fuzzy_pc6_output, fuzzy_blosum62_output, fuzzy_aac_output,gnn_out])


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






