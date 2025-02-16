# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:28:17 2022

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 02:19:52 2021

@author: joy
"""


import numpy as np
import torch
import pandas as pd
from Bio import SeqIO
import torch.utils.data
from sklearn.model_selection import KFold, ShuffleSplit
import re
import vocab
from sklearn.model_selection import train_test_split
# from transformers import T5Tokenizer,XLNetTokenizer

import numpy as np
import torch

# def AAI_embedding(seq, max_len=200):
#     f = open('data/AAindex.txt')
#     text = f.read()
#     f.close()
#     text = text.split('\n')
#     while '' in text:
#         text.remove('')
#     cha = text[0].split('\t')
#     while '' in cha:
#         cha.remove('')
#     cha = cha[1:]
#     index = []
#     for i in range(1, len(text)):
#         temp = text[i].split('\t')
#         while '' in temp:
#             temp.remove('')
#         temp = temp[1:]
#         for j in range(len(temp)):
#             temp[j] = float(temp[j])
#         index.append(temp)
#     index = np.array(index)
#     AAI_dict = {cha[j]: index[:, j] for j in range(len(cha))}
#     AAI_dict['X'] = np.zeros(531)
#     all_embeddings = []
#     for each_seq in seq:
#         temp_embeddings = [AAI_dict.get(each_char, np.zeros(531)) for each_char in each_seq]
#         if max_len > len(each_seq):
#             zero_padding = np.zeros((max_len - len(each_seq), 531))
#             data_pad = np.vstack((temp_embeddings, zero_padding))
#         elif max_len == len(each_seq):
#             data_pad = temp_embeddings
#         else:
#             data_pad = temp_embeddings[:max_len]
#         all_embeddings.append(data_pad)
#     all_embeddings = np.array(all_embeddings)
#     return torch.from_numpy(all_embeddings).float()
# def BLOSUM62_embedding(seq,max_len=200):
#     f=open('data/blosum62.txt')
#     text=f.read()
#     f.close()
#     text=text.split('\n')
#     while '' in text:
#         text.remove('')
#     cha=text[0].split(' ')
#     while '' in cha:
#         cha.remove('')
#     index=[]
#     for i in range(1,len(text)):
#         temp=text[i].split(' ')
#         while '' in temp:
#             temp.remove('')
#         for j in range(len(temp)):
#             temp[j]=float(temp[j])
#         index.append(temp)
#     index=np.array(index)
#     BLOSUM62_dict={}
#     for j in range(len(cha)):
#         BLOSUM62_dict[cha[j]]=index[:,j]
#     all_embeddings=[]
#     for each_seq in seq:
#         temp_embeddings=[]
#         for each_char in each_seq:
#             temp_embeddings.append(BLOSUM62_dict[each_char])
#         if max_len>len(each_seq):
#             zero_padding=np.zeros((max_len-len(each_seq),23))
#             data_pad=np.vstack((temp_embeddings,zero_padding))
#         elif max_len==len(each_seq):
#             data_pad=temp_embeddings
#         else:
#             data_pad=temp_embeddings[:max_len]
#         all_embeddings.append(data_pad)
#     all_embeddings=np.array(all_embeddings)
#     return torch.from_numpy(all_embeddings).float()
#
#
# def onehot_embedding(seq,max_len=200):
#     char_list='ARNDCQEGHILKMFPSTWYVX'
#     char_dict={}
#     for i in range(len(char_list)):
#         char_dict[char_list[i]]=i
#     all_embeddings=[]
#     for each_seq in seq:
#         temp_embeddings=[]
#         for each_char in each_seq:
#             codings=np.zeros(21)
#             if each_char in char_dict.keys():
#                 codings[char_dict[each_char]]=1
#             else:
#                 codings[20]=1
#             #如果 each_char 存在于 char_dict 中，将相应的位置设置为1；否则，将最后一个位置（索引为20）设置为1，表示未知氨基酸。
#             temp_embeddings.append(codings)
#         if max_len>len(each_seq):
#             zero_padding=np.zeros((max_len-len(each_seq),21))
#             data_pad=np.vstack((temp_embeddings,zero_padding))
#         elif max_len==len(each_seq):
#             data_pad=temp_embeddings
#         else:
#             data_pad=temp_embeddings[:max_len]
#
#         all_embeddings.append(data_pad)
#     all_embeddings=np.array(all_embeddings)
#     return torch.from_numpy(all_embeddings).float()
#
#
#
#
# def index_encoding(sequences,max_len=200):
#     '''
#     Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L110-L130
#
#     Parameters
#     ----------
#     sequences: list of equal-length sequences
#
#     Returns
#     -------
#     np.array with shape (#sequences, length of sequences)
#     '''
#     seq_list=[]
#     #创建一个空列表 seq_list，用于存储编码后的序列
#     for s in sequences:
#         temp=list(s)
#         while len(temp)<max_len:
#             temp.append(20)
#         temp=temp[:max_len]
#         seq_list.append(temp)
# #遍历输入的 sequences 列表中的每一个序列 s。
# # 将序列 s 转换为列表并赋值给临时变量 temp。
# # 使用循环，向 temp 列表中添加数值 20，直到 temp 的长度达到 max_len。
# # 通过切片操作，将 temp 列表的长度截断为 max_len。
# # 将处理好的 temp 列表添加到 seq_list 列表中。
#
#
#     df = pd.DataFrame(seq_list)
#     #使用 pd.DataFrame 创建一个数据帧 df，其中每一行是 seq_list 中的一个子列表。
#     encoding = df.replace(vocab.AMINO_ACID_INDEX)
#     #使用 df.replace(vocab.AMINO_ACID_INDEX) 将数据帧中的每个元素替换为 vocab.AMINO_ACID_INDEX 中对应的值。
#     # 这里的 vocab.AMINO_ACID_INDEX 是一个字典或映射，用于将氨基酸序列映射到整数索引
#     encoding = encoding.values.astype(int)
#     #将编码后的数据帧转换为整数类型的 ndarray。
#     return encoding
#
#
#
#
#
#
#
# class MetagenesisData(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.data[index]
# # MetagenesisData类是一个PyTorch的数据集类（torch.utils.data.Dataset），用于处理数据集。它有三个方法：
# # __init__(self, data): 用于初始化数据集，接收一个数据列表 data。
# # __len__(self): 返回数据集的长度（数据列表的长度）。
# # __getitem__(self, index): 根据给定的索引 index 返回数据集中对应位置的数据。
#
# class Dataset(object):
#     def __init__(self,
#             fasta=None,
#             label=None,
#             sep=',',
#             #这是一个分隔符参数，用于指定在读取数据文件时使用的字段分隔符。默认值是逗号 ,，表示常见的 CSV 文件格式
#             use_phy_feat=False,
#             #这是一个布尔值，用于指示是否要使用物理特征。如果设置为 True，则表示将使用物理特征；如果设置为 False，则表示不使用物理特征。默认值是 False
#             phy_feat_fl=None,random_seed=42):
#             #如果 use_phy_feat 被设置为 True，则这是一个文件路径或者文件名，用于存储物理特征数据。
#             #random_seed=42: 这是一个整数值，用于设置随机数生成器的种子，以确保在每次运行代码时得到相同的随机结果。默认种子值是 42。
#     #构造函数，接受多个参数来配置数据集。
#         self.label = label
#         self.fasta = fasta
#         #存储FASTA文件的路径或内容。
#         self.sep=sep
#         #存储分隔符，默认为逗号 ,
#         self.use_phy_feat = use_phy_feat
#         # 一个布尔值，指示是否使用物理特征，默认为 False。
#         self.rng = np.random.RandomState(random_seed)
#         # 一个随机数生成器对象，使用NumPy库的RandomState，用于控制随机种子。
#         self.native_sequence =fasta
#         #存储FASTA文件的内容。
#         if use_phy_feat==True:
#             self.phy_feat_fl=phy_feat_fl
#             self.phy_feat=self._read_features()
#         #如果 use_phy_feat 的值为 True，那么会设置类的属性 self.phy_feat_fl 为 phy_feat_fl，并调用 _read_features() 方法来读取并存储物理特征数据
#         # 到 self.phy_feat 变量中。
#
#     def _read_features(self):
#         df=pd.read_csv(self.phy_feat_fl,sep='\t',index_col=0)
#         #这一行使用pandas库中的read_csv函数读取一个CSV文件。self.phy_feat_fl是文件路径，sep='\t'表示使用制表符（\t）作为字段之间的分隔符，
#         # index_col=0表示将第一列作为索引。这会创建一个包含CSV文件数据的数据框（DataFrame）
#         feat=df.values
#         feat=torch.from_numpy(feat.astype(np.float32))
#
#         return feat
#     #一个私有方法，用于读取物理特征文件，将其转换为PyTorch张量并返回。
#
#     def _read_native_sequence(self):
#         sequence=[]
#         for seq_record in SeqIO.parse(self.fasta, "fasta"):
#             sequence.append(str(seq_record.seq).upper())
#         #for seq_record in SeqIO.parse(self.fasta, "fasta"):：这是一个循环，遍历通过SeqIO.parse函数解析的FASTA文件中的每个序列记录。
#         # self.fasta是FASTA文件的路径，而"fasta"是告诉SeqIO.parse函数数据文件的格式是FASTA。
#         #sequence.append(str(seq_record.seq).upper())：对于每个序列记录，获取其序列（seq_record.seq），将其转换为字符串（str()），
#         # 并使用.upper()将所有字母转换为大写。然后，将这个处理过的序列添加到之前创建的sequence列表中。
#         return sequence
#     # 一个私有方法，用于读取FASTA文件中的序列。
#
#     def _read_labels(self):
#
#         df=pd.read_csv(self.label,sep=self.sep,index_col=0)
#         labels=df.values
#         return labels
#     # 一个私有方法，用于读取标签数据。index_col=0表示从csv的第1列读取
#
#
#     #将序列编码为onehot编码的形式。
#     def encode_seq_enc_onehot(self, sequences,max_length):
#         seq_enc = onehot_embedding(sequences, max_len=max_length)
#         return seq_enc
#     #将序列编码为BLOSUM62编码的形式。
#     def encode_seq_enc_BLOSUM62(self, sequences,max_length):
#         seq_enc = BLOSUM62_embedding(sequences, max_len=max_length)
#         return seq_enc
#     def encode_seq_enc_AAI(self, sequences,max_length):
#         seq_enc = AAI_embedding(sequences, max_len=max_length)
#         return seq_enc
#     def encode_seq_enc_PAAC(self, sequences,max_length):
#         seq_enc = PAAC_embedding(sequences, max_len=max_length)
#         return seq_enc
#     def encode_seq_enc_PC6(self, sequences,max_length):
#         seq_enc = PC6_embedding(sequences, max_len=max_length)
#         return seq_enc
#     # def encode_seq_enc_bertT5(self, sequences,max_length):
#     #     seq_enc = bertT5_embedding(sequences, max_len=max_length)
#     #     return seq_enc
#
#     def encode_seq_enc(self, sequences,max_length):
#
#         seq_enc = index_encoding(sequences,max_length)
#         seq_enc = torch.from_numpy(seq_enc.astype(float))
#         return seq_enc
#
#     # def encode_seq_token(self,sequences):
#     #     seq_enc=[]
#     #     for each_seq in sequences:
#     #         seq_enc.append(self.tokenizer.encode(each_seq))
#     #     return seq_enc
#
#
#     # def encode_glob_feat(self, sequences):
#     #     feat = self.tape_encoder.encode(sequences)
#     #     feat = torch.from_numpy(feat).float()
#     #     return feat
#
#     def build_data(self, max_length):
#
#         sequences = self._read_native_sequence()
#         #从对象中获取原生序列数据
#         seq_enc_onehot = self.encode_seq_enc_onehot(sequences,max_length=max_length)
#
#
#         # train_seq_enc_bertencoding,train_seq_mask = self.encode_seq_enc_bertT5(train_sequences, max_length=max_length)
#         # valid_seq_enc_bertencoding,valid_seq_mask = self.encode_seq_enc_bertT5(valid_sequences, max_length=max_length)
#         # test_seq_enc_bertencoding,test_seq_mask = self.encode_seq_enc_bertT5(test_sequences, max_length=max_length)
#
#         seq_enc_blosum62 = self.encode_seq_enc_BLOSUM62(sequences,max_length=max_length)
#         seq_enc_AAI = self.encode_seq_enc_AAI(sequences,max_length=max_length)
#         seq_enc_PAAC = self.encode_seq_enc_PAAC(sequences,max_length=max_length)
#         seq_enc_PC6 = self.encode_seq_enc_PC6(sequences,max_length=max_length)
#         seq_enc = self.encode_seq_enc(sequences,max_length=max_length)
#         labels = self._read_labels()
#
#
#         samples = []
#
#         # print(labels)
#         print(len(sequences))
#         for i in range(len(sequences)):
#             sample = {
#                 'sequence':sequences[i],
#                 'label':labels[i],
#                 #'seq_enc': seq_enc[i],
#                 'seq_enc_onehot': seq_enc_onehot[i],
#                 # 'seq_enc_bert':train_seq_enc_bertencoding[i],
#                 # 'seq_enc_mask':train_seq_mask[i],
#                 #'seq_enc_pc6': seq_enc_PC6[i],
#
#                 'seq_enc_BLOSUM62': seq_enc_blosum62[i],
#                 'seq_enc_PAAC': seq_enc_PAAC[i],
#                 'seq_enc_AAI': seq_enc_AAI[i]
#             }
#             # print(type(sequences[i]))
#             # print(type(seq_enc_PAAC[i]))
#             # print(type(seq_enc_AAI[i]))
#             # print(type(seq_enc_blosum62[i]))
#             # print(type(seq_enc_onehot[i]))
#             # print((type(labels[i])))
#
#             if self.use_phy_feat:
#                 sample['phy_feat'] = self.phy_feat[i]
#             #通过循环遍历所有原生序列，对于每个序列，创建一个字典 sample 包含了序列本身以及通过不同编码方式得到的表示。如果开启了物理特征的使用
#             # (self.use_phy_feat)，还会将对应的物理特征添加到 sample 中。
#             samples.append(sample)
#         data = MetagenesisData(samples)
#         #将每个 sample 添加到 samples 列表中。
#         #最后，基于构建的 samples 列表创建一个名为 MetagenesisData 的数据集对象，并将其返回。这个对象可能是用于训练机器学习模型的数据集对象
#
#         return data
#
#
#     def get_dataloader(self,  max_length=200,batch_size=128,
#                        resample_train_valid=False):
#
#         data = self.build_data(max_length)
#         data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
#     #torch.utils.data.DataLoader(data, batch_size=batch_size)：这一行创建了一个 PyTorch 数据加载器，该加载器用于加载数据对象 data
#         return data_loader
#         # return data
# #返回一个 PyTorch 数据加载器（torch.utils.data.DataLoader）。
#
#
# import torch
# import torch.nn as nn
#
# device = torch.device("cpu")
#
# if __name__ == '__main__':
#
#     dataset = Dataset(
#         fasta='examples/samples.fasta',
#         label='examples/samples_label.csv',
#         sep=',')
#     train_loader = dataset.get_dataloader(
#         batch_size=32, max_length=200)
#     #通过Dataset类创建了一个数据加载器 train_loader。数据集包含从名为 'static/uploads/Insect.fasta' 的 FASTA 文件中加载的序列数据，
#     # 并使用逗号作为分隔符。
#
#     # y_pred=[]
#     y_true = []
#     all_seqs = []
#     # for batch in train_loader:
#     #     seq = batch['sequence'].to(device)
#     #     y = batch['label'].to(device)
#     #     x = batch['seq_enc'].to(device).int()
#     #     AAI_feat = batch['seq_enc_AAI'].to(device)
#     #     onehot_feat = batch['seq_enc_onehot'].to(device)
#     #     BLOSUM62_feat = batch['seq_enc_BLOSUM62'].to(device)
#     #     PAAC_feat = batch['seq_enc_PAAC'].to(device)
#     #     bert_feat = batch['seq_enc_bert'].to(device)
#     #     bert_mask = batch['seq_enc_mask'].to(device)
#     #     ## 提取数据
#     #     outputs = net(AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat)
#     #     outputs = model(x)
#     #     #  # 模型推断
#     #     y_pred.extend(outputs.cpu().numpy())
#     #     y_true.extend(y.cpu().numpy())
#     #     all_seqs.extend(seq.cpu().numpy())
#         #将输出和标签添加到列表中
#         #模型的输出被添加到 y_pred 列表中，真实标签被添加到 y_true 列表中，序列数据被添加到 all_seqs 列表中。
# #     print(df.head())
# #     print(len(loader.__iter__()))
# #     (loader, df), (_, _) = dataset.get_dataloader('train_valid',
# #         batch_size=32, return_df=True, resample_train_valid=True)
# #     print(df.head())
# # print(len(train_loader.__iter__()))
# # loader, df = dataset.get_dataloader('test',
# #     batch_size=32, return_df=True, resample_train_valid=True)
# # print(next(loader.__iter__()))
#
# # sequence = ['GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ','GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ']
# # seq_tape=AAI_embedding(sequence)


import numpy as np
import torch
import re
from collections import Counter
# def AAI_embedding(seq, max_len=50):
#     f = open('data/AAindex.txt')
#     text = f.read()
#     f.close()
#     text = text.split('\n')
#     while '' in text:
#         text.remove('')
#     cha = text[0].split('\t')
#     while '' in cha:
#         cha.remove('')
#     cha = cha[1:]
#     index = []
#     for i in range(1, len(text)):
#         temp = text[i].split('\t')
#         while '' in temp:
#             temp.remove('')
#         temp = temp[1:]
#         for j in range(len(temp)):
#             temp[j] = float(temp[j])
#         index.append(temp)
#     index = np.array(index)
#     AAI_dict = {}
#     for j in range(len(cha)):
#         AAI_dict[cha[j]] = index[:, j]
#     AAI_dict['X'] = np.zeros(531)
#     all_embeddings = []
#     for each_seq in seq:
#         temp_embeddings = []
#         for each_char in each_seq:
#             temp_embeddings.append(AAI_dict[each_char])
#         if max_len > len(each_seq):
#             zero_padding = np.zeros((max_len - len(each_seq), 531))
#             data_pad = np.vstack((temp_embeddings, zero_padding))
#         elif max_len == len(each_seq):
#             data_pad = temp_embeddings
#         else:
#             data_pad = temp_embeddings[:max_len]
#         all_embeddings.append(data_pad)
#     all_embeddings = np.array(all_embeddings)
#     return torch.from_numpy(all_embeddings).float()

#跟上面的比加入了碰到不认识的字符把值变为0的功能
#将蛋白质序列转换为基于预定义的氨基酸指数（AAindex）的嵌入表示，帮助生物信息学模型捕捉氨基酸的复杂属性
def AAI_embedding(seq, max_len=50):
    # 读取AAindex.txt文件
    f = open('data/AAindex.txt')
    text = f.read()
    f.close()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')
    cha = text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha = cha[1:]
    index = []
    for i in range(1, len(text)):
        temp = text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp = temp[1:]
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)

    # 创建AAI字典
    AAI_dict = {}
    for j in range(len(cha)):
        AAI_dict[cha[j]] = index[:, j]

    # 添加默认的全零向量用于未知氨基酸字母
    AAI_dict['X'] = np.zeros(531)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在AAI_dict中，如果不在则使用全零向量
            temp_embeddings.append(AAI_dict.get(each_char, np.zeros(531)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 531))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

# def PAAC_embedding(seq, max_len=50):
#     f = open('data/PAAC.txt')
#     text = f.read()
#     f.close()
#     text = text.split('\n')
#     while '' in text:
#         text.remove('')
#     cha = text[0].split('\t')
#     while '' in cha:
#         cha.remove('')
#     cha = cha[1:]
#     index = []
#     for i in range(1, len(text)):
#         temp = text[i].split('\t')
#         while '' in temp:
#             temp.remove('')
#         temp = temp[1:]
#         for j in range(len(temp)):
#             temp[j] = float(temp[j])
#         index.append(temp)
#     index = np.array(index)
#     AAI_dict = {}
#     for j in range(len(cha)):
#         AAI_dict[cha[j]] = index[:, j]
#     AAI_dict['X'] = np.zeros(3)
#     all_embeddings = []
#     for each_seq in seq:
#         temp_embeddings = []
#         for each_char in each_seq:
#             temp_embeddings.append(AAI_dict[each_char])
#         if max_len > len(each_seq):
#             zero_padding = np.zeros((max_len - len(each_seq), 3))
#             data_pad = np.vstack((temp_embeddings, zero_padding))
#         elif max_len == len(each_seq):
#             data_pad = temp_embeddings
#         else:
#             data_pad = temp_embeddings[:max_len]
#         all_embeddings.append(data_pad)
#     all_embeddings = np.array(all_embeddings)
#     return torch.from_numpy(all_embeddings).float()

#跟上面的比加入了碰到不认识的字符把值变为0的功能
#将蛋白质序列转换为基于预定义的氨基酸组成（PAAC）的嵌入表示，可以帮助模型捕捉到序列的基本组成特征
def PAAC_embedding(seq, max_len=50):
    # 读取PAAC.txt文件
    with open('data/PAAC.txt') as f:
        text = f.read()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')
    cha = text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha = cha[1:]
    index = []
    for i in range(1, len(text)):
        temp = text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp = temp[1:]
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)

    # 创建PAAC字典
    PAAC_dict = {}
    for j in range(len(cha)):
        PAAC_dict[cha[j]] = index[:, j]

    # 添加默认的全零向量用于未知氨基酸字母
    PAAC_dict['X'] = np.zeros(3)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在PAAC_dict中，如果不在则使用全零向量
            temp_embeddings.append(PAAC_dict.get(each_char, np.zeros(3)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 3))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

# def PC6_embedding(seq, max_len=50):
#     f = open('data/6-pc')
#     text = f.read()
#     f.close()
#     text = text.split('\n')
#     while '' in text:
#         text.remove('')
#     text = text[1:]
#     AAI_dict = {}
#     for each_line in text:
#         temp = each_line.split(' ')
#         while '' in temp:
#             temp.remove('')
#         for i in range(1, len(temp)):
#             temp[i] = float(temp[i])
#         AAI_dict[temp[0]] = temp[1:]
#     AAI_dict['X'] = np.zeros(6)
#     all_embeddings = []
#     for each_seq in seq:
#         temp_embeddings = []
#         for each_char in each_seq:
#             temp_embeddings.append(AAI_dict[each_char])
#         if max_len > len(each_seq):
#             zero_padding = np.zeros((max_len - len(each_seq), 6))
#             data_pad = np.vstack((temp_embeddings, zero_padding))
#         elif max_len == len(each_seq):
#             data_pad = temp_embeddings
#         else:
#             data_pad = temp_embeddings[:max_len]
#         all_embeddings.append(data_pad)
#     all_embeddings = np.array(all_embeddings)
#     return torch.from_numpy(all_embeddings).float()

#跟上面的比加入了碰到不认识的字符把值变为0的功能
#将蛋白质序列转换成基于预定义的6个物理化学属性（PC6）的数值嵌入，通过使用PC6属性，模型可以学习到序列中氨基酸的物理化学特性差异
def PC6_embedding(seq, max_len=50):
    # 读取6-pc文件
    with open('data/6-pc') as f:
        text = f.read()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')
    text = text[1:]

    # 创建PC6字典
    PC6_dict = {}
    for each_line in text:
        temp = each_line.split(' ')
        while '' in temp:
            temp.remove('')
        for i in range(1, len(temp)):
            temp[i] = float(temp[i])
        PC6_dict[temp[0]] = temp[1:]

    # 添加默认的全零向量用于未知氨基酸字母
    PC6_dict['X'] = np.zeros(6)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在PC6_dict中，如果不在则使用全零向量
            temp_embeddings.append(PC6_dict.get(each_char, np.zeros(6)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 6))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

# def BLOSUM62_embedding(seq, max_len=50):
#     f = open('data/blosum62.txt')
#     text = f.read()
#     f.close()
#     text = text.split('\n')
#     while '' in text:
#         text.remove('')
#     cha = text[0].split(' ')
#     while '' in cha:
#         cha.remove('')
#     index = []
#     for i in range(1, len(text)):
#         temp = text[i].split(' ')
#         while '' in temp:
#             temp.remove('')
#         for j in range(len(temp)):
#             temp[j] = float(temp[j])
#         index.append(temp)
#     index = np.array(index)
#     BLOSUM62_dict = {}
#     for j in range(len(cha)):
#         BLOSUM62_dict[cha[j]] = index[:, j]
#     all_embeddings = []
#     for each_seq in seq:
#         temp_embeddings = []
#         for each_char in each_seq:
#             temp_embeddings.append(BLOSUM62_dict[each_char])
#         if max_len > len(each_seq):
#             zero_padding = np.zeros((max_len - len(each_seq), 23))
#             data_pad = np.vstack((temp_embeddings, zero_padding))
#         elif max_len == len(each_seq):
#             data_pad = temp_embeddings
#         else:
#             data_pad = temp_embeddings[:max_len]
#         all_embeddings.append(data_pad)
#     all_embeddings = np.array(all_embeddings)
#     return torch.from_numpy(all_embeddings).float()

#跟上面的比加入了碰到不认识的字符把值变为0的功能
#将蛋白质序列转换成基于 BLOSUM62 矩阵的数值嵌入。BLOSUM62 矩阵是一种广泛用于序列比较和生物信息学研究的得分矩阵，它表示不同氨基酸对替换的得分，通常用于序列比对中
def BLOSUM62_embedding(seq, max_len=50):
    # 读取blosum62.txt文件
    with open('data/blosum62.txt') as f:
        text = f.read()

    # 处理文件内容
    text = text.split('\n')
    while '' in text:
        text.remove('')

    # 获取氨基酸字符
    cha = text[0].split(' ')
    while '' in cha:
        cha.remove('')

    # 处理矩阵数据
    index = []
    for i in range(1, len(text)):
        temp = text[i].split(' ')
        while '' in temp:
            temp.remove('')
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)

    # 创建BLOSUM62字典
    BLOSUM62_dict = {}
    for j in range(len(cha)):
        BLOSUM62_dict[cha[j]] = index[:, j]

    # 添加默认的全零向量用于未知氨基酸字母
    BLOSUM62_dict['X'] = np.zeros(23)

    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            # 检查每个字符是否在BLOSUM62_dict中，如果不在则使用全零向量
            temp_embeddings.append(BLOSUM62_dict.get(each_char, np.zeros(23)))
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 23))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)

    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

#可能有用
#(batchsize,max_seqlen,20)

# def AAC_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY', max_len=50):
#     # Define the amino acids order if it's not provided
#     AA = order
#
#     # Prepare to collect the AAC encodings
#     aac_encodings = []
#
#     for seq in sequences:
#         # Remove potential '-' characters for alignment gaps
#         sequence = re.sub('-', '', seq)
#
#         # Ensure each sequence is exactly max_len characters
#         if len(sequence) > max_len:
#             sequence = sequence[:max_len]
#         elif len(sequence) < max_len:
#             sequence += 'X' * (max_len - len(sequence))  # Pad with 'X' if not enough length
#
#         # Initialize frequency list for this sequence
#         temp_encoding = np.zeros((max_len, len(AA)))
#
#         # Calculate frequency for each position up to max_len
#         for idx, char in enumerate(sequence):
#             if char in AA:
#                 aa_index = AA.index(char)
#                 temp_encoding[idx, aa_index] = 1  # Set 1 for the existing amino acid position
#
#         aac_encodings.append(temp_encoding)
#
#     # Convert list of encodings to a numpy array
#     all_encodings = np.array(aac_encodings)
#
#     return torch.tensor(all_encodings, dtype=torch.float)

#计算蛋白质序列中的氨基酸组成
def AAC_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY', max_len=50):
    # Define the amino acids order if it's not provided
    AA = order

    # Prepare to collect the AAC encodings
    aac_encodings = []

    for seq in sequences:
        # Remove potential '-' characters for alignment gaps
        sequence = re.sub('-', '', seq)

        # Ensure each sequence is exactly max_len characters
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        elif len(sequence) < max_len:
            sequence += 'X' * (max_len - len(sequence))  # Pad with 'X' if not enough length

        # Initialize frequency list for this sequence
        temp_encoding = np.zeros((max_len, len(AA)))

        # Calculate frequency for each position up to max_len
        for idx, char in enumerate(sequence):
            if char in AA:
                aa_index = AA.index(char)
                temp_encoding[idx, aa_index] = 1  # Set 1 for the existing amino acid position
            else:
                temp_encoding[idx, :] = 0  # Set all zeros for unknown amino acid position

        aac_encodings.append(temp_encoding)

    # Convert list of encodings to a numpy array
    all_encodings = np.array(aac_encodings)

    return torch.tensor(all_encodings, dtype=torch.float)


#(batchsize,20)
# def AAC_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY'):
#     # Define the amino acids order if it's not provided
#     AA = order
#
#     # Prepare to collect the AAC encodings
#     aac_encodings = []
#
#     for seq in sequences:
#         sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
#         count = Counter(sequence)
#         total_length = len(sequence)
#
#         # Normalize counts to frequencies
#         freqs = [count[aa] / total_length if aa in count else 0 for aa in AA]
#         aac_encodings.append(freqs)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     aac_encodings = np.array(aac_encodings)
#     return torch.tensor(aac_encodings, dtype=torch.float)

#跟上面的比加入了碰到不认识的字符把值变为0的功能
# def AAC_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY'):
#     # Define the amino acids order if it's not provided
#     AA = order
#
#     # Prepare to collect the AAC encodings
#     aac_encodings = []
#
#     for seq in sequences:
#         sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
#         count = Counter(sequence)
#         total_length = len(sequence)
#
#         # Initialize frequency list with zeros for unknown characters
#         freqs = []
#         for aa in AA:
#             if aa in count:
#                 freqs.append(count[aa] / total_length)
#             else:
#                 freqs.append(0)
#         aac_encodings.append(freqs)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     aac_encodings = np.array(aac_encodings)
#     return torch.tensor(aac_encodings, dtype=torch.float)


#降
def DPC_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY'):
    # Define the amino acids order if it's not provided
    AA = order

    # Prepare to collect the DPC encodings
    dpc_encodings = []

    # Create a dictionary for amino acid indices
    AADict = {aa: i for i, aa in enumerate(AA)}

    for seq in sequences:
        sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] += 1

        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        dpc_encodings.append(tmpCode)

    # Convert list of lists to a PyTorch tensor for model compatibility
    dpc_encodings = np.array(dpc_encodings)
    return torch.tensor(dpc_encodings, dtype=torch.float)

#降
def BINARY_embedding(sequences, max_len=50, order='ARNDCQEGHILKMFPSTWYV'):
    # Define the amino acids order if it's not provided
    AA = order

    # Prepare to collect the BINARY encodings
    binary_encodings = []

    for seq in sequences:
        sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
        sequence = sequence[:max_len]  # Truncate sequences to the required length
        sequence = sequence.ljust(max_len, '-')  # Pad sequences to the required length
        encoding = []

        for aa in sequence:
            if aa == '-':
                encoding.extend([0] * 20)
            else:
                encoding.extend([1 if aa == aa1 else 0 for aa1 in AA])

        binary_encodings.append(encoding)

    # Convert list of lists to a PyTorch tensor for model compatibility
    binary_encodings = np.array(binary_encodings)
    return torch.tensor(binary_encodings, dtype=torch.float)


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair

#没效果,(batchsize,75)
# def CKSAAGP_embedding(sequences, gap=2):
#     if gap < 0:
#         raise ValueError("Error: the gap should be equal or greater than zero")
#
#     group = {
#         'alphaticr': 'GAVLMI',
#         'aromatic': 'FYW',
#         'postivecharger': 'KRH',
#         'negativecharger': 'DE',
#         'uncharger': 'STCPNQ'
#     }
#
#     AA = 'ARNDCQEGHILKMFPSTWYV'
#     groupKey = group.keys()
#
#     index = {}
#     for key in groupKey:
#         for aa in group[key]:
#             index[aa] = key
#
#     gPairIndex = []
#     for key1 in groupKey:
#         for key2 in groupKey:
#             gPairIndex.append(key1 + '.' + key2)
#
#     # Prepare to collect the CKSAAGP encodings
#     encodings = []
#
#     for seq in sequences:
#         name, sequence = seq[0], re.sub('-', '', seq[1])
#         code = []
#         for g in range(gap + 1):
#             gPair = generateGroupPairs(groupKey)
#             sum_pairs = 0
#             for p1 in range(len(sequence)):
#                 p2 = p1 + g + 1
#                 if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
#                     gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] += 1
#                     sum_pairs += 1
#
#             if sum_pairs == 0:
#                 code.extend([0] * len(gPairIndex))
#             else:
#                 for gp in gPairIndex:
#                     code.append(gPair[gp] / sum_pairs)
#
#         encodings.append(code)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     encodings = np.array(encodings)
#     return torch.tensor(encodings, dtype=torch.float)


import numpy as np
import torch
import re
#没效果,(batchsize,1200)
# def CKSAAGP_embedding(sequences, gap=2, order='ACDEFGHIKLMNPQRSTVWY'):
# # def CKSAAP_embedding(sequences, gap=2, order='ACDEFGHIKLMNPQRSTVWY'):#真名
#     if gap < 0:
#         raise ValueError("Error: the gap should be equal or greater than zero")
#
#     # Prepare to collect the CKSAAP encodings
#     encodings = []
#     aaPairs = [aa1 + aa2 for aa1 in order for aa2 in order]
#
#     for seq in sequences:
#         name, sequence = seq[0], re.sub('-', '', seq[1])
#         code = []
#         for g in range(gap + 1):
#             myDict = {pair: 0 for pair in aaPairs}
#             sum_pairs = 0
#             for index1 in range(len(sequence)):
#                 index2 = index1 + g + 1
#                 if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in order and sequence[
#                     index2] in order:
#                     myDict[sequence[index1] + sequence[index2]] += 1
#                     sum_pairs += 1
#
#             if sum_pairs == 0:
#                 code.extend([0] * len(aaPairs))
#             else:
#                 for pair in aaPairs:
#                     code.append(myDict[pair] / sum_pairs)
#
#         encodings.append(code)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     encodings = np.array(encodings)
#     return torch.tensor(encodings, dtype=torch.float)
#


import numpy as np
import torch
import re

#(batchsize, 39)
#降， def Count(seq1, seq2):
#     sum = 0
#     for aa in seq1:
#         sum = sum + seq2.count(aa)
#     return sum

# def CKSAAGP_embedding(sequences, gap=2, order='ACDEFGHIKLMNPQRSTVWY'):
# #def CTDC_embedding(sequences):#原名
#     group1 = {
#         'hydrophobicity_PRAM900101': 'RKEDQN',
#         'hydrophobicity_ARGP820101': 'QSTNGDE',
#         'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
#         'hydrophobicity_PONP930101': 'KPDESNQT',
#         'hydrophobicity_CASG920101': 'KDEQPSRNTG',
#         'hydrophobicity_ENGD860101': 'RDKENQHYP',
#         'hydrophobicity_FASG890101': 'KERSQD',
#         'normwaalsvolume': 'GASTPDC',
#         'polarity':        'LIFWCMVY',
#         'polarizability':  'GASDT',
#         'charge':          'KR',
#         'secondarystruct': 'EALMQKRH',
#         'solventaccess':   'ALFCGIVW'
#     }
#     group2 = {
#         'hydrophobicity_PRAM900101': 'GASTPHY',
#         'hydrophobicity_ARGP820101': 'RAHCKMV',
#         'hydrophobicity_ZIMJ680101': 'HMCKV',
#         'hydrophobicity_PONP930101': 'GRHA',
#         'hydrophobicity_CASG920101': 'AHYMLV',
#         'hydrophobicity_ENGD860101': 'SGTAW',
#         'hydrophobicity_FASG890101': 'NTPG',
#         'normwaalsvolume': 'NVEQIL',
#         'polarity':        'PATGS',
#         'polarizability':  'CPNVEQIL',
#         'charge':          'ANCQGHILMFPSTWYV',
#         'secondarystruct': 'VIYCWFT',
#         'solventaccess':   'RKQEND'
#     }
#     group3 = {
#         'hydrophobicity_PRAM900101': 'CLVIMFW',
#         'hydrophobicity_ARGP820101': 'LYPFIW',
#         'hydrophobicity_ZIMJ680101': 'LPFYI',
#         'hydrophobicity_PONP930101': 'YMFWLCVI',
#         'hydrophobicity_CASG920101': 'FIWC',
#         'hydrophobicity_ENGD860101': 'CVLIMF',
#         'hydrophobicity_FASG890101': 'AYHWVMFLIC',
#         'normwaalsvolume': 'MHKFRYW',
#         'polarity':        'HQRKNED',
#         'polarizability':  'KMHFRYW',
#         'charge':          'DE',
#         'secondarystruct': 'GNPSD',
#         'solventaccess':   'MSPTHY'
#     }
#
#     groups = [group1, group2, group3]
#     properties = (
#         'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
#         'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
#         'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess'
#     )
#
#     # Prepare to collect the CTDC encodings
#     encodings = []
#
#     for seq in sequences:
#         name, sequence = seq[0], re.sub('-', '', seq[1])
#         code = []
#         for p in properties:
#             c1 = Count(group1[p], sequence) / len(sequence)
#             c2 = Count(group2[p], sequence) / len(sequence)
#             c3 = 1 - c1 - c2
#             code.extend([c1, c2, c3])
#         encodings.append(code)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     encodings = np.array(encodings)
#     return torch.tensor(encodings, dtype=torch.float)

# import numpy as np
# import torch
# import re
# import math
# #(batchsize, 195)
# def Count(aaSet, sequence):
#     number = 0
#     for aa in sequence:
#         if aa in aaSet:
#             number += 1
#     cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
#     cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]
#
#     code = []
#     for cutoff in cutoffNums:
#         myCount = 0
#         for i in range(len(sequence)):
#             if sequence[i] in aaSet:
#                 myCount += 1
#                 if myCount == cutoff:
#                     code.append((i + 1) / len(sequence) * 100)
#                     break
#         if myCount == 0:
#             code.append(0)
#     return code
#
# def CKSAAGP_embedding(sequences, gap=2, order='ACDEFGHIKLMNPQRSTVWY'):
# #def CTDD_embedding(sequences):#原名
#     group1 = {
#         'hydrophobicity_PRAM900101': 'RKEDQN',
#         'hydrophobicity_ARGP820101': 'QSTNGDE',
#         'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
#         'hydrophobicity_PONP930101': 'KPDESNQT',
#         'hydrophobicity_CASG920101': 'KDEQPSRNTG',
#         'hydrophobicity_ENGD860101': 'RDKENQHYP',
#         'hydrophobicity_FASG890101': 'KERSQD',
#         'normwaalsvolume': 'GASTPDC',
#         'polarity':        'LIFWCMVY',
#         'polarizability':  'GASDT',
#         'charge':          'KR',
#         'secondarystruct': 'EALMQKRH',
#         'solventaccess':   'ALFCGIVW'
#     }
#     group2 = {
#         'hydrophobicity_PRAM900101': 'GASTPHY',
#         'hydrophobicity_ARGP820101': 'RAHCKMV',
#         'hydrophobicity_ZIMJ680101': 'HMCKV',
#         'hydrophobicity_PONP930101': 'GRHA',
#         'hydrophobicity_CASG920101': 'AHYMLV',
#         'hydrophobicity_ENGD860101': 'SGTAW',
#         'hydrophobicity_FASG890101': 'NTPG',
#         'normwaalsvolume': 'NVEQIL',
#         'polarity':        'PATGS',
#         'polarizability':  'CPNVEQIL',
#         'charge':          'ANCQGHILMFPSTWYV',
#         'secondarystruct': 'VIYCWFT',
#         'solventaccess':   'RKQEND'
#     }
#     group3 = {
#         'hydrophobicity_PRAM900101': 'CLVIMFW',
#         'hydrophobicity_ARGP820101': 'LYPFIW',
#         'hydrophobicity_ZIMJ680101': 'LPFYI',
#         'hydrophobicity_PONP930101': 'YMFWLCVI',
#         'hydrophobicity_CASG920101': 'FIWC',
#         'hydrophobicity_ENGD860101': 'CVLIMF',
#         'hydrophobicity_FASG890101': 'AYHWVMFLIC',
#         'normwaalsvolume': 'MHKFRYW',
#         'polarity':        'HQRKNED',
#         'polarizability':  'KMHFRYW',
#         'charge':          'DE',
#         'secondarystruct': 'GNPSD',
#         'solventaccess':   'MSPTHY'
#     }
#
#     groups = [group1, group2, group3]
#     properties = (
#         'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
#         'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
#         'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess'
#     )
#
#     # Prepare to collect the CTDD encodings
#     encodings = []
#
#     for seq in sequences:
#         name, sequence = seq[0], re.sub('-', '', seq[1])
#         code = []
#         for p in properties:
#             code.extend(Count(groups[0][p], sequence))
#             code.extend(Count(groups[1][p], sequence))
#             code.extend(Count(groups[2][p], sequence))
#         encodings.append(code)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     encodings = np.array(encodings)
#     return torch.tensor(encodings, dtype=torch.float)



# import numpy as np
# import torch
# import re
# import math
# #不变，(batchsize,400)
# def CKSAAGP_embedding(sequences, gap=2, order='ACDEFGHIKLMNPQRSTVWY'):
# #def DDE_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY'):#原名
#     AA = order
#
#     myCodons = {
#         'A': 4,
#         'C': 2,
#         'D': 2,
#         'E': 2,
#         'F': 2,
#         'G': 4,
#         'H': 2,
#         'I': 3,
#         'K': 2,
#         'L': 6,
#         'M': 1,
#         'N': 2,
#         'P': 4,
#         'Q': 2,
#         'R': 6,
#         'S': 6,
#         'T': 4,
#         'V': 4,
#         'W': 1,
#         'Y': 2
#     }
#
#     diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
#
#     myTM = [(myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61) for pair in diPeptides]
#
#     AADict = {AA[i]: i for i in range(len(AA))}
#
#     encodings = []
#
#     for seq in sequences:
#         name, sequence = seq[0], re.sub('-', '', seq[1])
#         if len(sequence) <= 1:
#             # 如果序列长度小于等于 1，用全 0 向量表示编码
#             code = [0] * 400
#         else:
#             code = []
#             tmpCode = [0] * 400
#             for j in range(len(sequence) - 1):
#                 tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] += 1
#             if sum(tmpCode) != 0:
#                 tmpCode = [i / sum(tmpCode) for i in tmpCode]
#
#             myTV = [myTM[j] * (1 - myTM[j]) / (len(sequence) - 1) for j in range(len(myTM))]
#
#             for j in range(len(tmpCode)):
#                 tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j]) if myTV[j] != 0 else 0
#
#             code.extend(tmpCode)
#         encodings.append(code)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     encodings = np.array(encodings)
#     return torch.tensor(encodings, dtype=torch.float)


# import numpy as np
# import torch
# import re
# from collections import Counter
# #没用，(btachsize,920)
#
# def CKSAAGP_embedding(sequences, order='ACDEFGHIKLMNPQRSTVWY'):
# #def EAAC_embedding(sequences, window=5, order='ACDEFGHIKLMNPQRSTVWY'):#原名
#     window = 5
#
#
#     # Define the amino acids order if it's not provided
#     AA = order
#
#     # Prepare to collect the EAAC encodings
#     eaac_encodings = []
#
#     for index, seq in enumerate(sequences):
#         sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
#         if len(sequence) < window:
#             print(
#                 f"Error: sequence at line {index + 1} has length less than the sliding window size: {len(sequence)} < {window}")
#             continue  # Skip this sequence
#
#         # Slide the window across the sequence
#         for j in range(len(sequence) - window + 1):
#             window_seq = sequence[j:j + window]
#             count = Counter(window_seq)
#             total_length = len(window_seq)
#
#             # Normalize counts to frequencies
#             freqs = [count[aa] / total_length if aa in count else 0 for aa in AA]
#             eaac_encodings.append(freqs)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     eaac_encodings = np.array(eaac_encodings)
#     return torch.tensor(eaac_encodings, dtype=torch.float)


# import numpy as np
# import torch
# import re
# from collections import Counter
# #降
# def CKSAAGP_embedding(sequences):
# #def EGAAC_embedding(sequences, window=4):#原名
#     # Define the amino acid groups
#     window = 4
#
#
#     group = {
#         'alphaticr': 'GAVLMI',
#         'aromatic': 'FYW',
#         'postivecharger': 'KRH',
#         'negativecharger': 'DE',
#         'uncharger': 'STCPNQ'
#     }
#
#     groupKey = group.keys()
#
#     # Prepare to collect the EGAAC encodings
#     egaac_encodings = []
#
#     for index, seq in enumerate(sequences):
#         sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
#         if len(sequence) < window:
#             print(
#                 f"Error: sequence at line {index + 1} has length less than the sliding window size: {len(sequence)} < {window}")
#             continue  # Skip this sequence
#
#         # Slide the window across the sequence
#         for j in range(len(sequence) - window + 1):
#             window_seq = sequence[j:j + window]
#             count = Counter(window_seq)
#             myDict = {}
#             for key in groupKey:
#                 myDict[key] = sum(count[aa] for aa in group[key])
#             freqs = [myDict[key] / window for key in groupKey]
#             egaac_encodings.append(freqs)
#
#     # Ensure the output shape matches the model's input expectation
#     # Here we need to ensure the shape is (200, 256)
#     # Flatten the list to match the expected dimensions
#     egaac_encodings = np.array(egaac_encodings)
#     egaac_encodings = egaac_encodings.flatten()[:200 * 256]  # Ensure the length is 200*256
#     egaac_encodings = egaac_encodings.reshape(200, 256)  # Reshape to (200, 256)
#
#     return torch.tensor(egaac_encodings, dtype=torch.float)


# import numpy as np
# import torch
# import re
# from collections import Counter
#
# #没效果
# def CKSAAGP_embedding(sequences):
# #def GAAC_embedding(sequences):#原名
#     # Define the amino acid groups
#     group = {
#         'alphatic': 'GAVLMI',
#         'aromatic': 'FYW',
#         'postivecharge': 'KRH',
#         'negativecharge': 'DE',
#         'uncharge': 'STCPNQ'
#     }
#
#     groupKey = group.keys()
#
#     # Prepare to collect the GAAC encodings
#     gaac_encodings = []
#
#     for index, seq in enumerate(sequences):
#         sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
#
#         if len(sequence) == 0:
#             print(f"Error: sequence at line {index + 1} is empty after removing gaps.")
#             continue  # Skip this sequence
#
#         count = Counter(sequence)
#         myDict = {}
#         for key in groupKey:
#             myDict[key] = sum(count[aa] for aa in group[key])
#         freqs = [myDict[key] / len(sequence) for key in groupKey]
#         gaac_encodings.append(freqs)
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     gaac_encodings = np.array(gaac_encodings)
#     return torch.tensor(gaac_encodings, dtype=torch.float)
#
#
# import numpy as np
# import torch
# import re
# from collections import Counter
# #没用，(batchsie,125)
# def CKSAAGP_embedding(sequences):
# #def GTPC_embedding(sequences):#原名
#     # Define the amino acid groups
#     group = {
#         'alphaticr': 'GAVLMI',
#         'aromatic': 'FYW',
#         'postivecharger': 'KRH',
#         'negativecharger': 'DE',
#         'uncharger': 'STCPNQ'
#     }
#
#     groupKey = list(group.keys())
#     baseNum = len(groupKey)
#     triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]
#
#     index = {}
#     for key in groupKey:
#         for aa in group[key]:
#             index[aa] = key
#
#     # Prepare to collect the GTPC encodings
#     gtpc_encodings = []
#
#     for seq in sequences:
#         sequence = re.sub('-', '', seq)  # Remove potential '-' characters for alignment gaps
#
#         if len(sequence) < 3:
#             print(f"Error: sequence {seq} is too short to form triplets.")
#             continue  # Skip this sequence
#
#         myDict = {t: 0 for t in triple}
#         sum = 0
#
#         for j in range(len(sequence) - 2):
#             if sequence[j] in index and sequence[j+1] in index and sequence[j+2] in index:
#                 triplet_key = index[sequence[j]] + '.' + index[sequence[j+1]] + '.' + index[sequence[j+2]]
#                 myDict[triplet_key] += 1
#                 sum += 1
#
#         if sum == 0:
#             gtpc_encodings.append([0] * len(triple))
#         else:
#             gtpc_encodings.append([myDict[t] / sum for t in triple])
#
#     # Convert list of lists to a PyTorch tensor for model compatibility
#     gtpc_encodings = np.array(gtpc_encodings)
#     return torch.tensor(gtpc_encodings, dtype=torch.float)


