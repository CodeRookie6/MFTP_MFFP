import datetime
import os
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from loss_functions import FocalDiceLoss
from train import DataTrain, evaluate, CosineScheduler
from data_feature import AAI_embedding, PAAC_embedding, PC6_embedding, BLOSUM62_embedding, AAC_embedding
from model import MMEN_MTPP
import torch_geometric
from torch_geometric.data import Data as GeometricData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']



def PadEncode(data, label, max_len):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            temp.append(elemt)
            seq_length.append(len(temp[b]))
            b += 1
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    print(f"PadEncode: Filtered sequences: {len(data_e)}, Filtered labels: {len(label_e)}")
    return np.array(data_e), np.array(label_e), np.array(seq_length)




# 会打印读取到的序列和标签总数
def getSequenceData(first_dir, file_name):
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path, 'r') as f:
        for each in f:
            each = each.strip()
            if each:  # Check if line is not empty
                if each[0] == '>':  # This assumes that labels are prefixed with '>'
                    try:
                        numeric_label = np.array(list(map(int, each[1:])),
                                                 dtype=int)  # Convert string labels to numeric vectors
                        label.append(numeric_label)
                    except ValueError:
                        print("Invalid label encountered and skipped:", each[1:])
                else:
                    data.append(each)

    print("Total sequences read:", len(data))
    print("Total labels read:", len(label))

    # if len(label) > 0:
    #     print("Sample labels (first 5):", label[:5])
    #     label_array = np.vstack(label)
    #     print("Label distribution per class:")
    #     for i in range(label_array.shape[1]):
    #         print(f"Class {i}: {np.sum(label_array[:, i])} positives, {len(label_array) - np.sum(label_array[:, i])} negatives")

    return data, label

def staticTrainAndTest(y_train, y_test):
    data_size_tr = np.zeros(len(filenames))
    data_size_te = np.zeros(len(filenames))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
                 data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    return data_size_tr

class SequenceGraphDataset(Dataset):
    def __init__(self, sequences, labels, seq_lengths, max_len, features):
        assert len(sequences) == len(labels) == len(
            seq_lengths), "Sequences, labels, and seq_lengths must have the same length."
        self.sequences = sequences
        self.labels = labels
        self.seq_lengths = seq_lengths
        self.max_len = max_len
        self.features = features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        seq_length = self.seq_lengths[idx]

        seq_tensor = torch.LongTensor(seq)
        label_tensor = torch.FloatTensor(label)
        seq_length_tensor = torch.tensor(seq_length, dtype=torch.long)  # 转换为张量

        # 获取特征
        aai = torch.FloatTensor(self.features['aai'][idx])
        paac = torch.FloatTensor(self.features['paac'][idx])
        pc6 = torch.FloatTensor(self.features['pc6'][idx])
        blosum62 = torch.FloatTensor(self.features['blosum62'][idx])
        aac = torch.FloatTensor(self.features['aac'][idx])

        # 构建邻接矩阵
        edge_index = []
        for i in range(self.max_len - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]

        # 标签作为全局特征
        label_global = torch.FloatTensor(label)

        # 构建图数据
        data = GeometricData(x=torch.randn(self.max_len, 256),  # 假设 GNN 输入特征维度为 256
                             edge_index=edge_index,
                             y=label_tensor,
                             label_global=label_global)

        return seq_tensor, label_tensor, seq_length_tensor, aai, paac, pc6, blosum62, aac, data

def collate_fn(batch):
    seq_tensors, label_tensors, seq_lengths, aai, paac, pc6, blosum62, aac, graphs = zip(*batch)
    seq_tensors = torch.stack(seq_tensors)
    label_tensors = torch.stack(label_tensors)
    seq_lengths = torch.stack(seq_lengths)
    aai = torch.stack(aai)
    paac = torch.stack(paac)
    pc6 = torch.stack(pc6)
    blosum62 = torch.stack(blosum62)
    aac = torch.stack(aac)
    graphs = torch_geometric.data.Batch.from_data_list(graphs)
    return seq_tensors, label_tensors, seq_lengths, aai, paac, pc6, blosum62, aac, graphs.edge_index, graphs.x, graphs.batch

def main(num, data):
    first_dir = 'dataset'
    max_length = 50

    # 获取训练和测试数据
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

    # 将标签转换为numpy数组
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)

    # 使用 PadEncode 进行序列和标签的处理
    x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
    x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)

    # 打印训练和测试集的大小（调试用）
    print(f"After PadEncode - Training set size: {x_train.shape[0]}, Training labels size: {y_train.shape[0]}")
    print(f"After PadEncode - Test set size: {x_test.shape[0]}, Test labels size: {y_test.shape[0]}")

    # 确保训练集和测试集中的序列和标签数量相同
    assert x_train.shape[0] == y_train.shape[0], "Training data and labels size mismatch!"
    assert x_test.shape[0] == y_test.shape[0], "Test data and labels size mismatch!"

    # 提取特征
    train_features_aai = AAI_embedding(train_sequence_data, max_len=max_length)
    test_features_aai = AAI_embedding(test_sequence_data, max_len=max_length)
    train_features_paac = PAAC_embedding(train_sequence_data, max_len=max_length)
    test_features_paac = PAAC_embedding(test_sequence_data, max_len=max_length)
    train_features_pc6 = PC6_embedding(train_sequence_data, max_len=max_length)
    test_features_pc6 = PC6_embedding(test_sequence_data, max_len=max_length)
    train_features_blosum62 = BLOSUM62_embedding(train_sequence_data, max_len=max_length)
    test_features_blosum62 = BLOSUM62_embedding(test_sequence_data, max_len=max_length)
    train_features_aac = AAC_embedding(train_sequence_data)
    test_features_aac = AAC_embedding(test_sequence_data)


    train_features = {
        'aai': train_features_aai,
        'paac': train_features_paac,
        'pc6': train_features_pc6,
        'blosum62': train_features_blosum62,
        'aac': train_features_aac
    }
    test_features = {
        'aai': test_features_aai,
        'paac': test_features_paac,
        'pc6': test_features_pc6,
        'blosum62': test_features_blosum62,
        'aac': test_features_aac
    }

    # 创建数据集，传递 seq_length
    train_dataset = SequenceGraphDataset(x_train, y_train, train_length, max_length, train_features)
    test_dataset = SequenceGraphDataset(x_test, y_test, test_length, max_length, test_features)

    # 打印数据集的长度（调试用）
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

    # 定义 DataLoader
    dataset_train = DataLoader(
        train_dataset,
        batch_size=data['batch_size'],
        shuffle=False,  # 暂时关闭 shuffle 以便调试
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False  # 确保每个批次有完整的数据
    )
    dataset_test = DataLoader(
        test_dataset,
        batch_size=data['batch_size'],
        shuffle=False,  # 关闭 shuffle
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    # 打印 DataLoader 生成的批次数量（调试用）
    print(f"Number of batches in train DataLoader: {len(dataset_train)}")
    print(f"Number of batches in test DataLoader: {len(dataset_test)}")

    # 初始化模型
    vocab_size = 50
    output_size = 21

    model = MFFTPC(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'],
                data['num_heads'])
    model.to(DEVICE)
    rate_learning = data['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
    criterion = MarginalFocalDiceLoss()
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = BCEFocalLoss(gamma=10)

    #criterion = LDAM_loss(max_m=0.5, class_weight="balanced")
    #criterion = FocalDiceLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight'])
    #criterion = AsymmetricLoss()

    # 开始训练
    Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

    a = time.time()
    Train.train_step(dataset_train, epochs=data['epochs'], plot_picture=True)
    b = time.time()
    test_score = evaluate(model, dataset_test, device=DEVICE)
    runtime = b - a

    # 保存模型
    PATH = os.getcwd()
    print(PATH)
    each_model = os.path.join(PATH, 'result', 'Model',  f'model{num+1}.h5')

    torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)

    # 打印结果
    print(f"runtime:{runtime:.3f}s")
    print("测试集：")
    print(f'aiming: {test_score["aiming"]:.3f}')
    print(f'coverage: {test_score["coverage"]:.3f}')
    print(f'accuracy: {test_score["accuracy"]:.3f}')
    print(f'absolute_true: {test_score["absolute_true"]:.3f}')
    print(f'absolute_false: {test_score["absolute_false"]:.3f}')

    # 保存结果到 CSV
    title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
    model_name = f'model{num+1}'
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = [[model_name, '%.3f' % test_score["aiming"],
                '%.3f' % test_score["coverage"],
                '%.3f' % test_score["accuracy"],
                '%.3f' % test_score["absolute_true"],
                '%.3f' % test_score["absolute_false"],
                '%.3f' % runtime,
                now]]

    path = os.path.join('result', 'model_result.csv')

    if os.path.exists(path):
        data1 = pd.read_csv(path, header=None)
        one_line = list(data1.iloc[0])
        if one_line == title:
            with open(path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
        else:
            with open(path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)
                writer.writerows(content)
    else:
        with open(path, 'w', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)

if __name__ == '__main__':


    clip_pos = 0.7
    clip_neg = 0.5
    pos_weight = 0.7

    batch_size = 256
    epochs = 256

    learning_rate = 0.0018

    embedding_size = 256

    dropout = 0.6
    fan_epochs = 1
    num_heads = 8

    para = {
        'clip_pos': clip_pos,
        'clip_neg': clip_neg,
        'pos_weight': pos_weight,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'embedding_size': embedding_size,
        'dropout': dropout,
        'fan_epochs': fan_epochs,
        'num_heads': num_heads
    }
    for i in range(100):
        print(f"Starting training iteration {i + 1}/100")
        main(i, para)










