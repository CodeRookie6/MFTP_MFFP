#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/28 17:13
# @Author  : fhh
# @File    : train.py
# @Software: PyCharm
import time
import torch
import math
import numpy as np
from sklearn import metrics
from torch import nn
#from torchinfo import summary
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
import evaluation
import matplotlib.pyplot as plt
import os
from data_feature import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device('cpu')

# def evaluate(model, datadl, device="cpu"):
#     # 将模型移动到指定设备上
#     model.to(device)
#     # 设置模型为评估模式
#     model.eval()
#     # 存储预测结果
#     predictions = []
#     # 存储真实标签
#     labels = []
#     with torch.no_grad():
#         for x, y, z in datadl:
#             # 将输入数据和标签移动到指定设备上，并转换为长整型
#             x = x.to(device).long()
#             y = y.to(device).long()
#             # 获取模型的预测结果，并将其转换为列表形式
#             predictions.extend(model(x, z).tolist())
#             # 将真实标签转换为列表形式
#             labels.extend(y.tolist())
#
#     # 使用 evaluation.evaluate 函数对预测结果进行评估，返回评估分数
#     scores = evaluation.evaluate(np.array(predictions), np.array(labels))
#     return scores
#
#
#
#
#
# def scoring(y_true, y_score):
#     #设置分类的阈值为0.5，用于将连续的预测得分转换为二分类预测结果。
#     threshold = 0.5
#     #根据设定的阈值，将预测得分 y_score 转换为二分类预测结果 y_pred，其中大于等于阈值的预测得分被视为正例，小于阈值的预测得分被视为负例。
#     y_pred = [int(i >= threshold) for i in y_score]
#     #利用真实标签 y_true 和预测标签 y_pred 计算混淆矩阵，其中包括真正例（TP）、真负例（TN）、假正例（FP）和假负例（FN）。
#     confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
#     #将混淆矩阵展平，并将各项元素分配给相应的变量，方便后续指标的计算。
#     tn, fp, fn, tp = confusion_matrix.flatten()
#     #计算灵敏度（召回率），即真正例率，表示所有真实正例中被正确预测为正例的比例。
#     sen = tp / (fn + tp)
#     # 计算特异度，表示所有真实负例中被正确预测为负例的比例。
#     spe = tn / (fp + tn)
#     #计算精确度，表示所有被预测为正例中真实正例的比例。
#     pre = metrics.precision_score(y_true, y_pred)
#     #计算ROC曲线下的面积（AUC），用于评估模型的分类能力，值越接近1表示模型性能越好。
#     auc = metrics.roc_auc_score(y_true, y_score)
#     #根据真实标签和预测得分计算精确度-召回率曲线（PR曲线）的精确度、召回率和阈值。
#     pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
#     # 计算PR曲线下的面积（AUPR），用于评估模型在正例类别上的性能，值越接近1表示模型性能越好。\
#     aupr = metrics.auc(rc, pr)
#     f1 = metrics.f1_score(y_true, y_pred)
#     mcc = metrics.matthews_corrcoef(y_true, y_pred)
#     acc = metrics.accuracy_score(y_true, y_pred)
#     return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)
#
# #一个名为 DataTrain 的类，用于训练模型
# class DataTrain:
#     #类的初始化方法，用于初始化训练所需的模型、优化器、损失函数、学习率调度器以及设备类型等。
#     def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.lr_scheduler = scheduler
#
#         self.device = device
#         self.model.to(self.device)
#
#     #定义了一个训练步骤的方法 train_step，它接受训练数据迭代器 train_iter，还有训练的轮数 epochs 和是否绘制图片的标志 plot_picture。
#     def train_step(self, train_iter, epochs=None, plot_picture=False):
#         x_plot = []
#         y_plot = []
#         epochTrainLoss = []
#         # for train_data, train_label in train_iter:
#         #     train_data, train_label = train_data.to(self.device), train_label.to(self.device)
#         #     summary(self.model, train_data.shape, dtypes=['torch.IntTensor'])
#         #     break
#         #这部分是训练循环，对每个 epoch 和每个 batch 进行训练。在每个 batch 中，将数据和标签移动到指定的设备上，通过模型计算预测值，
#         # 计算损失并进行反向传播和参数更新。同时，如果有学习率调度器，则根据步数或者 epoch 来调整学习率，并记录相关数据以便后续绘图。
#         steps = 1
#         for epoch in range(1, epochs+1):
#             # metric = Accumulator(2)
#             start = time.time()
#             total_loss = 0
#             for train_data, train_label, train_length in train_iter:
#                 self.model.train()  # 进入训练模式
#                 train_data, train_label, train_length = train_data.to(self.device), train_label.to(self.device), train_length.to(self.device)
#                 y_hat = self.model(train_data.long(), train_length)
#                 loss = self.criterion(y_hat, train_label.float())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if self.lr_scheduler:
#                     if self.lr_scheduler.__module__ == lr_scheduler.__name__:
#                         # Using PyTorch In-Built scheduler
#                         self.lr_scheduler.step()
#                     else:
#                         # Using custom defined scheduler
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = self.lr_scheduler(steps)
#
#                 x_plot.append(epoch)
#                 y_plot.append(self.lr_scheduler(epoch))
#                 total_loss += loss.item()
#                 steps += 1
#             # train_loss = metric[0] / metric[1]
#             # x_plot.append(self.lr_scheduler.last_epoch)
#             # y_plot.append(self.lr_scheduler.get_lr()[0])
#             # x_plot.append(epoch)
#             # y_plot.append(self.lr_scheduler(epoch))
#             # epochTrainLoss.append(train_loss)
#             finish = time.time()
#
#             print(f'[ Epoch {epoch} ', end='')
#             print("运行时间{}s".format(finish - start))
#             print(f'loss={total_loss / len(train_iter)} ]')#计算平均损失（average loss）
#
#         #如果需要绘制图片，则绘制学习率变化曲线，并保存为图片文件
#         if plot_picture:
#             # 绘制学习率变化曲线
#             plt.plot(x_plot, y_plot, 'r')
#             plt.title('lr value of LambdaLR with (Cos_warmup) ')
#             plt.xlabel('step')
#             plt.ylabel('lr')
#             plt.savefig('./result/Cos_warmup.jpg')
#             # plt.show()
#
#             # # 绘制损失函数曲线
#             # plt.figure()
#             # plt.plot(x_plot, epochTrainLoss)
#             # plt.legend(['trainLoss'])
#             # plt.xlabel('epochs')
#             # plt.ylabel('SHLoss')
#             # plt.savefig('./image/Cos_warmup_Loss(200).jpg')
#             # # plt.show()
#
#     #这部分是另一个训练方法 KD_step，与前面的 train_step 类似，不同之处在于没有使用额外的长度信息，即没有使用 train_length
#     def KD_step(self, train_iter, epochs=None, plot_picture=False):
#         steps = 1
#         for epoch in range(1, epochs+1):
#             # metric = Accumulator(2)
#             start = time.time()
#             total_loss = 0
#             for train_data, train_label in train_iter:
#                 self.model.train()  # 进入训练模式
#
#                 train_data, train_label = train_data.to(self.device), train_label.to(self.device)
#                 y_hat = self.model(train_data.long())
#                 loss = self.criterion(y_hat, train_label.float())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if self.lr_scheduler:
#                     if self.lr_scheduler.__module__ == lr_scheduler.__name__:
#                         # Using PyTorch In-Built scheduler
#                         self.lr_scheduler.step()
#                     else:
#                         # Using custom defined scheduler
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = self.lr_scheduler(steps)
#
#                 # x_plot.append(epoch)
#                 # y_plot.append(self.lr_scheduler(epoch))
#                 total_loss += loss.item()
#                 steps += 1
#             finish = time.time()
#
#             print(f'[ Epoch {epoch} ', end='')
#             print("运行时间{}s".format(finish - start))
#             print(f'loss={total_loss / len(train_iter)} ]')
#
#
#
#
# def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )
#     return LambdaLR(optimizer_, lr_lambda, last_epoch)
#
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                            (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr

"""四个特征提取用这个"""
# import time
# import torch
# import math
# import numpy as np
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import shap
#
# def evaluate(model, datadl, device="cpu"):
#     model.to(device)
#     model.eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for x, y, z, aai, paac, pc6, blosum62,aac in datadl:
#             x = x.to(device).long()
#             y = y.to(device).long()
#             features = {'aai': aai.to(device), 'paac': paac.to(device), 'pc6': pc6.to(device), 'blosum62': blosum62.to(device),'aac':aac.to(device)}
#             predictions.extend(model(x, z, features).tolist())
#             labels.extend(y.tolist())
#
#     scores = evaluation.evaluate(np.array(predictions), np.array(labels))
#     return scores
#
#
#
#
#
# def scoring(y_true, y_score):
#     threshold = 0.5
#     y_pred = [int(i >= threshold) for i in y_score]
#     confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = confusion_matrix.flatten()
#     sen = tp / (fn + tp)
#     spe = tn / (fp + tn)
#     pre = metrics.precision_score(y_true, y_pred)
#     auc = metrics.roc_auc_score(y_true, y_score)
#     pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
#     aupr = metrics.auc(rc, pr)
#     f1 = metrics.f1_score(y_true, y_pred)
#     mcc = metrics.matthews_corrcoef(y_true, y_pred)
#     acc = metrics.accuracy_score(y_true, y_pred)
#     return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)
#
# class DataTrain:
#     def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.lr_scheduler = scheduler
#
#         self.device = device
#         self.model.to(self.device)
#
#     def train_step(self, train_iter, epochs=None, plot_picture=True):
#         x_plot = []
#         y_plot = []
#         epochTrainLoss = []
#         steps = 1
#         for epoch in range(1, epochs+1):
#             start = time.time()
#             total_loss = 0
#             for train_data, train_label, train_length, aai, paac, pc6, blosum62,aac in train_iter:
#
#
#                 self.model.train()  # 进入训练模式
#                 train_data, train_label, train_length = train_data.to(self.device), train_label.to(self.device), train_length.to(self.device)
#                 features = {'aai': aai.to(self.device), 'paac': paac.to(self.device), 'pc6': pc6.to(self.device), 'blosum62': blosum62.to(self.device),'aac':aac.to(self.device)}
#                 y_hat = self.model(train_data, train_length, features)
#                 loss = self.criterion(y_hat, train_label.float())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#
#
#                 if self.lr_scheduler:
#                     if self.lr_scheduler.__module__ == lr_scheduler.__name__:
#                         self.lr_scheduler.step()
#                     else:
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = self.lr_scheduler(steps)
#
#                 x_plot.append(epoch)
#                 y_plot.append(self.lr_scheduler(epoch))
#                 total_loss += loss.item()
#                 steps += 1
#
#             finish = time.time()
#
#             print(f'[ Epoch {epoch} ', end='')
#             print("运行时间{}s".format(finish - start))
#             print(f'loss={total_loss / len(train_iter)} ]')
#
#         if plot_picture:
#             plt.plot(x_plot, y_plot, 'r')
#             plt.title('lr value of LambdaLR with (Cos_warmup) ')
#             plt.xlabel('step')
#             plt.ylabel('lr')
#             plt.savefig('./result/Cos_warmup.jpg')
#
#
#
# def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )
#     return LambdaLR(optimizer_, lr_lambda, last_epoch)
#
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                            (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr




#PSO训练用
# import time
# import torch
# import math
# import numpy as np
# from sklearn import metrics
# import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import LambdaLR
# import evaluation  # 确保导入 evaluation.py
#
#
# def evaluate(model, data_iter, device="cuda"):
#     model.eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for idx, batch in enumerate(data_iter):
#             if len(batch) == 8:
#                 data, label, valid_len, aai, paac, pc6, blosum62, aac = batch
#             else:
#                 data, label, valid_len = batch
#                 aai = paac = pc6 = blosum62 = aac = None
#
#             data = data.to(device)
#             label = label.to(device)
#             valid_len = valid_len.to(device)
#             features = {}
#             if aai is not None:
#                 features['aai'] = aai.to(device)
#             if paac is not None:
#                 features['paac'] = paac.to(device)
#             if pc6 is not None:
#                 features['pc6'] = pc6.to(device)
#             if blosum62 is not None:
#                 features['blosum62'] = blosum62.to(device)
#             if aac is not None:
#                 features['aac'] = aac.to(device)
#             outputs = model(data, valid_len, features)
#             outputs = torch.sigmoid(outputs)
#             outputs_np = outputs.cpu().numpy()
#             labels_np = label.cpu().numpy()
#             # 检查并调整维度
#             if outputs_np.ndim == 1:
#                 outputs_np = np.expand_dims(outputs_np, axis=0)
#             if labels_np.ndim == 1:
#                 labels_np = np.expand_dims(labels_np, axis=0)
#             print(f"Batch {idx}: outputs_np shape {outputs_np.shape}, labels_np shape {labels_np.shape}")
#             predictions.append(outputs_np)
#             labels.append(labels_np)
#     # 拼接数组
#     predictions = np.concatenate(predictions, axis=0)
#     labels = np.concatenate(labels, axis=0)
#     print("Predictions shape:", predictions.shape)
#     print("Labels shape:", labels.shape)
#     # 调用 evaluation.evaluate
#     scores = evaluation.evaluate(predictions, labels)
#     return scores
#
#
# def evaluate(model, data_iter, device="cuda"):
#     model.eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for idx, batch in enumerate(data_iter):
#             if len(batch) == 9:
#                 data, label, valid_len, aai, paac, pc6, blosum62, aac, additional_feature = batch
#             elif len(batch) == 8:
#                 data, label, valid_len, aai, paac, pc6, blosum62, aac = batch
#                 additional_feature = None
#             elif len(batch) == 6:  # 假设 batch 中有 6 个元素（无 aai 和 paac）
#                 data, label, valid_len, pc6, blosum62, aac = batch
#                 aai = paac = additional_feature = None
#             elif len(batch) == 5:  # 假设 batch 中有 5 个元素（无 aai、paac、和 pc6）
#                 data, label, valid_len, blosum62, aac = batch
#                 aai = paac = pc6 = additional_feature = None
#             elif len(batch) == 3:  # 最简单的情况，只有数据、标签、和有效长度
#                 data, label, valid_len = batch
#                 aai = paac = pc6 = blosum62 = aac = additional_feature = None
#             else:
#                 raise ValueError(f"Unexpected batch length: {len(batch)}")
#
#             data = data.to(device)
#             label = label.to(device)
#             valid_len = valid_len.to(device)
#             features = {}
#             if aai is not None:
#                 features['aai'] = aai.to(device)
#             if paac is not None:
#                 features['paac'] = paac.to(device)
#             if pc6 is not None:
#                 features['pc6'] = pc6.to(device)
#             if blosum62 is not None:
#                 features['blosum62'] = blosum62.to(device)
#             if aac is not None:
#                 features['aac'] = aac.to(device)
#             if additional_feature is not None:
#                 features['additional_feature'] = additional_feature.to(device)  # 如果有额外的特征，添加到features字典
#
#             # 传递到模型中
#             outputs = model(data, valid_len, features)
#             outputs = torch.sigmoid(outputs)
#             outputs_np = outputs.cpu().numpy()
#             labels_np = label.cpu().numpy()
#
#             # 检查并调整维度
#             if outputs_np.ndim == 1:
#                 outputs_np = np.expand_dims(outputs_np, axis=0)
#             if labels_np.ndim == 1:
#                 labels_np = np.expand_dims(labels_np, axis=0)
#
#             #print(f"Batch {idx}: outputs_np shape {outputs_np.shape}, labels_np shape {labels_np.shape}")
#             predictions.append(outputs_np)
#             labels.append(labels_np)
#
#     # 拼接数组
#     predictions = np.concatenate(predictions, axis=0)
#     labels = np.concatenate(labels, axis=0)
#     # print("Predictions shape:", predictions.shape)
#     # print("Labels shape:", labels.shape)
#
#     # 调用 evaluation.evaluate
#     scores = evaluation.evaluate(predictions, labels)
#     return scores
#
#
# class DataTrain:
#     def __init__(self, model, optimizer, criterion, lr_scheduler=None, device="cuda", *args, **kwargs):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.lr_scheduler = lr_scheduler
#
#         self.device = device
#         self.model.to(self.device)
#
#     def train_step(self, train_iter, epochs=None, plot_picture=True):
#         x_plot = []
#         y_plot = []
#         epochTrainLoss = []
#         steps = 1
#         for epoch in range(1, epochs+1):
#             start = time.time()
#             total_loss = 0
#             for batch in train_iter:
#                 if len(batch) == 8:
#                     train_data, train_label, train_length, aai, paac, pc6, blosum62, aac = batch
#                 else:
#                     train_data, train_label, train_length = batch
#                     aai = paac = pc6 = blosum62 = aac = None
#
#                 self.model.train()  # 进入训练模式
#                 train_data, train_label, train_length = train_data.to(self.device), train_label.to(self.device), train_length.to(self.device)
#                 features = {}
#                 if aai is not None:
#                     features['aai'] = aai.to(self.device).float()
#                 if paac is not None:
#                     features['paac'] = paac.to(self.device).float()
#                 if pc6 is not None:
#                     features['pc6'] = pc6.to(self.device).float()
#                 if blosum62 is not None:
#                     features['blosum62'] = blosum62.to(self.device).float()
#                 if aac is not None:
#                     features['aac'] = aac.to(self.device).float()
#                 y_hat = self.model(train_data, train_length, features)
#                 loss = self.criterion(y_hat, train_label.float())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 # 更新学习率调度器
#                 if self.lr_scheduler is not None:
#                     if hasattr(self.lr_scheduler, 'step'):
#                         self.lr_scheduler.step()
#                     elif callable(self.lr_scheduler):
#                         current_lr = self.lr_scheduler(steps)
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = current_lr
#                     else:
#                         pass  # 如果调度器既没有 step 方法也不可调用，则不进行操作
#
#                 # 记录学习率
#                 x_plot.append(steps)
#                 if self.lr_scheduler is not None:
#                     if hasattr(self.lr_scheduler, 'get_last_lr'):
#                         current_lr = self.lr_scheduler.get_last_lr()[0]
#                     elif callable(self.lr_scheduler):
#                         current_lr = self.lr_scheduler(steps)
#                     else:
#                         current_lr = self.optimizer.param_groups[0]['lr']
#                     y_plot.append(current_lr)
#                 else:
#                     current_lr = self.optimizer.param_groups[0]['lr']
#                     y_plot.append(current_lr)
#
#                 total_loss += loss.item()
#                 steps += 1
#
#             finish = time.time()
#
#             print(f'[ Epoch {epoch} ', end='')
#             print("运行时间{:.2f}s".format(finish - start))
#             print(f'loss={total_loss / len(train_iter)} ]')
#
#         if plot_picture and len(y_plot) > 0:
#             plt.plot(x_plot, y_plot, 'r')
#             plt.title('Learning Rate Schedule')
#             plt.xlabel('Step')
#             plt.ylabel('Learning Rate')
#             plt.savefig('./result/Cos_warmup.jpg')
#
# def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )
#     return LambdaLR(optimizer_, lr_lambda, last_epoch)
#
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                            (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr


#GAN
# import time
# import torch
# import math
# import numpy as np
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
#
# class DataTrain:
#     def __init__(self, model, optimizer, criterion, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, scheduler=None, device="cuda"):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.lr_scheduler = scheduler
#         self.device = device
#         self.model.to(self.device)
#
#         # GAN 相关
#         self.generator = generator
#         self.discriminator = discriminator
#         self.optimizer_G = optimizer_G
#         self.optimizer_D = optimizer_D
#         self.adversarial_loss = adversarial_loss
#         self.generator.to(self.device)
#         self.discriminator.to(self.device)
#
#     def train_step(self, train_iter, epochs=None, plot_picture=True):
#         x_plot = []
#         y_plot = []
#         epochTrainLoss = []
#         steps = 1
#         for epoch in range(1, epochs+1):
#             start = time.time()
#             total_loss = 0
#             for train_data, train_label, train_length, aai, paac, pc6, blosum62, aac in train_iter:
#                 batch_size = train_data.size(0)
#                 self.model.train()  # 进入训练模式
#
#                 # 准备真实数据
#                 train_data = train_data.to(self.device)
#                 train_label = train_label.to(self.device)
#                 train_length = train_length.to(self.device)
#                 features = {'aai': aai.to(self.device), 'paac': paac.to(self.device), 'pc6': pc6.to(self.device), 'blosum62': blosum62.to(self.device), 'aac': aac.to(self.device)}
#
#                 # 将真实数据转换为 one-hot 表示
#                 real_one_hot = F.one_hot(train_data, num_classes=self.generator.vocab_size).float()
#
#                 # ---------------------
#                 #  训练判别器
#                 # ---------------------
#                 self.discriminator.train()
#                 self.optimizer_D.zero_grad()
#
#                 valid = torch.ones(batch_size, 1, device=self.device)
#                 fake = torch.zeros(batch_size, 1, device=self.device)
#
#                 # 判别真实数据
#                 real_validity = self.discriminator(real_one_hot)
#                 d_real_loss = self.adversarial_loss(real_validity, valid)
#
#                 # 生成假数据
#                 z = torch.randn(batch_size, self.generator.model[0].in_features, device=self.device)
#                 gen_data = self.generator(z)
#
#                 # 判别假数据
#                 fake_validity = self.discriminator(gen_data.detach())
#                 d_fake_loss = self.adversarial_loss(fake_validity, fake)
#
#                 d_loss = (d_real_loss + d_fake_loss) / 2
#                 d_loss.backward()
#                 self.optimizer_D.step()
#
#                 # -----------------
#                 #  训练生成器
#                 # -----------------
#                 self.generator.train()
#                 self.optimizer_G.zero_grad()
#
#                 gen_data = self.generator(z)
#                 validity = self.discriminator(gen_data)
#                 g_loss = self.adversarial_loss(validity, valid)
#                 g_loss.backward()
#                 self.optimizer_G.step()
#
#                 # -----------------
#                 #  训练原始模型（ETFC）
#                 # -----------------
#                 self.model.train()
#                 self.optimizer.zero_grad()
#
#                 # 将生成的数据转换为索引序列
#                 gen_indices = torch.argmax(gen_data, dim=-1)  # (batch_size, seq_length)
#
#                 # 准备 ETFC 模型的输入
#                 combined_data = torch.cat((train_data, gen_indices), 0)
#                 combined_labels = torch.cat((train_label, train_label), 0)
#                 combined_length = torch.cat((train_length, train_length), 0)
#                 combined_features = {
#                     'aai': torch.cat((features['aai'], features['aai']), 0),
#                     'paac': torch.cat((features['paac'], features['paac']), 0),
#                     'pc6': torch.cat((features['pc6'], features['pc6']), 0),
#                     'blosum62': torch.cat((features['blosum62'], features['blosum62']), 0),
#                     'aac': torch.cat((features['aac'], features['aac']), 0)
#                 }
#
#                 perm = torch.randperm(combined_data.size(0))
#                 combined_data = combined_data[perm]
#                 combined_labels = combined_labels[perm]
#                 combined_length = combined_length[perm]
#                 for key in combined_features:
#                     combined_features[key] = combined_features[key][perm]
#
#                 y_hat = self.model(combined_data, combined_length, combined_features)
#                 loss = self.criterion(y_hat, combined_labels.float())
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if self.lr_scheduler:
#                     if callable(getattr(self.lr_scheduler, "step", None)):
#                         self.lr_scheduler.step()
#                     else:
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = self.lr_scheduler(steps)
#
#                 x_plot.append(steps)
#                 y_plot.append(self.lr_scheduler(steps) if self.lr_scheduler else self.optimizer.param_groups[0]['lr'])
#                 total_loss += loss.item()
#                 steps += 1
#
#             finish = time.time()
#
#             print(f'[ Epoch {epoch} ]', end='')
#             print(" 运行时间: {:.2f}s".format(finish - start))
#             print(f'Loss={total_loss / len(train_iter):.8f} | D Loss={d_loss.item():.8f} | G Loss={g_loss.item():.8f}')
#
#             epochTrainLoss.append(total_loss / len(train_iter))
#
#         # if plot_picture:
#         #     plt.figure(figsize=(10, 6))
#         #     plt.plot(x_plot, y_plot, 'r')
#         #     plt.title('Learning Rate over Steps')
#         #     plt.xlabel('Step')
#         #     plt.ylabel('Learning Rate')
#         #     plt.ylim(bottom=0)
#         #     plt.tight_layout()
#         #     plt.subplots_adjust(left=0.15)  # 调整左边距
#         #     plt.savefig('./result/Cos_warmup.jpg')
#         #     plt.show()
#         if plot_picture:
#             # 全局设置字体
#             #plt.rcParams['font.family'] = 'Arial'
#
#             # 应用灰度样式
#             plt.style.use('grayscale')
#
#             plt.figure(figsize=(10, 6))
#
#             # 绘制黑色虚线线条
#             plt.plot(x_plot, y_plot, 'k--', linewidth=2)  # 'k--' 表示黑色虚线
#
#             # 设置标题和标签，并确保文字为黑色
#             plt.title('Learning Rate over Steps', color='black', fontsize=16)
#             plt.xlabel('Step', color='black', fontsize=14)
#             plt.ylabel('Learning Rate', color='black', fontsize=14)
#
#             # 添加网格线
#             plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
#
#             # 设置y轴下限为0
#             plt.ylim(bottom=0)
#
#             # 调整布局
#             plt.tight_layout()
#             plt.subplots_adjust(left=0.15)  # 调整左边距
#
#             # 保存为黑白图片
#             plt.savefig('./result/Cos_warmup.jpg', dpi=300, format='jpg', bbox_inches='tight', facecolor='w',
#                         edgecolor='w')
#
#             # 显示图表
#             plt.show()
#
#
# def evaluate(model, datadl, device="cpu"):
#     model.to(device)
#     model.eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for x, y, z, aai, paac, pc6, blosum62, aac in datadl:
#             x = x.to(device).long()
#             y = y.to(device).float()
#             z = z.to(device)
#             features = {'aai': aai.to(device), 'paac': paac.to(device), 'pc6': pc6.to(device), 'blosum62': blosum62.to(device), 'aac': aac.to(device)}
#             outputs = model(x, z, features)
#             predictions.extend(outputs.cpu().numpy())
#             labels.extend(y.cpu().numpy())
#
#     # 根据您的任务定义计算指标的方式
#     # 这里需要您自己实现 evaluation.evaluate 方法
#     scores = evaluation.evaluate(np.array(predictions), np.array(labels))
#     return scores
#
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                            (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr








#用BAS（Beetle Antennae Search）进行超参优化，还有CNN,LSTM的超参也一起优化了
# import time
# import torch
# import math
# import numpy as np
# from sklearn import metrics
# import matplotlib.pyplot as plt
#
# class DataTrain:
#     def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.lr_scheduler = scheduler
#
#         self.device = device
#         self.model.to(self.device)
#
#     def train_step(self, train_iter, epochs=None, plot_picture=True):
#         x_plot = []
#         y_plot = []
#         epochTrainLoss = []
#         steps = 1
#         for epoch in range(1, epochs+1):
#             start = time.time()
#             total_loss = 0
#             for train_data, train_label, train_length, aai, paac, pc6, blosum62,aac in train_iter:
#
#                 self.model.train()  # 进入训练模式
#                 train_data, train_label, train_length = train_data.to(self.device), train_label.to(self.device), train_length.to(self.device)
#                 features = {'aai': aai.to(self.device), 'paac': paac.to(self.device), 'pc6': pc6.to(self.device), 'blosum62': blosum62.to(self.device),'aac':aac.to(self.device)}
#                 y_hat = self.model(train_data, train_length, features)
#                 loss = self.criterion(y_hat, train_label.float())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if self.lr_scheduler:
#                     if hasattr(self.lr_scheduler, 'step'):
#                         self.lr_scheduler.step()
#                     else:
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = self.lr_scheduler(steps)
#
#                 x_plot.append(epoch)
#                 y_plot.append(self.lr_scheduler(epoch))
#                 total_loss += loss.item()
#                 steps += 1
#
#             finish = time.time()
#
#             print(f'[ Epoch {epoch} ', end='')
#             print("运行时间{}s".format(finish - start))
#             print(f'loss={total_loss / len(train_iter)} ]')
#
#         if plot_picture:
#             plt.plot(x_plot, y_plot, 'r')
#             plt.title('lr value of LambdaLR with (Cos_warmup) ')
#             plt.xlabel('step')
#             plt.ylabel('lr')
#             plt.savefig('./result/Cos_warmup.jpg')
#
# def evaluate(model, datadl, device="cpu"):
#     model.to(device)
#     model.eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for x, y, z, aai, paac, pc6, blosum62, aac in datadl:
#             x = x.to(device).long()
#             y = y.to(device)
#             z = z.to(device)
#             features = {
#                 'aai': aai.to(device),
#                 'paac': paac.to(device),
#                 'pc6': pc6.to(device),
#                 'blosum62': blosum62.to(device),
#                 'aac': aac.to(device)
#             }
#             outputs = model(x, z, features)
#             predictions.append(outputs.cpu())
#             labels.append(y.cpu())
#
#     # 将列表中的张量拼接起来
#     predictions_tensor = torch.cat(predictions)
#     labels_tensor = torch.cat(labels)
#
#     # 计算损失和其他指标
#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     loss = loss_fn(predictions_tensor, labels_tensor.float())
#
#     # 如果您的 evaluation.evaluate 函数需要 NumPy 数组，可以在这里进行转换
#     predictions_numpy = predictions_tensor.numpy()
#     labels_numpy = labels_tensor.numpy()
#
#     # 假设您有一个自定义的 evaluation.evaluate 函数
#     scores = evaluation.evaluate(predictions_numpy, labels_numpy)
#     scores["loss"] = loss.item()
#     return scores
#
# def scoring(y_true, y_score):
#     threshold = 0.5
#     y_pred = [int(i >= threshold) for i in y_score]
#     confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = confusion_matrix.flatten()
#     sen = tp / (fn + tp)
#     spe = tn / (fp + tn)
#     pre = metrics.precision_score(y_true, y_pred)
#     auc = metrics.roc_auc_score(y_true, y_score)
#     pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
#     aupr = metrics.auc(rc, pr)
#     f1 = metrics.f1_score(y_true, y_pred)
#     mcc = metrics.matthews_corrcoef(y_true, y_pred)
#     acc = metrics.accuracy_score(y_true, y_pred)
#     return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)
#
# def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )
#     return LambdaLR(optimizer_, lr_lambda, last_epoch)
#
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                            (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr



#转成图卷积
# import torch
# import time
# import matplotlib.pyplot as plt
# from sklearn import metrics
# import numpy as np
#
# from evaluation import evaluate as eval_metric  # 保持 evaluation.py 不变
#
# def evaluate(model, datadl, device="cpu"):
#     model.to(device)
#     model.eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for data in datadl:
#             data = data.to(device)
#             y_hat = model(data)
#             predictions.extend(torch.sigmoid(y_hat).cpu().numpy())
#             labels.extend(data.y.cpu().numpy())
#
#     # 确保 predictions 和 labels 形状为 (n_samples, n_labels)
#     predictions = np.array(predictions).reshape(-1, 21)
#     labels = np.array(labels).reshape(-1, 21)
#
#     # 使用 evaluation.evaluate 函数对预测结果进行评估，返回评估分数
#     scores = evaluation.evaluate(predictions, labels)
#     return scores
#
#
#
#
# # 定义训练类
# class DataTrain:
#     def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.lr_scheduler = scheduler
#         self.device = device
#         self.model.to(self.device)
#
#     def train_step(self, train_iter, epochs=None, plot_picture=False):
#         x_plot = []
#         y_plot = []
#         epochTrainLoss = []
#         steps = 1
#         for epoch in range(1, epochs + 1):
#             start = time.time()
#             total_loss = 0
#             self.model.train()
#             for data in train_iter:
#                 data = data.to(self.device)
#                 self.optimizer.zero_grad()
#                 y_hat = self.model(data)
#
#                 # 确保 data.y 的形状与 y_hat 一致
#                 target = data.y.view(y_hat.size())  # 调整形状
#
#
#                 loss = self.criterion(y_hat, target)
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if self.lr_scheduler:
#                     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
#                         self.lr_scheduler.step()
#                     else:
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = self.lr_scheduler(steps)
#                         steps += 1
#
#                 total_loss += loss.item()
#
#             finish = time.time()
#             avg_loss = total_loss / len(train_iter)
#             epochTrainLoss.append(avg_loss)
#             print(f'[ Epoch {epoch} ', end='')
#             print(f"运行时间{finish - start:.2f}s", end=' ')
#             print(f'loss={avg_loss:.4f} ]')
#
#             if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
#                 current_lr = self.lr_scheduler.get_last_lr()[0]
#                 x_plot.append(epoch)
#                 y_plot.append(current_lr)
#
#         # 绘制学习率变化曲线
#         if plot_picture and self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
#             plt.plot(x_plot, y_plot, 'r')
#             plt.title('Learning Rate Schedule')
#             plt.xlabel('Epoch')
#             plt.ylabel('Learning Rate')
#             plt.savefig('./result/Cos_warmup_lr_schedule.jpg')
#
#
#
#
# # 定义学习率调度器
# def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )
#     return torch.optim.lr_scheduler.LambdaLR(optimizer_, lr_lambda, last_epoch)
#
#
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#         self.base_lr = base_lr
#
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                            (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr


# import time
# import torch
# import math
# import numpy as np
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import shap
#
# def evaluate(model, datadl, device="cpu"):
#     model.to(device)
#     model.eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for x, y, z, aai, paac, pc6, blosum62,aac in datadl:
#             x = x.to(device).long()
#             y = y.to(device).long()
#             features = {'aai': aai.to(device), 'paac': paac.to(device), 'pc6': pc6.to(device), 'blosum62': blosum62.to(device),'aac':aac.to(device)}
#             predictions.extend(model(x, z, features).tolist())
#             labels.extend(y.tolist())
#
#     scores = evaluation.evaluate(np.array(predictions), np.array(labels))
#     return scores
#
#
#
#
#
# def scoring(y_true, y_score):
#     threshold = 0.5
#     y_pred = [int(i >= threshold) for i in y_score]
#     confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = confusion_matrix.flatten()
#     sen = tp / (fn + tp)
#     spe = tn / (fp + tn)
#     pre = metrics.precision_score(y_true, y_pred)
#     auc = metrics.roc_auc_score(y_true, y_score)
#     pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
#     aupr = metrics.auc(rc, pr)
#     f1 = metrics.f1_score(y_true, y_pred)
#     mcc = metrics.matthews_corrcoef(y_true, y_pred)
#     acc = metrics.accuracy_score(y_true, y_pred)
#     return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)
#
# class DataTrain:
#     def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.lr_scheduler = scheduler
#
#         self.device = device
#         self.model.to(self.device)
#
#     def train_step(self, train_iter, epochs=None, plot_picture=True):
#         x_plot = []
#         y_plot = []
#         epochTrainLoss = []
#         steps = 1
#         for epoch in range(1, epochs+1):
#             start = time.time()
#             total_loss = 0
#             for batch in train_iter:
#                 if len(batch) == 9:
#                     x, y, z, aai, paac, pc6, blosum62, aac, edge_index = batch
#                     gnn_features = torch.randn(x.size(0), 256).to(self.device)  # 假设 GNN 输入特征为随机初始化，需根据实际情况替换
#                 elif len(batch) == 10:
#                     x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features = batch
#                 else:
#                     raise ValueError("Unexpected batch size")
#
#                 self.model.train()  # 进入训练模式
#                 x = x.to(self.device).long()
#                 y = y.to(self.device).float()
#                 features = {'aai': aai.to(self.device), 'paac': paac.to(self.device),
#                             'pc6': pc6.to(self.device), 'blosum62': blosum62.to(self.device),
#                             'aac': aac.to(self.device)}
#                 edge_index = edge_index.to(self.device)
#                 # gnn_features 需要根据实际情况获取
#                 y_hat = self.model(x, z, features, edge_index, gnn_features)
#                 loss = self.criterion(y_hat, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if self.lr_scheduler:
#                     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
#                         self.lr_scheduler.step()
#                     else:
#                         for param_group in self.optimizer.param_groups:
#                             param_group['lr'] = self.lr_scheduler(steps)
#
#                 x_plot.append(epoch)
#                 y_plot.append(self.lr_scheduler(steps) if self.lr_scheduler else 0)
#                 total_loss += loss.item()
#                 steps += 1
#
#             finish = time.time()
#
#             print(f'[ Epoch {epoch} ', end='')
#             print("运行时间{}s".format(finish - start))
#             print(f'loss={total_loss / len(train_iter):.4f} ]')
#
#         if plot_picture:
#             plt.plot(x_plot, y_plot, 'r')
#             plt.title('lr value of LambdaLR with (Cos_warmup) ')
#             plt.xlabel('step')
#             plt.ylabel('lr')
#             plt.savefig('./result/Cos_warmup.jpg')
#
# def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
#     from torch.optim.lr_scheduler import LambdaLR
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )
#     return LambdaLR(optimizer_, lr_lambda, last_epoch)
#
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#
#     def __call__(self, step):
#         epoch = step  # 这里假设 step 与 epoch 同步
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                            (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr



# train.py

import time
import torch
import math
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import shap

def evaluate(model, datadl, device="cpu"):
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in datadl:
            if len(batch) == 10:
                x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features = batch
                batch_batch = torch.zeros(gnn_features.size(0), dtype=torch.long).to(device)  # 假设只有一个图
            elif len(batch) == 11:
                x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features, batch_batch = batch
                batch_batch = batch_batch.to(device)
            else:
                raise ValueError("Unexpected batch size")

            x = x.to(device).long()
            y = y.to(device).long()
            features = {
                'aai': aai.to(device),
                'paac': paac.to(device),
                'pc6': pc6.to(device),
                'blosum62': blosum62.to(device),
                'aac': aac.to(device)
            }
            edge_index = edge_index.to(device)
            gnn_features = gnn_features.to(device)
            y_hat = model(x, z, features, edge_index, gnn_features, batch_batch)
            #predictions.extend(y_hat.squeeze().tolist())
            predictions.extend(y_hat.tolist())
           # print(predictions)
            labels.extend(y.tolist())

    scores = evaluation.evaluate(np.array(predictions), np.array(labels))
    return scores

def scoring(y_true, y_score):
    threshold = 0.5
    y_pred = [int(i >= threshold) for i in y_score]
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    if confusion_matrix.size == 1:
        # Handle cases where only one class is present
        tn, fp, fn, tp = 0, 0, 0, 0
        if y_true[0] == 0:
            tn = 1
        else:
            tp = 1
    else:
        tn, fp, fn, tp = confusion_matrix.flatten()
    sen = tp / (fn + tp) if (fn + tp) > 0 else 0
    spe = tn / (fp + tn) if (fp + tn) > 0 else 0
    pre = metrics.precision_score(y_true, y_pred, zero_division=0)
    auc = metrics.roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0
    pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(rc, pr)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    mcc = metrics.matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
    acc = metrics.accuracy_score(y_true, y_pred)
    return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)

class DataTrain:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler

        self.device = device
        self.model.to(self.device)

    def train_step(self, train_iter, epochs=None, plot_picture=True):
        x_plot = []
        y_plot = []
        epochTrainLoss = []
        steps = 1
        for epoch in range(1, epochs+1):
            start = time.time()
            total_loss = 0
            for batch in train_iter:
                if len(batch) == 10:
                    x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features = batch
                    batch_batch = torch.zeros(gnn_features.size(0), dtype=torch.long).to(self.device)  # 假设只有一个图
                elif len(batch) == 11:
                    x, y, z, aai, paac, pc6, blosum62, aac, edge_index, gnn_features, batch_batch = batch
                    batch_batch = batch_batch.to(self.device)
                else:
                    raise ValueError("Unexpected batch size")

                self.model.train()  # 进入训练模式
                x = x.to(self.device).long()
                y = y.to(self.device).float()
                features = {
                    'aai': aai.to(self.device),
                    'paac': paac.to(self.device),
                    'pc6': pc6.to(self.device),
                    'blosum62': blosum62.to(self.device),
                    'aac': aac.to(self.device)
                }
                edge_index = edge_index.to(self.device)
                gnn_features = gnn_features.to(self.device)

                y_hat = self.model(x, z, features, edge_index, gnn_features, batch_batch)
                loss = self.criterion(y_hat, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                        self.lr_scheduler.step()
                    else:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                x_plot.append(epoch)
                y_plot.append(self.lr_scheduler(steps) if self.lr_scheduler else 0)
                total_loss += loss.item()
                steps += 1

            finish = time.time()

            print(f'[ Epoch {epoch} ', end='')
            print(f"运行时间 {finish - start:.2f}s")
            print(f'loss={total_loss / len(train_iter):.8f} ]')

        if plot_picture:
            plt.plot(x_plot, y_plot, 'r')
            plt.title('lr value of LambdaLR with (Cos_warmup)')
            plt.xlabel('step')
            plt.ylabel('lr')
            plt.savefig('./result/Cos_warmup.jpg')

def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer_, lr_lambda, last_epoch)

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        if self.warmup_steps == 0:
            return self.base_lr_orig
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, step):
        epoch = step  # 这里假设 step 与 epoch 同步
        if epoch <= self.warmup_steps and self.warmup_steps > 0:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update and self.max_steps > 0:
            return self.final_lr + (self.base_lr_orig - self.final_lr) * \
                   (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.final_lr






