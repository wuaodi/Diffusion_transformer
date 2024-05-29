"""Transformer的encoder部分实现"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 构建输入输出的序列，将句子中的文字单词变成单词表的索引
max_words_num = 8  # 单词表中单词的数量
max_seq_len = 5  # 定义最大序列长度
batch_size= 2
src_len = torch.tensor([2, 4], dtype=torch.int32)  # 两个句子，第一个句子有2个单词，第二个句子有4个单词
tgt_len = torch.tensor([4, 3], dtype=torch.int32)  # 两个句子，第一个句子有4个单词，第二个句子有3个单词
# 单词索引构成的句子，单词表长度为8，0位置空出来用于padding，使用 0 padding为每个句子长度为max_seq_len，
src_seq = [F.pad(torch.randint(1, 8, (L,)), (0, max_seq_len-L)) for L in src_len]
tgt_seq = [F.pad(torch.randint(1, 8, (L,)), (0, max_seq_len-L)) for L in tgt_len]
# 合并为一个张量
src_seq = torch.stack(src_seq, dim=0)
tgt_seq = torch.stack(tgt_seq, dim=0)
# print('输入序列单词索引：', src_seq)

# 构造word embedding，目的是word2vec，将每个单词索引变换为一个数值向量
# 进一步的深层次理解：索引是one-hot编码，word embedding是对稀疏的编码乘以一个矩阵变换为另一个向量
model_dim = 8  # 原文是512，即用一个长度为512的向量表示一个单词
src_embedding_table = nn.Embedding(max_words_num+1, model_dim)  # 在训练的过程中参数会自动学习和更新
tgt_embedding_table = nn.Embedding(max_words_num+1, model_dim)
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)
# print('src_embedding_table: ', src_embedding_table.weight)
# print('src_embedding: ', src_embedding)

# 构造position embedding，维度与word_embedding一样，是在单词embedding之后添加的
pos_mat = torch.arange(max_seq_len).reshape([-1, 1])
i_mat = torch.pow(10000, torch.arange(0, model_dim, 2).reshape([1, -1]) / model_dim)
pe_embedding_table = torch.zeros(max_seq_len, model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)  # torch.Size([5, 8])
# print('position_embedding_table: ', pe_embedding_table)

# 将位置编码添加到词嵌入中
src_embedding += pe_embedding_table.unsqueeze(0)  # torch.Size([1, 5, 8]); src_embedding torch.Size([2, 5, 8])
tgt_embedding += pe_embedding_table.unsqueeze(0)
# print('src_embedding with position encoding: ', src_embedding)
# print('tgt_embedding with position encoding: ', tgt_embedding)

# softmax演示，解释为什么需要除以根号下dk
# alpha1 = 0.1
# alpha2 = 10
# score = torch.randn(5)
# prob1 = F.softmax(score*alpha1, -1)
# prob2 = F.softmax(score*alpha2, -1)
# print(score)
# print(prob1)  # 分布还可以
# print(prob2)  # 缩放之后，差异特别大

# 构造encoder的masked self-attention
# masked的shape：[batch_size, max_seq_len, max_seq_len]，值为1或者负无穷
valid_position = [F.pad(torch.ones(L), (0, max_seq_len-L)) for L in src_len]
valid_position = torch.stack(valid_position, dim=0)
valid_position = torch.unsqueeze(valid_position, 2)
valid_position_matrix = torch.bmm(valid_position, valid_position.transpose(1,2))  # 自己与自己的转置相乘
invalid_position_matrix = 1 - valid_position_matrix
mask_self_attention = invalid_position_matrix.to(torch.bool) # True代表这些位置需要mask
# 随机一个score
score = torch.randn(batch_size, max_seq_len, max_seq_len)
# 根据socre和mask的位置得到相关性矩阵
masked_score = score.masked_fill(mask_self_attention, -1e9)  # 不相关部分给一个很大的负值
prob = F.softmax(masked_score, -1)  # 最后一个维度做softmax
print(masked_score)
print(prob)
