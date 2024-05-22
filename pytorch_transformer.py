"""pytorch中的Transformer API源码"""
# ------encoder------
# 1、input word embedding 将稀疏的one-hot进入一个不带bias的全连接得到稠密连续向量
# 2、position encoding 增加位置信息
# 3、multi-head self-attention 使得建模能力更强，相关性度量是多种多样的
# 4、feed-forward network 只考虑每个单独位置建模、不同位置参数共享
# ------decoder------
# 1、output word embedding
# 2、masked multi-head self-attention
# 3、multi-head cross-attention
# 4、feed-forward network
# 5、softmax

import torch
import torch.nn as nn

transformer_model = nn.Transformer()
# 1、TransformerEncoderLayer
# 2、TransformerEncoder
# 3、TransformerDecoderLayer
# 4、TransformerDecoder

