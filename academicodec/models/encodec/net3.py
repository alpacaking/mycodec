import math
import random

import numpy as np
import torch.nn as nn
from academicodec.modules.seanet import SEANetDecoder
from academicodec.modules.seanet import SEANetEncoder
from academicodec.quantization import ResidualVectorQuantizer

from academicodec.quantization.fsq import FiniteScalarQuantizer

# Generator
class SoundStream(nn.Module):
    def __init__(self,
                 n_filters,
                 D,
                 target_bandwidths=[1.5, 3.0, 6.0], # kbps
                 ratios=[8, 5, 4, 2],
                 sample_rate=16000,
                 bins=1024, # RVQ码本大小
                 n_q_max=None, # RVQ最大量化器数量
                 normalize=False, # SEANet 中的 normalize 参数，通常为 False
                 use_fsq=False, # 是否使用 FSQ
                 fsq_levels=None): # FSQ 的级别列表
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = np.prod(ratios)
        self.encoder = SEANetEncoder(
            n_filters=n_filters, dimension=D, ratios=ratios)
        
        self.frame_rate = math.ceil(sample_rate / self.hop_length)
        self.bits_per_codebook = int(math.log2(bins)) # 用于 RVQ
        self.target_bandwidths = target_bandwidths
        self.use_fsq = use_fsq

        if use_fsq:
            if FiniteScalarQuantizer is None:
                raise ImportError("FiniteScalarQuantizer not found. Please ensure fsq.py exists in academicodec.quantization.")
            if fsq_levels is None:
                # 默认 FSQ 设置，例如，如果 D=128，可以分成几组，或者每维用相同level
                # 示例：假设 D=128，我们想用 4 个 FSQ 阶段，每个阶段的 levels 不同
                # 这需要 FSQ 实现支持分组或直接处理 D 维。
                # 简单起见，如果 fsq_levels 未提供，可以报错或使用一个简单的默认值
                # 例如，将 D 维平均分配到 len(fsq_levels) 组中，或者每维都用 fsq_levels[0]
                # 这里的 fsq_levels 应该是一个列表，其长度等于 FSQ 的维度或分组数
                # 假设 fsq_levels = [8,6,5,5] 对应 Encodec 论文中的 Nq=4
                # 并且 FSQ 内部处理维度 D 和这些 levels 的映射
                print(f"Warning: fsq_levels not provided for FSQ. Using a default or expecting FSQ to handle it.")
            
            # FSQ 的维度应与编码器输出维度 D 匹配
            # FSQ 的 `levels` 参数通常是一个整数列表，表示每个（子）维度的量化级别数
            self.quantizer = FiniteScalarQuantizer(levels=fsq_levels, dimension=D)
            # FSQ 的 n_q (码本数/阶段数) 由 fsq_levels 的结构决定
            # 例如，如果 fsq_levels = [L1, L2, ..., Lk]，则有 k 个阶段/码本
            self.n_q = len(fsq_levels) if isinstance(fsq_levels, list) else 1

        else: # 使用 RVQ
            if n_q_max is not None and n_q_max > 0 :
                actual_n_q = n_q_max
            else:
                # 根据最大目标带宽计算 RVQ 的 n_q (总码本数)
                # bits_per_frame_for_max_bw = target_bandwidths[-1] * 1000 / self.frame_rate
                # actual_n_q = math.ceil(bits_per_frame_for_max_bw / self.bits_per_codebook)
                # Encodec 论文中 n_q 的计算方式：
                # n_q = floor( (B * 1000) / (R_f * log2(V)) )
                # B: target bitrate (kbps), R_f: frame rate, V: codebook size (bins)
                # Example: 6kbps, 50Hz frame rate, 1024 bins -> floor(6000 / (50 * 10)) = 12
                # The original calculation in your code:
                actual_n_q = int(1000 * target_bandwidths[-1] // (self.frame_rate * self.bits_per_codebook))
                # Ensure n_q is at least 1
                actual_n_q = max(1, actual_n_q)
                print(f"Calculated actual_n_q for RVQ: {actual_n_q} based on max_bw={target_bandwidths[-1]}kbps, frame_rate={self.frame_rate}, bins={bins}")

            self.n_q = actual_n_q
            self.quantizer = ResidualVectorQuantizer(
                dimension=D, n_q=self.n_q, bins=bins)

        self.decoder = SEANetDecoder(
            n_filters=n_filters, dimension=D, ratios=ratios)

    def get_last_layer(self):
        # SEANetDecoder 的最后一层通常是 SConv1d，其内部有 .conv.conv
        # 或者直接是 Conv1d
        # 检查 self.decoder.model 列表中的最后一个元素
        last_conv_module = self.decoder.model[-1] # 通常是激活层之前的卷积
        if hasattr(last_conv_module, 'conv'): # SConv1d or SConvTranspose1d
            return last_conv_module.conv.conv.weight
        elif isinstance(last_conv_module, nn.Conv1d) or isinstance(last_conv_module, nn.ConvTranspose1d):
             return last_conv_module.weight
        print("Warning: Could not get last layer of decoder for adaptive loss weighting.")
        return None


    def forward(self, x):
        e = self.encoder(x)
        
        if self.use_fsq:
            # FSQ 的 forward 可能不需要 frame_rate 和 bw
            # 它应该返回 (quantized_st, codes, bandwidth_tensor, commit_loss_tensor)
            # FSQ 的 commit_loss 通常为0，bandwidth 需要 FSQ 内部计算或返回一个名义值
            quantized, codes, bandwidth, commit_loss = self.quantizer(e)
            # 如果 FSQ 的 forward 签名不同，需要适配
        else: # RVQ
            # 随机选择一个目标带宽进行训练
            if self.training:
                bw_idx = random.randint(0, len(self.target_bandwidths) - 1)
                bw = self.target_bandwidths[bw_idx]
            else: # 评估时通常使用最高或指定的带宽
                bw = self.target_bandwidths[-1]
            
            quantized, codes, bandwidth, commit_loss = self.quantizer(
                e, self.frame_rate, bw)
        
        o = self.decoder(quantized)
        # 确保返回的第三个元素是 last_layer 的权重，如果 loss_g 需要它
        # last_layer_weights = self.get_last_layer()
        return o, commit_loss, None # 暂时保持 None，因为 loss_g 中的 d_weight 计算被注释了

    def encode(self, x, target_bw=None, st=None): # st 参数在 RVQ 中用于分块编码，FSQ 可能不需要
        e = self.encoder(x)
        if self.use_fsq:
            # FSQ 的 encode 可能不需要 frame_rate, bw, st
            codes = self.quantizer.encode(e)
        else: # RVQ
            if target_bw is None:
                # 默认使用最高带宽进行编码
                bw = self.target_bandwidths[-1]
            else:
                bw = target_bw
            
            # st (start_level) for RVQ, default to 0 if not provided
            # RVQ 的 encode 方法签名是 (x, frame_rate, bandwidth, start_level=0)
            # 这里的 st 对应 start_level
            start_level = st if st is not None else 0
            # codes = self.quantizer.encode(e, self.frame_rate, bw, start_level=start_level)
            codes = self.quantizer.encode(e, self.frame_rate, bw, st=start_level)
        return codes

    def decode(self, codes):
        # FSQ 和 RVQ 的 decode 都应该接受 codes 并返回量化后的隐向量
        quantized = self.quantizer.decode(codes)
        o = self.decoder(quantized)
        return o