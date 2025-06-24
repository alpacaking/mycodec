import torch
import torch.nn as nn
import math

class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels: list[int], dimension: int, input_range=(-1.0, 1.0)):
        super().__init__()
        # levels: 一个整数列表，例如 [8, 6, 5, 5] 表示四个阶段/码本，每个阶段的量化级别数
        # dimension: 输入隐向量的维度 D
        # FSQ 通常将 D 维输入分成 Nq 组 (Nq = len(levels))，每组的维度是 D/Nq
        # 或者，如果 FSQ 实现为逐维量化，则 levels 的长度应为 D
        
        self.levels_list = levels # 例如 [8,6,5,5]
        self.num_quantizers = len(self.levels_list) # Nq
        self.dimension = dimension
        
        if dimension % self.num_quantizers != 0:
            raise ValueError(f"Dimension ({dimension}) must be divisible by the number of FSQ levels/quantizers ({self.num_quantizers}).")
        self.sub_dimension = dimension // self.num_quantizers

        self.input_min = input_range[0]
        self.input_max = input_range[1]
        
        self.codebooks = nn.ParameterList()
        for L_d in self.levels_list:
            if L_d <= 1:
                # 对于级别为1的情况，可以将其视为输出固定的0，或者需要特殊处理
                # 这里我们假设级别至少为2，或者FSQ论文中通常不考虑级别为1的情况
                # 如果 L_d=1，linspace 会产生单个值，这可能是期望的
                values = torch.tensor([ (self.input_min + self.input_max) / 2.0 ]) if L_d == 1 else torch.linspace(self.input_min, self.input_max, L_d)
            else:
                values = torch.linspace(self.input_min, self.input_max, L_d)
            self.codebooks.append(nn.Parameter(values, requires_grad=False))

    def _quantize_scalar_to_index(self, x_sub_dim, codebook_d):
        # x_sub_dim: (B*T, sub_dimension)
        # codebook_d: (L_d,)
        # FSQ 通常是逐标量维度进行量化
        
        # (B*T*sub_dimension, 1) vs (1, L_d) -> (B*T*sub_dimension, L_d)
        x_flat = x_sub_dim.reshape(-1, 1)
        distances = torch.abs(x_flat - codebook_d.unsqueeze(0))
        indices = torch.argmin(distances, dim=1) # (B*T*sub_dimension,)
        quantized_values = codebook_d[indices] # (B*T*sub_dimension,)
        
        return indices.reshape_as(x_sub_dim), quantized_values.reshape_as(x_sub_dim)


    def forward(self, x):
        # x: (B, D, T_frames) from SEANetEncoder
        B, D, T_frames = x.shape
        x_permuted = x.permute(0, 2, 1) # (B, T_frames, D)
        x_reshaped = x_permuted.reshape(-1, D) # (B*T_frames, D)

        # 将 D 维分割成 num_quantizers 组，每组 sub_dimension 维
        x_split = torch.split(x_reshaped, self.sub_dimension, dim=1) # Tuple of (B*T, sub_dimension)

        all_indices_grouped = []
        all_quantized_grouped = []
        
        for i in range(self.num_quantizers):
            sub_x = x_split[i] # (B*T, sub_dimension)
            codebook_i = self.codebooks[i] # (L_i,)
            
            # 对 sub_dimension 中的每一维独立量化
            indices_i_list = []
            quantized_i_list = []
            for j in range(self.sub_dimension):
                scalar_input = sub_x[:, j].unsqueeze(1) # (B*T, 1)
                # (B*T, 1) vs (1, L_i) -> (B*T, L_i)
                distances = torch.abs(scalar_input - codebook_i.unsqueeze(0))
                idx_j = torch.argmin(distances, dim=1) # (B*T,)
                quantized_j = codebook_i[idx_j] # (B*T,)
                indices_i_list.append(idx_j.unsqueeze(1))
                quantized_i_list.append(quantized_j.unsqueeze(1))

            indices_i = torch.cat(indices_i_list, dim=1) # (B*T, sub_dimension)
            quantized_i = torch.cat(quantized_i_list, dim=1) # (B*T, sub_dimension)

            all_indices_grouped.append(indices_i)
            all_quantized_grouped.append(quantized_i)
            
        # 合并所有组的索引和量化值
        # codes: (B*T, D) - 每个元素是其对应码本的索引
        # FSQ 的 "codes" 通常是每个阶段/码本的索引序列
        # (B*T, num_quantizers, sub_dimension) -> (B, T, num_quantizers, sub_dimension) -> (B, num_quantizers, sub_dimension, T)
        # 或者更简单地，将索引视为 (B*T, num_quantizers) 如果每个 sub_dimension 共享一个索引（FSQ论文方式）
        # 这里的实现是每个标量都有一个索引，所以 codes 形状是 (B*T, D)
        # 为了与 RVQ 的 codes (B, N_q, T) 对齐，我们需要调整 FSQ codes 的表示
        # FSQ 论文中的 codes 是 (B, T, N_q)，每个元素是 0 到 L_i-1 的整数
        # 让我们调整 FSQ 的 codes 输出为 (B, N_q, T_frames)
        
        # 当前 all_indices_grouped 是一个 list of (B*T, sub_dimension)
        # 我们需要每个 (B*T, sub_dimension) 得到一个单一的索引（如果 sub_dimension > 1，这不标准）
        # FSQ 论文中，每个 quantizer (stage) 输出一个 index.
        # 这里我们假设每个 self.codebooks[i] 对应一个 stage，它量化 self.sub_dimension 维的输入。
        # 如果 sub_dimension > 1，FSQ 通常意味着对这 sub_dimension 维的每一维都用同一个 codebook_i。
        
        # 重新思考 FSQ 索引：
        # FSQ (levels=[L1,L2,...LNq]) 对 D 维输入 x，将其分为 Nq 组，每组 D/Nq 维。
        # 第 i 组 (D/Nq 维) 被量化到 Li 个级别。
        # 输出是 Nq 个索引，每个索引在 [0, Li-1] 之间。
        
        final_indices_list = [] # List of (B*T, 1)
        final_quantized_list = [] # List of (B*T, sub_dimension)

        for i in range(self.num_quantizers):
            sub_x_group = x_split[i] # (B*T, sub_dimension)
            codebook_i = self.codebooks[i] # (L_i,)
            
            # 对 sub_x_group (D/Nq 维) 找到一个索引 (0 to L_i-1)
            # 这是 FSQ 的核心，通常通过 product quantization 的思想，或者更简单地，
            # 将 D/Nq 维向量映射到 L_i 个原型中的一个。
            # 简单 FSQ：对 D/Nq 维的每一维独立量化，然后组合索引。
            # 假设 levels=[8,6,5,5]，D=128, Nq=4, sub_dim=32.
            # 第1组32维，用8级量化。这通常意味着这32维共享一个8级码本，或者每维用一个8级码本。
            # 如果每维用一个8级码本，则有 32*log2(8) bits。
            # 如果32维共享一个8级码本（即从8个32维原型中选一个），则需要一个 (L_i, sub_dimension) 的码本。
            # 当前 self.codebooks[i] 是 (L_i,)，意味着标量量化。

            # 假设我们对 sub_dimension 的每一维都用 codebooks[i] 进行量化
            group_indices_list = []
            group_quantized_list = []
            for j in range(self.sub_dimension):
                scalar_input = sub_x_group[:, j] # (B*T,)
                distances = torch.abs(scalar_input.unsqueeze(1) - codebook_i.unsqueeze(0)) # (B*T, L_i)
                idx_j = torch.argmin(distances, dim=1) # (B*T,)
                quantized_j = codebook_i[idx_j] # (B*T,)
                group_indices_list.append(idx_j.unsqueeze(1)) # (B*T, 1)
                group_quantized_list.append(quantized_j.unsqueeze(1)) # (B*T, 1)
            
            # final_indices_list.append(torch.cat(group_indices_list, dim=1)) # (B*T, sub_dimension) - 这不是单个索引
            final_quantized_list.append(torch.cat(group_quantized_list, dim=1)) # (B*T, sub_dimension)

            # 为了得到 Nq 个索引，我们需要一种方法从 group_indices_list (sub_dimension 个索引) 得到一个索引
            # FSQ 论文的方式是，每个 quantizer i 有一个 levels[i]，它直接作用于 D/Nq 维的输入，
            # 并产生一个单一的索引。这通常意味着 D/Nq 维的输入被 project 到一个标量，然后量化，
            # 或者使用更复杂的映射。
            # 为了简单和与当前代码结构兼容，我们让每个 "stage" i (对应 levels[i])
            # 输出一个索引。如果 sub_dimension > 1，我们可以取均值然后量化，或者只取第一维。
            # 让我们假设 FSQ 的 "codes" 是 (B, N_q, T_frames)，每个索引是 0 到 L_i-1。
            # 我们需要从 x_split[i] (B*T, sub_dimension) 得到一个 (B*T, 1) 的索引。
            # 简单处理：只用每组的第一维进行索引选择，但用所有维度进行量化值重构。
            first_dim_of_group = x_split[i][:, 0] # (B*T,)
            distances = torch.abs(first_dim_of_group.unsqueeze(1) - codebook_i.unsqueeze(0))
            single_index_for_group = torch.argmin(distances, dim=1) # (B*T,)
            final_indices_list.append(single_index_for_group.unsqueeze(1)) # (B*T, 1)


        quantized_flat = torch.cat(final_quantized_list, dim=1) # (B*T, D)
        codes_flat = torch.cat(final_indices_list, dim=1) # (B*T, N_q)

        # Reshape back
        quantized_reshaped = quantized_flat.reshape(B, T_frames, D)
        quantized_out = quantized_reshaped.permute(0, 2, 1) # (B, D, T_frames)
        
        codes_reshaped = codes_flat.reshape(B, T_frames, self.num_quantizers)
        codes_out = codes_reshaped.permute(0, 2, 1) # (B, N_q, T_frames)

        # Straight-through estimator
        quantized_st = x + (quantized_out - x).detach()
        
        # Commit loss for FSQ is typically 0, as codebooks are not learned,
        # or can be MSE between input to quantizer and output of quantizer if codebooks are learned.
        # Here, codebooks are fixed.
        commit_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Bandwidth calculation for FSQ
        # bits_per_frame = sum(math.log2(L) for L in self.levels_list)
        # bandwidth_kbps = bits_per_frame * frame_rate / 1000.0 (frame_rate needs to be passed or known)
        # For now, return a dummy bandwidth
        bandwidth_tensor = torch.tensor(0.0, device=x.device, dtype=x.dtype) 
                                     # Placeholder, actual bandwidth depends on frame_rate

        return quantized_st, codes_out, bandwidth_tensor, commit_loss

    def encode(self, x):
        # x: (B, D, T_frames)
        # Returns codes: (B, N_q, T_frames)
        _, codes, _, _ = self.forward(x) # Use forward to get codes
        return codes

    def decode(self, codes):
        # codes: (B, N_q, T_frames)
        B, N_q, T_frames = codes.shape
        if N_q != self.num_quantizers:
            raise ValueError(f"Number of code groups in input codes ({N_q}) does not match FSQ quantizers ({self.num_quantizers})")

        D = self.dimension  # 使用类初始化时传入的维度值
        
        codes_permuted = codes.permute(0, 2, 1) # (B, T_frames, N_q)
        codes_flat = codes_permuted.reshape(-1, N_q) # (B*T_frames, N_q)

        all_decoded_groups = []
        for i in range(self.num_quantizers):
            indices_i = codes_flat[:, i] # (B*T_frames,)
            codebook_i = self.codebooks[i] # (L_i,)
            
            # Clamp indices to be safe
            indices_i = torch.clamp(indices_i, 0, len(codebook_i) - 1)
            
            # We need to reconstruct sub_dimension vectors from a single index and a scalar codebook.
            # This implies that each of the sub_dimension scalars was quantized to codebook_i.
            # So, we take the indexed value from codebook_i and repeat it sub_dimension times.
            decoded_scalars_for_group = codebook_i[indices_i] # (B*T_frames,)
            decoded_group = decoded_scalars_for_group.unsqueeze(1).repeat(1, self.sub_dimension) # (B*T, sub_dimension)
            all_decoded_groups.append(decoded_group)
        
        decoded_flat = torch.cat(all_decoded_groups, dim=1) # (B*T_frames, D)
        decoded_reshaped = decoded_flat.reshape(B, T_frames, D)
        decoded_out = decoded_reshaped.permute(0, 2, 1) # (B, D, T_frames)
        return decoded_out