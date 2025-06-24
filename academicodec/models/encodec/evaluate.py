import torch
import torchaudio
import argparse
from academicodec.models.encodec.net3 import SoundStream
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import os
import librosa
from tqdm import tqdm
from academicodec.models.encodec.test import remove_encodec_weight_norm, save_audio
from collections import OrderedDict
from pathlib import Path
import numpy as np

def load_model_from_checkpoint(model_path, args):
    """加载模型从检查点，支持RVQ和FSQ配置"""
    # 创建模型实例，考虑RVQ和FSQ的参数差异
    soundstream_model = SoundStream(
        n_filters=args.n_filters,
        D=args.D,
        ratios=args.ratios,
        sample_rate=args.sample_rate,
        target_bandwidths=args.target_bandwidths,
        bins=args.bins,
        n_q_max=args.n_q_max,
        use_fsq=args.use_fsq,
        fsq_levels=args.fsq_levels if args.use_fsq else None
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 处理不同类型的检查点保存格式
    if isinstance(checkpoint, dict) and 'soundstream' in checkpoint:
        # 从训练脚本保存的checkpoint (latest.pth格式)
        state_dict = checkpoint['soundstream']
    else:
        # 直接保存的模型state_dict
        state_dict = checkpoint
    
    # 处理可能的DDP模型前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    soundstream_model.load_state_dict(state_dict)
    remove_encodec_weight_norm(soundstream_model)
    return soundstream_model

def compute_bitrate(args):
    """计算模型的比特率"""
    if args.use_fsq:
        # FSQ的比特率计算
        bits_per_vector = sum(np.log2(level) for level in args.fsq_levels)
        hop_length = np.prod(args.ratios)
        frame_rate = args.sample_rate / hop_length
        bitrate = bits_per_vector * frame_rate / 1000  # kbps
        return bitrate
    else:
        # 直接返回目标评估带宽
        return args.target_eval_bandwidth

def main_evaluate():
    parser = argparse.ArgumentParser(description="评估音频编解码模型")
    # 基本参数
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_audio_dir", type=str, required=True, help="测试音频目录")
    parser.add_argument("--output_recon_dir", type=str, default="reconstructed_audio", help="重建音频输出目录")
    parser.add_argument("--sample_rate", "--sr", type=int, default=16000, help="音频采样率")
    parser.add_argument("--target_eval_bandwidth", type=float, default=6.0, help="评估带宽 (kbps)")
    
    # 模型配置参数
    parser.add_argument("--ratios", type=int, nargs="+", default=[8, 5, 4, 2], help="编码器/解码器下采样比率")
    parser.add_argument("--target_bandwidths", type=float, nargs="+", default=[1.5, 3.0, 6.0], help="训练目标带宽")
    parser.add_argument("--n_filters", type=int, default=32, help="模型滤波器数量")
    parser.add_argument("--D", type=int, default=128, help="隐空间维度")
    
    # RVQ参数
    parser.add_argument("--bins", type=int, default=1024, help="RVQ码本大小")
    parser.add_argument("--n_q_max", type=int, default=None, help="RVQ最大量化器数量")
    
    # FSQ参数
    parser.add_argument("--use_fsq", action="store_true", help="使用FSQ量化")
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=None, help="FSQ量化级别")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_recon_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 计算实际比特率
    actual_bitrate = compute_bitrate(args)
    print(f"评估比特率: {actual_bitrate:.2f} kbps")
    
    # 加载模型
    codec_model = load_model_from_checkpoint(args.model_checkpoint_path, args)
    codec_model.to(device)
    
    # 初始化评估指标
    pesq_metric_nb = PerceptualEvaluationSpeechQuality(fs=args.sample_rate, mode='nb').to(device)
    pesq_metric_wb = PerceptualEvaluationSpeechQuality(fs=args.sample_rate, mode='wb').to(device)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=args.sample_rate, extended=False).to(device)
    
    # 保存结果
    results = {"pesq_nb": [], "pesq_wb": [], "stoi": []}
    
    # 获取所有音频文件
    audio_files = list(Path(args.test_audio_dir).glob('**/*.wav')) + list(Path(args.test_audio_dir).glob('**/*.flac'))
    if not audio_files:
        print(f"警告: 在 {args.test_audio_dir} 中未找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个测试音频文件")
    
    for audio_file_path in tqdm(audio_files, desc="评估进度"):
        try:
            # 加载音频
            original_wav_np, _ = librosa.load(str(audio_file_path), sr=args.sample_rate, mono=True)
            original_wav = torch.from_numpy(original_wav_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
            
            # 编码和解码
            with torch.no_grad():
                if args.use_fsq:
                    # FSQ编码不需要target_bw参数
                    encoded_frames = codec_model.encode(original_wav)
                else:
                    # RVQ编码需要target_bw参数
                    encoded_frames = codec_model.encode(original_wav, target_bw=args.target_eval_bandwidth)
                
                reconstructed_wav = codec_model.decode(encoded_frames)
            
            # 准备评估
            original_wav_eval = original_wav.squeeze(0).cpu()
            reconstructed_wav_eval = reconstructed_wav.squeeze(0).cpu()
            
            # 确保长度一致
            min_len = min(original_wav_eval.shape[-1], reconstructed_wav_eval.shape[-1])
            original_wav_eval = original_wav_eval[..., :min_len]
            reconstructed_wav_eval = reconstructed_wav_eval[..., :min_len]
            
            # 保存重建音频
            output_path = Path(args.output_recon_dir) / audio_file_path.name
            save_audio(reconstructed_wav_eval, output_path, args.sample_rate)
            
            # 计算指标
            results["pesq_nb"].append(pesq_metric_nb(reconstructed_wav_eval, original_wav_eval).item())
            results["pesq_wb"].append(pesq_metric_wb(reconstructed_wav_eval, original_wav_eval).item())
            results["stoi"].append(stoi_metric(reconstructed_wav_eval, original_wav_eval).item())
            
        except Exception as e:
            print(f"处理 {audio_file_path.name} 时出错: {e}")
            continue
    
    # 打印结果
    print("\n--- 评估结果摘要 ---")
    print(f"模型: {Path(args.model_checkpoint_path).parent.name}")
    print(f"比特率: {actual_bitrate:.2f} kbps")
    
    for metric_name, values in results.items():
        if values:
            avg_value = sum(values) / len(values)
            print(f"平均 {metric_name.upper()}: {avg_value:.4f}")
    
    # 保存结果到文件
    result_file = Path(args.output_recon_dir) / "evaluation_results.txt"
    with open(result_file, "w") as f:
        f.write(f"模型: {Path(args.model_checkpoint_path).parent.name}\n")
        f.write(f"比特率: {actual_bitrate:.2f} kbps\n")
        for metric_name, values in results.items():
            if values:
                avg_value = sum(values) / len(values)
                f.write(f"平均 {metric_name.upper()}: {avg_value:.4f}\n")
    
    print(f"结果已保存到 {result_file}")

if __name__ == '__main__':
    main_evaluate()