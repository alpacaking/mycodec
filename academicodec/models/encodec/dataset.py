import glob
import random

import torch
import torchaudio
from torch.utils.data import Dataset

import torchaudio
import os
from torch.utils.data import Dataset
import random
import glob # 用于 NSynthDataset
from pathlib import Path # 导入 Path

class NSynthDataset(Dataset): # 保留现有的 NSynthDataset
    """用于加载 NSynth 数据的 Dataset。"""

    def __init__(self, audio_dir):
        super().__init__()
        self.filenames = []
        self.filenames.extend(glob.glob(audio_dir + "/*.wav"))
        print(f"在 NSynth 目录中找到 {len(self.filenames)} 个文件: {audio_dir}")
        if not self.filenames:
            raise FileNotFoundError(f"在 {audio_dir} 中未找到 .wav 文件")
        _, self.sr = torchaudio.load(self.filenames[0])
        self.max_len = 24000  # 24000, 或者使其可配置

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        try:
            audio, sr = torchaudio.load(self.filenames[index])
            if sr != self.sr: # 必要时重采样
                audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(audio)
            if audio.shape[0] > 1: # 转为单声道
                audio = torch.mean(audio, dim=0, keepdim=True)

            if audio.shape[1] > self.max_len:
                st = random.randint(0, audio.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                return audio[:, st:ed]
            else:
                ans[:, :audio.shape[1]] = audio
                return ans
        except Exception as e:
            print(f"加载或处理 {self.filenames[index]} 时出错: {e}")
            # 返回一个随机的有效样本或零张量
            if len(self.filenames) > 1:
                 return self.__getitem__(random.randint(0, len(self.filenames) -1))
            return ans


class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, segment_length_secs=1.5, sample_rate=16000, subset="train-clean-100", file_ext=".flac"):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length_secs * sample_rate)
        
        self.walker = []
        subset_path = os.path.join(root_dir, subset)
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f"LibriSpeech 子集 {subset_path} 未找到。请下载并准备好。")

        for speaker_id_path in Path(subset_path).iterdir():
            if speaker_id_path.is_dir():
                for chapter_id_path in speaker_id_path.iterdir():
                    if chapter_id_path.is_dir():
                        for file_path in chapter_id_path.glob(f"*{file_ext}"):
                            self.walker.append(str(file_path))
        
        if not self.walker:
            raise RuntimeError(f"在 {subset_path} 中未找到 {file_ext} 文件。请检查数据集路径和结构。")

        print(f"在 LibriSpeech 子集中找到 {len(self.walker)} 个音频文件: {subset}.")
        # 如果不重新采样所有文件，你可能需要验证第一个文件的采样率
        # info = torchaudio.info(self.walker[0])
        # self.original_sr = info.sample_rate 

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, index):
        filepath = self.walker[index]
        try:
            wav, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"加载 {filepath} 时出错: {e}。跳过或返回虚拟数据。")
            # 后备：返回一个随机的不同样本或零张量
            if len(self.walker) > 1:
                return self.__getitem__(random.randint(0, len(self.walker) - 1))
            return torch.zeros(1, self.segment_samples)


        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)

        if wav.shape[0] > 1: # 转换为单声道
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        # 填充或截断/裁剪
        current_len = wav.shape[1]
        if current_len < self.segment_samples:
            padding_needed = self.segment_samples - current_len
            wav = torch.nn.functional.pad(wav, (0, padding_needed))
        elif current_len > self.segment_samples:
            start = random.randint(0, current_len - self.segment_samples)
            wav = wav[:, start:start + self.segment_samples]
        
        return wav # 形状: (1, segment_samples)