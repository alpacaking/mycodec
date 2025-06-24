import argparse
import itertools
import os
import time

import torch
import torch.distributed as dist
from academicodec.models.encodec.dataset import NSynthDataset , LibriSpeechDataset
from academicodec.models.encodec.loss import criterion_d
from academicodec.models.encodec.loss import criterion_g
from academicodec.models.encodec.loss import loss_dis
from academicodec.models.encodec.loss import loss_g
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.encodec.net3 import SoundStream
from academicodec.models.soundstream.models import MultiPeriodDiscriminator
from academicodec.models.soundstream.models import MultiScaleDiscriminator
from academicodec.utils import Logger
from academicodec.utils import seed_everything
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    # args for random
    parser.add_argument(
        '--seed',
        type=int,
        default=6666,
        help='seed for initializing training. ')
    parser.add_argument(
        '--cudnn_deterministic',
        action='store_true',
        help='set cudnn.deterministic True')
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='use tensorboard for logging')

    # args for training
    parser.add_argument(
        '--LAMBDA_WAV', # 这个在 loss.py 中 reconstruction_loss 使用了，但 Encodec 主要用 Mel 损失
        type=float,
        default=1.0, # Encodec 论文中 reconstruction loss (mel) 权重为 1
        help='hyper-parameter for wav time-domain loss (or part of Mel-reconstruction)')
    parser.add_argument(
        '--LAMBDA_ADV',
        type=float,
        default=1.0, # Encodec 论文中对抗损失权重为 1
        help='hyper-parameter for adver loss')
    parser.add_argument(
        '--LAMBDA_FEAT',
        type=float,
        default=2.0, # Encodec 论文中特征匹配损失权重为 2
        help='hyper-parameter for feat loss')
    parser.add_argument(
        '--LAMBDA_REC', # 这个参数在 loss.py 的 loss_g 中没有直接使用，而是 reconstruction_loss 的总权重
        type=float,
        default=1.0, # 保持为1，具体各项在 reconstruction_loss 内部处理
        help='hyper-parameter for overall rec loss weight')
    parser.add_argument(
        '--LAMBDA_COM',
        type=float,
        default=0.25, # Encodec 论文中 commitment loss 权重为 0.25
        help='hyper-parameter for commit loss')
    parser.add_argument(
        '--N_EPOCHS', type=int, default=100, help='Total training epoch')
    parser.add_argument(
        '--st_epoch', type=int, default=0, help='start training epoch')
    parser.add_argument(
        '--global_step', type=int, default=0, help='record the global step')
    parser.add_argument(
        '--discriminator_iter_start', 
        type=int, 
        default=1000, # Encodec 论文中判别器在1000步后开始训练，但这里的实现是判别器损失权重在1000步后生效
        help='Iteration to start applying discriminator loss weight fully')
    parser.add_argument(
        '--BATCH_SIZE', type=int, default=8, help='batch size per GPU') # 调整为你GPU能承受的大小
    parser.add_argument(
        '--PATH', type=str, default='experiments/my_rvqgan_codec', help='model save path') # 修改为你希望的路径
    parser.add_argument('--sr', type=int, default=16000, help='sample rate')
    parser.add_argument(
        '--print_freq', type=int, default=100, help='the print number of iterations') # 调整打印频率
    parser.add_argument(
        '--save_dir', type=str, default='runs', help='tensorboard log save path') # 修改为你希望的路径
    
    # --- 数据集路径修改 ---
    parser.add_argument(
        '--dataset_type',
        type=str,
        default="librispeech", # 'librispeech' or 'nsynth'
        help='Type of dataset to use')
    parser.add_argument(
        '--librispeech_root_dir',
        type=str,
        default='~/datasets/LibriSpeech/LibriSpeech', # 指向包含 train-clean-100 等子目录的 LibriSpeech 文件夹
        help='Root directory of LibriSpeech dataset')
    parser.add_argument(
        '--librispeech_train_subset',
        type=str,
        default='train-clean-100',
        help='LibriSpeech training subset')
    parser.add_argument(
        '--librispeech_valid_subset',
        type=str,
        default='dev-clean',
        help='LibriSpeech validation subset')
    parser.add_argument(
        '--nsynth_train_data_path', # 如果使用 NSynth
        type=str,
        default="/path/to/nsynth/train",
        help='NSynth training data directory')
    parser.add_argument(
        '--nsynth_valid_data_path', # 如果使用 NSynth
        type=str,
        default="/path/to/nsynth/valid",
        help='NSynth validation data directory')
    parser.add_argument(
        '--segment_duration_secs',
        type=float,
        default=1.5, # 音频片段长度（秒）
        help='Duration of audio segments for training')

    parser.add_argument(
        '--resume', action='store_true', help='whether re-train model')
    parser.add_argument(
        '--resume_path', type=str, default=None, help='resume_path (directory containing latest.pth)')
    
    # --- SoundStream 模型参数 ---
    parser.add_argument(
        '--n_filters', type=int, default=32, help='Initial number of filters in SEANet')
    parser.add_argument(
        '--D', type=int, default=128, help='Dimension of the latent space for RVQ') # Encodec 使用 128
    parser.add_argument(
        '--ratios',
        type=int,
        nargs='+',
        default=[8, 5, 4, 2], # Encodec for 24kHz uses [8,5,4,2] -> hop_size 320. For 16kHz, might need adjustment if target hop_size is different.
                               # If sr=16kHz, hop_length = 320 -> frame_rate = 16000/320 = 50 Hz.
        help='Ratios of SoundStream SEANet encoder/decoder layers')
    parser.add_argument(
        '--target_bandwidths', # 单位 kbps
        type=float,
        nargs='+',
        default=[1.5, 3.0, 6.0], # 示例比特率，你可以根据需求调整
        help='Target bandwidths for RVQ training')
    parser.add_argument(
        '--bins', type=int, default=1024, help='Number of entries in each RVQ codebook')
    parser.add_argument(
        '--n_q_max', type=int, default=8, help='Maximum number of quantizers in RVQ. SoundStream will calculate actual n_q based on max bandwidth if this is not used directly.')


    # --- FSQ 参数 (用于实验 5.2) ---
    parser.add_argument(
        '--use_fsq', action='store_true', help='Use Finite Scalar Quantizer instead of RVQ')
    parser.add_argument(
        '--fsq_levels', type=int, nargs='+', default=[8, 6, 5, 5], # 示例: 4个量化级别组，对应论文中的 Nq=4, levels=[8,6,5,5]
        help='List of levels for each dimension/group in FSQ. Length should match D or D/num_fsq_groups if grouping.')


    args = parser.parse_args()
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    
    # 确保 PATH 和 save_dir 中的 ~ 被正确展开
    args.PATH = os.path.expanduser(args.PATH)
    args.save_dir = os.path.expanduser(args.save_dir)
    args.librispeech_root_dir = os.path.expanduser(args.librispeech_root_dir)


    if args.resume and args.resume_path is not None:
        args.resume_path = os.path.expanduser(args.resume_path)
        args.PATH = args.resume_path  # direcly use the old model path
    else:
        args.PATH = os.path.join(args.PATH, time_str)
    
    args.save_dir = os.path.join(args.save_dir, args.PATH.split('/')[-1]) # 使用和模型路径一样的子目录名

    os.makedirs(args.PATH, exist_ok=True)
    # os.makedirs(args.save_dir, exist_ok=True) # Logger 会创建它
    return args



def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()


def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    args.ngpus_per_node = torch.cuda.device_count()
    main_worker(args.local_rank, args)


def main_worker(local_rank, args):
    rank = local_rank
    args.local_rank = local_rank
    # args.global_rank = local_rank
    
    # args.global_rank = local_rank # global_rank 通常是 node_rank * ngpus_per_node + local_rank
    # 对于单节点训练，local_rank 就是 global_rank
    if args.ngpus_per_node > 1:
        args.global_rank = dist.get_rank() # 获取正确的全局 rank
    else:
        args.global_rank = local_rank
        
    args.distributed = args.ngpus_per_node > 1

    if args.ngpus_per_node > 1:
        from torch.distributed import init_process_group
        torch.cuda.set_device(local_rank)
        # init_process_group(backend='nccl')
        init_process_group(backend='nccl', rank=args.global_rank, world_size=args.ngpus_per_node)

    #CUDA_VISIBLE_DEVICES = int(args.local_rank)
    logger = Logger(args)
    
    # --- 实例化 SoundStream 模型 ---
    soundstream = SoundStream(
        n_filters=args.n_filters, 
        D=args.D, 
        ratios=args.ratios,
        sample_rate=args.sr,
        target_bandwidths=args.target_bandwidths,
        bins=args.bins,
        n_q_max=args.n_q_max, # 传递 n_q_max
        use_fsq=args.use_fsq, # 传递 FSQ 参数
        fsq_levels=args.fsq_levels # 传递 FSQ 参数
    )
    msd = MultiScaleDiscriminator()
    mpd = MultiPeriodDiscriminator()
    stft_disc = MultiScaleSTFTDiscriminator(filters=32)

    if logger.is_primary:
        getModelSize(soundstream)
        getModelSize(msd)
        getModelSize(mpd)
        getModelSize(stft_disc)

    # --- 数据集加载 ---
    if args.dataset_type == 'librispeech':
        logger.log_info('Using LibriSpeech dataset')
        train_dataset = LibriSpeechDataset(
            root_dir=args.librispeech_root_dir,
            subset=args.librispeech_train_subset,
            sample_rate=args.sr,
            segment_length_secs=args.segment_duration_secs
        )
        valid_dataset = LibriSpeechDataset(
            root_dir=args.librispeech_root_dir,
            subset=args.librispeech_valid_subset,
            sample_rate=args.sr,
            segment_length_secs=args.segment_duration_secs # 验证时也可以用固定长度或完整长度
        )
        # args.sr = train_dataset.sample_rate # 确保 args.sr 与数据集一致，尽管我们已经通过参数设置了
    elif args.dataset_type == 'nsynth':
        logger.log_info('Using NSynth dataset')
        train_dataset = NSynthDataset(audio_dir=os.path.expanduser(args.nsynth_train_data_path))
        valid_dataset = NSynthDataset(audio_dir=os.path.expanduser(args.nsynth_valid_data_path))
        # args.sr = train_dataset.sr
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")


    if args.distributed:
        soundstream = torch.nn.SyncBatchNorm.convert_sync_batchnorm(soundstream)
        stft_disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stft_disc)
        msd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(msd)
        mpd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mpd)

    # torch.distributed.barrier()
    args.device = torch.device('cuda', args.local_rank)
    soundstream.to(args.device)
    stft_disc.to(args.device)
    msd.to(args.device)
    mpd.to(args.device)
    find_unused_parameters = False
    if args.distributed:
        soundstream = DDP(
            soundstream,
            device_ids=[args.local_rank],
            find_unused_parameters=find_unused_parameters
        )  # device_ids=[args.local_rank], output_device=args.local_rank
        stft_disc = DDP(stft_disc,
                        device_ids=[args.local_rank],
                        find_unused_parameters=find_unused_parameters)
        msd = DDP(msd,
                  device_ids=[args.local_rank],
                  find_unused_parameters=find_unused_parameters)
        mpd = DDP(mpd,
                  device_ids=[args.local_rank],
                  find_unused_parameters=find_unused_parameters)
    # 这里之后需要看下 sr 的问题，如果输入 wav 的 sr 和 `--sr` 不一致则会有问题
    # logger.log_info('Training set')
    # train_dataset = LibriSpeechDataset(root_dir="/path/to/LibriSpeech", sample_rate=args.sr, subset="train-clean-100")
    # # train_dataset = NSynthDataset(audio_dir=args.train_data_path)
    # logger.log_info('valid set')
    # valid_dataset = LibriSpeechDataset(root_dir="/path/to/LibriSpeech", sample_rate=args.sr, subset="dev-clean")
    # valid_dataset = NSynthDataset(audio_dir=args.valid_data_path)
    
    # args.sr = train_dataset.sr
    args.sr = train_dataset.sample_rate
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=True, shuffle=True)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset)
    else:
        train_sampler = None
        valid_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        sampler=valid_sampler)
    logger.log_info("Build optimizers and lr-schedulers")
    
    
    # --- 优化器和学习率调度器 ---
    # Encodec 论文使用 AdamW，beta1=0.9, beta2=0.95, weight_decay=0.1
    # 学习率初始为 1e-4, 预热1000步，然后余弦衰减
    # 这里的实现是 ExponentialLR，你可以根据需要调整或替换为更复杂的调度器
    optimizer_g = torch.optim.AdamW(
        soundstream.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1) # 更新超参数
    lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=0.999) # 你可能需要一个更复杂的学习率调度器

    optimizer_d_params = itertools.chain(
        stft_disc.parameters(),
        msd.parameters(), # msd 和 mpd 来自 soundstream.models
        mpd.parameters()
    )
    optimizer_d = torch.optim.AdamW(
        optimizer_d_params,
        lr=1e-4, # 判别器学习率
        betas=(0.9, 0.95), weight_decay=0.1)
    lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_d, gamma=0.999)

    if args.resume and args.resume_path is not None:
        # 确保 resume_path 指向包含 latest.pth 的目录
        latest_checkpoint_path = os.path.join(args.resume_path, 'latest.pth')
        if os.path.exists(latest_checkpoint_path):
            logger.log_info(f"Resuming from checkpoint: {latest_checkpoint_path}")
            latest_info = torch.load(latest_checkpoint_path, map_location='cpu') # 加载到 CPU 以避免设备问题
            args.st_epoch = latest_info['epoch'] + 1 # 从下一个 epoch 开始
            args.global_step = latest_info.get('global_step', 0) # 恢复 global_step

            # 加载模型状态
            # 处理 DDP 可能添加的 'module.' 前缀
            def load_state_dict_flexible(model, state_dict):
                # 如果 state_dict 来自 DDP 模型而当前模型不是，则移除 'module.'
                if not isinstance(model, DDP) and all(key.startswith('module.') for key in state_dict):
                    new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict)
                # 如果 state_dict 不是来自 DDP 模型而当前模型是，则添加 'module.' (较少见，通常 DDP 保存时已处理)
                elif isinstance(model, DDP) and not all(key.startswith('module.') for key in state_dict):
                    new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
            
            load_state_dict_flexible(soundstream, latest_info['soundstream'])
            load_state_dict_flexible(stft_disc, latest_info['stft_disc'])
            load_state_dict_flexible(mpd, latest_info['mpd'])
            load_state_dict_flexible(msd, latest_info['msd'])

            optimizer_g.load_state_dict(latest_info['optimizer_g'])
            lr_scheduler_g.load_state_dict(latest_info['lr_scheduler_g'])
            optimizer_d.load_state_dict(latest_info['optimizer_d'])
            lr_scheduler_d.load_state_dict(latest_info['lr_scheduler_d'])
            logger.log_info(f"Resumed from epoch {args.st_epoch-1}, global_step {args.global_step}")
        else:
            logger.log_warning(f"Resume path {latest_checkpoint_path} not found. Starting from scratch.")
            args.resume = False # 避免后续逻辑错误

    train(args, soundstream, stft_disc, msd, mpd, train_loader, valid_loader,
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger)


def train(args, soundstream, stft_disc, msd, mpd, train_loader, valid_loader,
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger):
    print('args ', args.global_rank)
    best_val_loss = float("inf")
    best_val_epoch = -1
    global_step = args.global_step 
    for epoch in range(args.st_epoch, args.N_EPOCHS + 1):
        soundstream.train()
        stft_disc.train()
        msd.train()
        mpd.train()
        train_loss_d = 0.0
        train_adv_g_loss = 0.0
        train_feat_loss = 0.0
        train_rec_loss = 0.0
        train_loss_g = 0.0
        train_commit_loss = 0.0
        k_iter = 0
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        for x in tqdm(train_loader):
            x = x.to(args.device)
            k_iter += 1
            global_step += 1  # record the global step
            for optimizer_idx in [0, 1]:  # we have two optimizer
                x_wav = get_input(x)
                G_x, commit_loss, last_layer = soundstream(x_wav)
                if optimizer_idx == 0:
                    # update generator
                    y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                    y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        x_wav.contiguous(), G_x.contiguous())
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                        x_wav.contiguous(), G_x.contiguous())
                    total_loss_g, rec_loss, adv_g_loss, feat_loss, d_weight = loss_g(
                        commit_loss,
                        x_wav,
                        G_x,
                        fmap_r,
                        fmap_gen,
                        y_disc_r,
                        y_disc_gen,
                        global_step,
                        y_df_hat_r,
                        y_df_hat_g,
                        y_ds_hat_r,
                        y_ds_hat_g,
                        fmap_f_r,
                        fmap_f_g,
                        fmap_s_r,
                        fmap_s_g,
                        last_layer=last_layer,
                        is_training=True,
                        args=args)
                    train_commit_loss += commit_loss.item()
                    train_loss_g += total_loss_g.item()
                    train_adv_g_loss += adv_g_loss.item()
                    train_feat_loss += feat_loss.item()
                    train_rec_loss += rec_loss.item()
                    optimizer_g.zero_grad()
                    total_loss_g.backward()
                    optimizer_g.step()
                else:
                    # update discriminator
                    y_disc_r_det, fmap_r_det = stft_disc(x.detach())
                    y_disc_gen_det, fmap_gen_det = stft_disc(G_x.detach())

                    # MPD
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        x.detach(), G_x.detach())
                    #MSD
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                        x.detach(), G_x.detach())

                    loss_d = loss_dis(
                        y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det,
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r,
                        y_ds_hat_g, fmap_s_r, fmap_s_g, global_step, args)
                    train_loss_d += loss_d.item()
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
            message = '<epoch:{:d}, iter:{:d}, total_loss_g:{:.4f}, adv_g_loss:{:.4f}, feat_loss:{:.4f}, rec_loss:{:.4f}, commit_loss:{:.4f}, loss_d:{:.4f}, d_weight: {:.4f}>'.format(
                epoch, k_iter,
                total_loss_g.item(),
                adv_g_loss.item(),
                feat_loss.item(),
                rec_loss.item(),
                commit_loss.item(), loss_d.item(), d_weight.item())
            if k_iter % args.print_freq == 0:
                logger.log_info(message)
        lr_scheduler_g.step()
        lr_scheduler_d.step()
        message = '<epoch:{:d}, <total_loss_g_train:{:.4f}, recon_loss_train:{:.4f}, adversarial_loss_train:{:.4f}, feature_loss_train:{:.4f}, commit_loss_train:{:.4f}>'.format(
            epoch, train_loss_g / len(train_loader), train_rec_loss /
            len(train_loader), train_adv_g_loss / len(train_loader),
            train_feat_loss / len(train_loader),
            train_commit_loss / len(train_loader))
        logger.log_info(message)
        with torch.no_grad():
            soundstream.eval()
            stft_disc.eval()
            mpd.eval()
            msd.eval()
            valid_loss_d = 0.0
            valid_loss_g = 0.0
            valid_commit_loss = 0.0
            valid_adv_g_loss = 0.0
            valid_feat_loss = 0.0
            valid_rec_loss = 0.0
            if args.distributed:
                valid_loader.sampler.set_epoch(epoch)
            for x in tqdm(valid_loader):
                x = x.to(args.device)
                for optimizer_idx in [0, 1]:
                    x_wav = get_input(x)
                    G_x, commit_loss, _ = soundstream(x_wav)
                    if optimizer_idx == 0:
                        valid_commit_loss += commit_loss
                        y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                        y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                            x_wav.contiguous(), G_x.contiguous())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                            x_wav.contiguous(), G_x.contiguous())

                        total_loss_g, adv_g_loss, feat_loss, rec_loss = criterion_g(
                            commit_loss,
                            x_wav,
                            G_x,
                            fmap_r,
                            fmap_gen,
                            y_disc_r,
                            y_disc_gen,
                            y_df_hat_r,
                            y_df_hat_g,
                            fmap_f_r,
                            fmap_f_g,
                            y_ds_hat_r,
                            y_ds_hat_g,
                            fmap_s_r,
                            fmap_s_g,
                            args=args)
                        valid_loss_g += total_loss_g.item()
                        valid_adv_g_loss += adv_g_loss.item()
                        valid_feat_loss += feat_loss.item()
                        valid_rec_loss += rec_loss.item()
                    else:
                        y_disc_r_det, fmap_r_det = stft_disc(
                            x_wav.contiguous().detach())
                        y_disc_gen_det, fmap_gen_det = stft_disc(
                            G_x.contiguous().detach())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                            x_wav.contiguous().detach(),
                            G_x.contiguous().detach())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                            x_wav.contiguous().detach(),
                            G_x.contiguous().detach())
                        loss_d = criterion_d(y_disc_r_det, y_disc_gen_det,
                                             fmap_r_det, fmap_gen_det,
                                             y_df_hat_r, y_df_hat_g, fmap_f_r,
                                             fmap_f_g, y_ds_hat_r, y_ds_hat_g,
                                             fmap_s_r, fmap_s_g)
                        valid_loss_d += loss_d.item()
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                best_model = soundstream.state_dict().copy()
                latest_model_soundstream = soundstream.state_dict().copy()
                latest_model_dis = stft_disc.state_dict().copy()
                latest_mpd = mpd.state_dict().copy()
                latest_msd = msd.state_dict().copy()
                if valid_rec_loss < best_val_loss:
                    best_val_loss = valid_rec_loss
                    best_val_epoch = epoch
                torch.save(best_model,
                           args.PATH + '/best_' + str(epoch) + '.pth')
                latest_save = {}
                latest_save['soundstream'] = latest_model_soundstream
                latest_save['stft_disc'] = latest_model_dis
                latest_save['mpd'] = latest_mpd
                latest_save['msd'] = latest_msd
                latest_save['epoch'] = epoch
                latest_save['optimizer_g'] = optimizer_g.state_dict()
                latest_save['optimizer_d'] = optimizer_d.state_dict()
                latest_save['lr_scheduler_g'] = lr_scheduler_g.state_dict()
                latest_save['lr_scheduler_d'] = lr_scheduler_d.state_dict()
                
                latest_save['global_step'] = global_step # 保存 global_step
                
                torch.save(latest_save, args.PATH + '/latest.pth')

            message = '<epoch:{:d}, total_loss_g_valid:{:.4f}, recon_loss_valid:{:.4f}, adversarial_loss_valid:{:.4f}, feature_loss_valid:{:.4f}, commit_loss_valid:{:.4f}, valid_loss_d:{:.4f}, best_epoch:{:d}>'.format(
                epoch, valid_loss_g / len(valid_loader), valid_rec_loss /
                len(valid_loader), valid_adv_g_loss / len(valid_loader),
                valid_feat_loss / len(valid_loader),
                valid_commit_loss / len(valid_loader),
                valid_loss_d / len(valid_loader), best_val_epoch)
            logger.log_info(message)


if __name__ == '__main__':
    main()
