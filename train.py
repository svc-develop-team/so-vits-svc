import os
import sys
import json
import argparse
import itertools
import math
import time
import logging

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

sys.path.append('../..')
import modules.commons as commons
import utils

from data_utils import DatasetConstructor

from models import (
    SynthesizerTrn,
    Discriminator
)

from modules.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
)
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch

torch.backends.cudnn.benchmark = True
global_step = 0
use_cuda = torch.cuda.is_available()
print("use_cuda, ", use_cuda)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def main():
    """Assume Single Node Multi GPUs Training Only"""

    hps = utils.get_hparams()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(hps.train.port)

    if (torch.cuda.is_available()):
        n_gpus = torch.cuda.device_count()
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    else:
        cpurun(0, 1, hps)


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps.train)
        logger.info(hps.data)
        logger.info(hps.model)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    dataset_constructor = DatasetConstructor(hps, num_replicas=n_gpus, rank=rank)

    train_loader = dataset_constructor.get_train_loader()
    if rank == 0:
        valid_loader = dataset_constructor.get_valid_loader()

    net_g = SynthesizerTrn(hps).cuda(rank)
    net_d = Discriminator(hps, hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               [train_loader, valid_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def cpurun(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps.train)
        logger.info(hps.data)
        logger.info(hps.model)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    torch.manual_seed(hps.train.seed)
    dataset_constructor = DatasetConstructor(hps, num_replicas=n_gpus, rank=rank)

    train_loader = dataset_constructor.get_train_loader()
    if rank == 0:
        valid_loader = dataset_constructor.get_valid_loader()

    net_g = SynthesizerTrn(hps)
    net_d = Discriminator(hps, hps.model.use_spectral_norm)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    skip_optimizer = True
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                           [train_loader, valid_loader], logger, [writer, writer_eval])

        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, data_dict in enumerate(train_loader):

        c = data_dict["c"]
        mel = data_dict["mel"]
        f0 = data_dict["f0"]
        uv = data_dict["uv"]
        wav = data_dict["wav"]
        spkid = data_dict["spkid"]

        c_lengths = data_dict["c_lengths"]
        mel_lengths = data_dict["mel_lengths"]
        wav_lengths = data_dict["wav_lengths"]
        f0_lengths = data_dict["f0_lengths"]

        # data
        if (use_cuda):
            c, c_lengths = c.cuda(rank, non_blocking=True), c_lengths.cuda(rank, non_blocking=True)
            mel, mel_lengths = mel.cuda(rank, non_blocking=True), mel_lengths.cuda(rank, non_blocking=True)
            wav, wav_lengths = wav.cuda(rank, non_blocking=True), wav_lengths.cuda(rank, non_blocking=True)
            f0, f0_lengths = f0.cuda(rank, non_blocking=True), f0_lengths.cuda(rank, non_blocking=True)
            spkid = spkid.cuda(rank, non_blocking=True)
            uv = uv.cuda(rank, non_blocking=True)

        # forward
        y_hat, ids_slice, LF0, y_ddsp, kl_div, predict_mel, mask, \
                pred_lf0, loss_f0, norm_f0 = net_g(c, c_lengths, f0,uv, mel, mel_lengths, spk_id=spkid)
        y_ddsp = y_ddsp.unsqueeze(1)

        # Discriminator
        y = commons.slice_segments(wav, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
        y_ddsp_mel = mel_spectrogram_torch(
            y_ddsp.squeeze(1),
            hps.data.n_fft,
            hps.data.acoustic_dim,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_size,
            hps.data.fmin,
            hps.data.fmax
        )

        y_logspec = torch.log(spectrogram_torch(
            y.squeeze(1),
            hps.data.n_fft,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_size
        ) + 1e-7)

        y_ddsp_logspec = torch.log(spectrogram_torch(
            y_ddsp.squeeze(1),
            hps.data.n_fft,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_size
        ) + 1e-7)

        y_mel = mel_spectrogram_torch(
            y.squeeze(1),
            hps.data.n_fft,
            hps.data.acoustic_dim,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_size,
            hps.data.fmin,
            hps.data.fmax
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.n_fft,
            hps.data.acoustic_dim,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_size,
            hps.data.fmin,
            hps.data.fmax
        )

        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

        optim_d.zero_grad()
        loss_disc_all.backward()
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        # loss
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

        loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
        loss_mel_dsp = F.l1_loss(y_mel, y_ddsp_mel) * 45
        loss_spec_dsp = F.l1_loss(y_logspec, y_ddsp_logspec) * 45

        loss_mel_am = F.mse_loss(mel * mask, predict_mel * mask)  # * 10

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)

        loss_fm = loss_fm / 2
        loss_gen = loss_gen / 2
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_mel_dsp + kl_div + loss_mel_am + loss_spec_dsp +\
                       loss_f0

        loss_gen_all = loss_gen_all / hps.train.accumulation_steps

        loss_gen_all.backward()
        if ((global_step + 1) % hps.train.accumulation_steps == 0):
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            optim_g.step()
            optim_g.zero_grad()

        if rank == 0:
            if (global_step + 1) % (hps.train.accumulation_steps * 10) == 0:
                print(["step&time&loss", global_step, time.asctime(time.localtime(time.time())), loss_gen_all])

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_gen_all, loss_mel]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/total": loss_gen_all,
                               "loss/mel": loss_mel,
                               "loss/adv": loss_gen,
                               "loss/fm": loss_fm,
                               "loss/mel_ddsp": loss_mel_dsp,
                               "loss/spec_ddsp": loss_spec_dsp,
                               "loss/mel_am": loss_mel_am,
                               "loss/kl_div": kl_div,
                               "loss/lf0": loss_f0,
                               "learning_rate": lr}
                image_dict = {
                    "train/lf0": utils.plot_data_to_numpy(LF0[0,0, :].cpu().numpy(), pred_lf0[0,0,  :].detach().cpu().numpy()),
                    "train/norm_lf0": utils.plot_data_to_numpy(LF0[0,0, :].cpu().numpy(), norm_f0[0,0,  :].detach().cpu().numpy()),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                    images=image_dict)

            if global_step % hps.train.eval_interval == 0:
                # logger.info(['All training params(G): ', utils.count_parameters(net_g), ' M'])
                # print('Sub training params(G): ', \
                #      'text_encoder: ', utils.count_parameters(net_g.module.text_encoder), ' M, ', \
                #      'decoder: ', utils.count_parameters(net_g.module.decoder), ' M, ', \
                #      'mel_decoder: ', utils.count_parameters(net_g.module.mel_decoder), ' M, ', \
                #      'dec: ', utils.count_parameters(net_g.module.dec), ' M, ', \
                #      'dec_harm: ', utils.count_parameters(net_g.module.dec_harm), ' M, ', \
                #      'dec_noise: ', utils.count_parameters(net_g.module.dec_noise), ' M, ', \
                #      'posterior: ', utils.count_parameters(net_g.module.posterior_encoder), ' M, ', \
                #     )

                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

                net_g.train()
        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(eval_loader):
            if batch_idx == 8:
                break
            c = data_dict["c"]
            mel = data_dict["mel"]
            f0 = data_dict["f0"]
            uv = data_dict["uv"]
            wav = data_dict["wav"]
            spkid = data_dict["spkid"]

            wav_lengths = data_dict["wav_lengths"]

            # data
            if (use_cuda):
                c = c.cuda(0)
                wav = wav.cuda(0)
                mel = mel.cuda(0)
                f0 = f0.cuda(0)
                uv = uv.cuda(0)
                spkid = spkid.cuda(0)
            # remove else
            c = c[:1]
            wav = wav[:1]
            mel = mel[:1]
            f0 = f0[:1]
            spkid = spkid[:1]
            if use_cuda:
                y_hat, y_harm, y_noise, _ = generator.module.infer(c, f0=f0,uv=uv, g=spkid)
            else:
                y_hat, y_harm, y_noise, _ = generator.infer(c, f0=f0,uv=uv, g=spkid)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.n_fft,
                hps.data.acoustic_dim,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_size,
                hps.data.fmin,
                hps.data.fmax
            )
            image_dict.update({
                f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            })
            audio_dict.update( {
                f"gen/audio_{batch_idx}": y_hat[0, :, :],
                f"gen/harm": y_harm[0, :, :],
                "gen/noise": y_noise[0, :, :]
            })
            # if global_step == 0:
            image_dict.update({f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
            audio_dict.update({f"gt/audio_{batch_idx}": wav[0, :, :wav_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
