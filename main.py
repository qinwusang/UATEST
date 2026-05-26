import torch.optim as optim
from net.network import SwinJSCC
from data.datasets import get_loader
from utils import *

# torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision
import csv
import numpy as np


parser = argparse.ArgumentParser(description='SwinJSCC')
parser.add_argument('--training', action='store_true', help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K', choices=['CIFAR10', 'DIV2K'], help='train dataset name')
parser.add_argument('--testset', type=str, default='ffhq', choices=['kodak', 'CLIC21', 'ffhq'], help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'], help='evaluation metrics')
parser.add_argument('--model',
                    type=str,
                    default='SwinJSCC_w/_SAandRA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'],
                    help='SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet')
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'], help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=str, default='96', help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='10', help='random or fixed snr')
parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'], help='SwinJSCC model size')
parser.add_argument('--ua-train', action='store_true',
                    help='enable SNR-uncertainty-aware training')

parser.add_argument('--delta-train', type=float, default=0.0,
                    help='range of SNR estimation error during UA training')

parser.add_argument('--snr-min', type=float, default=1.0,
                    help='minimum clipped estimated SNR')

parser.add_argument('--snr-max', type=float, default=13.0,
                    help='maximum clipped estimated SNR')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint path for loading pretrained model')
args = parser.parse_args()


class config():
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 1000
    plot_step = 10000
    # filename = datetime.now().__str__()[:-7]
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        # train_data_dir = "/media/D/Dataset/CIFAR10/"
        # test_data_dir = "/media/D/Dataset/CIFAR10/"
        train_data_dir = "/root/autodl-tmp/SwinJSCC/SwinJSCC/dataset/cifar-10-batches-py"
        test_data_dir = "/root/autodl-tmp/SwinJSCC/SwinJSCC/dataset/cifar-10-batches-py"
        batch_size = 256
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 256],
            depths=[2, 4],
            num_heads=[4, 8],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128],
            depths=[4, 2],
            num_heads=[8, 4],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        base_path = "/media/D/Dataset/DIV2K/"
        if args.testset == 'kodak':
            test_data_dir = ["/media/D/Dataset/test/Kodak/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/D/Dataset/HR_Image_dataset/clic2021/test/"]
        elif args.testset == 'ffhq':
            test_data_dir = ["/media/D/yangke/SwinJSCC/data/ffhq/"]

        train_data_dir = [
            base_path + '/clic2020/**', base_path + '/clic2021/train', base_path + '/clic2021/valid', base_path + '/clic2022/val', base_path + '/DIV2K_train_HR', base_path + '/DIV2K_valid_HR'
        ]
        batch_size = 16
        downsample = 4
        if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 2, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 2, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == 'base':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 6, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 6, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == 'large':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 18, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 18, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def sample_ua_snr_pair(args, net):
    """
    snr_true: true channel SNR, used by the physical channel
    snr_hat: estimated SNR, used by Channel ModNet
    """
    snr_true = float(np.random.choice(net.multiple_snr))

    if args.ua_train and args.delta_train > 0:
        eps = np.random.uniform(-args.delta_train, args.delta_train)
        snr_hat = snr_true + eps
        snr_hat = float(np.clip(snr_hat, args.snr_min, args.snr_max))
    else:
        snr_hat = snr_true

    return snr_true, snr_hat


def train_one_epoch(args):
    net.train()

    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]

    global global_step

    if args.trainset == 'CIFAR10':
        data_iter = train_loader
    else:
        data_iter = train_loader

    for batch_idx, batch in enumerate(data_iter):
        start_time = time.time()
        global_step += 1

        if args.trainset == 'CIFAR10':
            input, label = batch
        else:
            input = batch

        input = input.cuda()

        # =========================
        # Core training call
        # =========================
        if args.ua_train:
            snr_true, snr_hat = sample_ua_snr_pair(args, net)

            recon_image, CBR, SNR, mse, loss_G = net(
                input,
                given_SNR=snr_true,
                given_rate=config.channel_number,
                mod_SNR=snr_hat
            )
        else:
            snr_true = None
            snr_hat = None

            # Original SwinJSCC training
            recon_image, CBR, SNR, mse, loss_G = net(input)

        loss = loss_G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)

        if mse.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            psnrs.update(psnr.item())

            # Disable MS-SSIM during training to avoid CUDA/NVRTC error.
            # Use PSNR as the training log metric.
            msssims.update(0.0)

        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0

            if args.ua_train:
                snr_log = f'SNR_true {SNR:.1f} | SNR_hat {snr_hat:.2f}'
            else:
                snr_log = f'SNR {snrs.val:.1f} ({snrs.avg:.1f})'

            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.val:.3f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                snr_log,
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM Disabled',
                f'Lr {cur_lr}',
            ]))

            logger.info(log)

            for i in metrics:
                i.clear()

    for i in metrics:
        i.clear()






# def train_one_epoch(args):
#     net.train()
#     elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
#     metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
#     global global_step
#     if args.trainset == 'CIFAR10':
#         for batch_idx, (input, label) in enumerate(train_loader):
#             start_time = time.time()
#             global_step += 1
#             input = input.cuda()
#             recon_image, CBR, SNR, mse, loss_G = net(
#                 input,
#                 given_SNR=snr_true,
#                 given_rate=config.channel_number,
#                 mod_SNR=snr_hat
#             )
#             loss = loss_G
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             elapsed.update(time.time() - start_time)
#             losses.update(loss.item())
#             cbrs.update(CBR)
#             snrs.update(SNR)
#             if mse.item() > 0:
#                 psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
#                 psnrs.update(psnr.item())
#                 # 先算出 SSIM 的张量，并在 GPU 上求均值，然后立刻 .item() 拿回 CPU 变成普通 Python 浮点数
#                 ssim_val = CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
#                 # 在纯 Python 环境下做减法，彻底不给 GPU 报错的机会
#                 msssim = 1.0 - ssim_val
#                 msssims.update(msssim)

#             if (global_step % config.print_step) == 0:
#                 process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
#                 log = (' | '.join([
#                     f'Epoch {epoch}',
#                     f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
#                     f'Time {elapsed.val:.3f}',
#                     f'Loss {losses.val:.3f} ({losses.avg:.3f})',
#                     f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
#                     f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
#                     f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
#                     f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
#                     f'Lr {cur_lr}',
#                 ]))
#                 logger.info(log)
#                 for i in metrics:
#                     i.clear()
#     else:
#         for batch_idx, input in enumerate(train_loader):
#             start_time = time.time()
#             global_step += 1
#             # input = input.cuda()
#             # recon_image, CBR, SNR, mse, loss_G = net(input)
#             # loss = loss_G

#             input = input.cuda()

#             if args.ua_train:
#                 snr_true, snr_hat = sample_ua_snr_pair(args, net)

#                 recon_image, CBR, SNR, mse, loss_G = net(
#                     input,
#                     given_SNR=snr_true,
#                     given_rate=config.channel_number,
#                     mod_SNR=snr_hat
#                 )
#             else:
#                 snr_true = None
#                 snr_hat = None
#                 recon_image, CBR, SNR, mse, loss_G = net(input)

#             loss = loss_G

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             elapsed.update(time.time() - start_time)
#             losses.update(loss.item())
#             cbrs.update(CBR)
#             snrs.update(SNR)
#             if mse.item() > 0:
#                 psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
#                 psnrs.update(psnr.item())
#                 msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
#                 msssims.update(msssim)

#             if (global_step % config.print_step) == 0:
#                 process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0


#                 if args.ua_train:
#                     snr_log = f'SNR_true {SNR:.1f} | SNR_hat {snr_hat:.2f}'
#                 else:
#                     snr_log = f'SNR {snrs.val:.1f} ({snrs.avg:.1f})'

#                 log = (' | '.join([
#                     f'Epoch {epoch}',
#                     f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
#                     f'Time {elapsed.val:.3f}',
#                     f'Loss {losses.val:.3f} ({losses.avg:.3f})',
#                     f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
#                     # f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
#                     snr_log,

#                     f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
#                     f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
#                     f'Lr {cur_lr}',
#                 ]))
#                 logger.info(log)
#                 for i in metrics:
#                     i.clear()
#     for i in metrics:
#         i.clear()


def test():
    import csv
    import math

    config.isTrain = False
    net.eval()

    # 1. Parse SNR and C lists from command line
    # Example:
    #   --multiple-snr 1,4,7,10,13
    #   --C 32
    true_snr_list = [int(x) for x in args.multiple_snr.split(",")]
    hat_snr_list = [int(x) for x in args.multiple_snr.split(",")]
    channel_number = [int(x) for x in args.C.split(",")]

    # 2. Output directories
    result_dir = "./mismatch_results"
    vis_root = "./mismatch_vis_cifar10" if args.trainset == "CIFAR10" else "./mismatch_vis_hr"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)

    csv_path = os.path.join(
        result_dir,
        f"{args.trainset}_{args.channel_type}_{args.model.replace('/', '-')}_C{args.C}_mismatch.csv"
    )

    results = []

    print("=================== Mismatch Experiment Starting ===================")
    logger.info("=================== Mismatch Experiment Starting ===================")
    logger.info(f"Dataset: {args.trainset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Channel: {args.channel_type}")
    logger.info(f"SNR_true list: {true_snr_list}")
    logger.info(f"SNR_hat list: {hat_snr_list}")
    logger.info(f"C list: {channel_number}")

    with torch.no_grad():
        for SNR_true in true_snr_list:
            for SNR_hat in hat_snr_list:
                for rate in channel_number:

                    elapsed = AverageMeter()
                    psnrs = AverageMeter()
                    ms_ssims = AverageMeter()
                    ms_ssim_dbs = AverageMeter()
                    cbrs = AverageMeter()
                    snrs = AverageMeter()

                    save_dir = os.path.join(
                        vis_root,
                        f"true{SNR_true}_hat{SNR_hat}_C{rate}"
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    if args.trainset == "CIFAR10":
                        for batch_idx, (input, label) in enumerate(test_loader):
                            start_time = time.time()
                            input = input.cuda()

                            recon_image, CBR, SNR, mse, loss_G = net(
                                input,
                                given_SNR=SNR_true,
                                given_rate=rate,
                                mod_SNR=SNR_hat
                            )

                            recon_clamped = recon_image.clamp(0., 1.)

                            # Save the first batch for visual comparison
                            if batch_idx == 0:
                                torchvision.utils.save_image(
                                    recon_clamped,
                                    os.path.join(save_dir, "batch0_recon_grid.png"),
                                    nrow=8
                                )

                                # Save original image grid only once per true/hat/rate setting
                                torchvision.utils.save_image(
                                    input,
                                    os.path.join(save_dir, "batch0_original_grid.png"),
                                    nrow=8
                                )

                            elapsed.update(time.time() - start_time)
                            cbrs.update(CBR)
                            snrs.update(SNR)

                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / math.log(10))
                                psnrs.update(psnr.item())

                            # CalcuSSIM returns 1 - MS-SSIM in your distortion.py
                            try:
                                ms_ssim_loss = CalcuSSIM(input, recon_clamped).mean().item()
                                ms_ssim_value = 1.0 - ms_ssim_loss
                                ms_ssim_value = max(0.0, min(1.0, ms_ssim_value))
                                ms_ssims.update(ms_ssim_value)

                                # MS-SSIM(dB) = -10 log10(1 - MS-SSIM)
                                ms_ssim_db = -10.0 * math.log10(max(1e-8, 1.0 - ms_ssim_value))
                                ms_ssim_dbs.update(ms_ssim_db)
                            except Exception as e:
                                # Do not kill the whole PSNR experiment if MS-SSIM fails
                                if batch_idx == 0:
                                    warn_msg = f"MS-SSIM calculation failed: {repr(e)}"
                                    print(warn_msg)
                                    logger.info(warn_msg)

                    else:
                        for batch_idx, batch in enumerate(test_loader):
                            # HR loader normally returns (input, names)
                            input, names = batch
                            start_time = time.time()
                            input = input.cuda()

                            recon_image, CBR, SNR, mse, loss_G = net(
                                input,
                                given_SNR=SNR_true,
                                given_rate=rate,
                                mod_SNR=SNR_hat
                            )

                            recon_clamped = recon_image.clamp(0., 1.)

                            # Save the first few HR reconstructed images
                            if batch_idx < 3:
                                if isinstance(names, (list, tuple)):
                                    name = str(names[0])
                                else:
                                    name = f"img{batch_idx}"

                                if not (name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")):
                                    name = name + ".png"

                                torchvision.utils.save_image(
                                    recon_clamped,
                                    os.path.join(save_dir, f"recon_{name}")
                                )
                                torchvision.utils.save_image(
                                    input,
                                    os.path.join(save_dir, f"original_{name}")
                                )

                            elapsed.update(time.time() - start_time)
                            cbrs.update(CBR)
                            snrs.update(SNR)

                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / math.log(10))
                                psnrs.update(psnr.item())

                            try:
                                ms_ssim_loss = CalcuSSIM(input, recon_clamped).mean().item()
                                ms_ssim_value = 1.0 - ms_ssim_loss
                                ms_ssim_value = max(0.0, min(1.0, ms_ssim_value))
                                ms_ssims.update(ms_ssim_value)

                                ms_ssim_db = -10.0 * math.log10(max(1e-8, 1.0 - ms_ssim_value))
                                ms_ssim_dbs.update(ms_ssim_db)
                            except Exception as e:
                                if batch_idx == 0:
                                    warn_msg = f"MS-SSIM calculation failed: {repr(e)}"
                                    print(warn_msg)
                                    logger.info(warn_msg)

                    row = {
                        "trainset": args.trainset,
                        "model": args.model,
                        "channel_type": args.channel_type,
                        "C": rate,
                        "SNR_true": SNR_true,
                        "SNR_hat": SNR_hat,
                        "SNR_error": SNR_hat - SNR_true,
                        "CBR": cbrs.avg,
                        "PSNR": psnrs.avg,
                        "MS_SSIM": ms_ssims.avg if ms_ssims.count > 0 else "",
                        "MS_SSIM_dB": ms_ssim_dbs.avg if ms_ssim_dbs.count > 0 else "",
                        "avg_time": elapsed.avg,
                    }
                    results.append(row)

                    log_msg = (
                        f"SNR_true: {SNR_true:2d} dB | "
                        f"SNR_hat: {SNR_hat:2d} dB | "
                        f"error: {SNR_hat - SNR_true:+3d} dB | "
                        f"C: {rate:3d} | "
                        f"CBR: {cbrs.avg:.4f} | "
                        f"PSNR: {psnrs.avg:.4f} | "
                    )

                    if ms_ssims.count > 0:
                        log_msg += (
                            f"MS-SSIM: {ms_ssims.avg:.6f} | "
                            f"MS-SSIM(dB): {ms_ssim_dbs.avg:.4f}"
                        )
                    else:
                        log_msg += "MS-SSIM: N/A"

                    print(log_msg)
                    logger.info(log_msg)

    # 3. Save CSV
    fieldnames = [
        "trainset",
        "model",
        "channel_type",
        "C",
        "SNR_true",
        "SNR_hat",
        "SNR_error",
        "CBR",
        "PSNR",
        "MS_SSIM",
        "MS_SSIM_dB",
        "avg_time",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved mismatch results to: {csv_path}")
    logger.info(f"Saved mismatch results to: {csv_path}")

    # 4. Optional: print PSNR matrix for quick copy into notes
    print("=================== PSNR Matrix ===================")
    logger.info("=================== PSNR Matrix ===================")

    for rate in channel_number:
        print(f"C = {rate}")
        logger.info(f"C = {rate}")

        header = "true\\hat" + "".join([f"\t{h}" for h in hat_snr_list])
        print(header)
        logger.info(header)

        for SNR_true in true_snr_list:
            line = f"{SNR_true}"
            for SNR_hat in hat_snr_list:
                matched = [
                    r for r in results
                    if r["C"] == rate
                    and r["SNR_true"] == SNR_true
                    and r["SNR_hat"] == SNR_hat
                ]
                if len(matched) == 1:
                    line += f"\t{matched[0]['PSNR']:.4f}"
                else:
                    line += "\tNA"
            print(line)
            logger.info(line)

    print("=================== Mismatch Experiment Finished ===================")
    logger.info("=================== Mismatch Experiment Finished ===================")

    return

# def test():
#     config.isTrain = False
#     net.eval()
#     elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
#     metrics = [elapsed, psnrs, msssims, snrs, cbrs]
#     multiple_snr = args.multiple_snr.split(",")
#     for i in range(len(multiple_snr)):
#         multiple_snr[i] = int(multiple_snr[i])
#     channel_number = args.C.split(",")
#     for i in range(len(channel_number)):
#         channel_number[i] = int(channel_number[i])
#     results_snr = np.zeros((len(multiple_snr), len(channel_number)))
#     results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
#     results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
#     results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
#     for i, SNR in enumerate(multiple_snr):
#         for j, rate in enumerate(channel_number):
#             with torch.no_grad():
#                 if args.trainset == 'CIFAR10':
#                     for batch_idx, (input, label) in enumerate(test_loader):
#                         start_time = time.time()
#                         input = input.cuda()
#                         recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)

#                         elapsed.update(time.time() - start_time)
#                         cbrs.update(CBR)
#                         snrs.update(SNR)
#                         if mse.item() > 0:
#                             psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
#                             psnrs.update(psnr.item())
#                             msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
#                             msssims.update(msssim)

#                         log = (' | '.join([
#                             f'Time {elapsed.val:.3f}',
#                             f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
#                             f'SNR {snrs.val:.1f}',
#                             f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
#                             f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
#                             f'Lr {cur_lr}',
#                         ]))
#                         logger.info(log)
#                 else:
#                     for batch_idx, batch in enumerate(test_loader):
#                         input, names = batch
#                         start_time = time.time()
#                         input = input.cuda()
#                         recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)
#                         # torchvision.utils.save_image(recon_image, os.path.join("/media/D/yangke/SwinJSCC/data/", f"recon/{names[0]}"))
#                         elapsed.update(time.time() - start_time)
#                         cbrs.update(CBR)
#                         snrs.update(SNR)
#                         if mse.item() > 0:
#                             psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
#                             psnrs.update(psnr.item())
#                             msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
#                             msssims.update(msssim)
#                             MSSSIM = -10 * np.math.log10(1 - msssim)
#                         log = (' | '.join([
#                             f'Time {elapsed.val:.3f}',
#                             f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
#                             f'SNR {snrs.val:.1f}',
#                             f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
#                             f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
#                             f'Lr {cur_lr}',
#                         ]))
#                         logger.info(log)
#             results_snr[i, j] = snrs.avg
#             results_cbr[i, j] = cbrs.avg
#             results_psnr[i, j] = psnrs.avg
#             results_msssim[i, j] = msssims.avg
#             for t in metrics:
#                 t.clear()

    # print("SNR: {}".format(results_snr.tolist()))
    # print("CBR: {}".format(results_cbr.tolist()))
    # print("PSNR: {}".format(results_psnr.tolist()))
    # print("MS-SSIM: {}".format(results_msssim.tolist()))
    # print("Finish Test!")


# if __name__ == '__main__':
#     seed_torch()
#     logger = logger_configuration(config, save_log=False)
#     logger.info(config.__dict__)
#     torch.manual_seed(seed=config.seed)
#     net = SwinJSCC(args, config)
#     model_path = "./checkpoint/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
#     load_weights(model_path)
#     net = net.cuda()


if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=False)
    os.makedirs(config.workdir, exist_ok=True)
    os.makedirs(config.samples, exist_ok=True)
    os.makedirs(config.models, exist_ok=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config)
    
    # 修改为你指定的 CIFAR10 预训练模型
    # model_path = "/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model" 
    # load_weights(model_path)

    default_model_path = "/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"

    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = default_model_path

    load_weights(model_path)
    logger.info(f"Loaded checkpoint from: {model_path}")
    net = net.cuda()
    # ...后续保持不变...
    



    # ...后续代码保持不变...
    # model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    model_params = [{'params': net.parameters(), 'lr': config.learning_rate}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            if (epoch + 1) % config.save_model_freq == 0:
                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
                test()
    else:
        test()
