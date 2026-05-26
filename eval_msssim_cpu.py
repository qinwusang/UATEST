import os
import csv
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

from net.network import SwinJSCC
from data.datasets import get_loader
from loss.distortion import MS_SSIM
from utils import seed_torch, logger_configuration, AverageMeter


parser = argparse.ArgumentParser(description="CPU MS-SSIM evaluation for SwinJSCC mismatch matrix")

parser.add_argument('--trainset', type=str, default='CIFAR10', choices=['CIFAR10', 'DIV2K'])
parser.add_argument('--testset', type=str, default='ffhq', choices=['kodak', 'CLIC21', 'ffhq'])
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'])
parser.add_argument('--model',
                    type=str,
                    default='SwinJSCC_w/_SA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'])
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'])
parser.add_argument('--C', type=str, default='32')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13')
parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--output', type=str, default='./mismatch_results/cifar10_msssim_cpu.csv')

args = parser.parse_args()


class config:
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False

    print_step = 100
    plot_step = 10000
    filename = "eval_msssim_cpu"
    workdir = './history/eval_msssim_cpu'
    log = workdir + '/Log_eval_msssim_cpu.log'
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    normalize = False
    learning_rate = 0.0001
    tot_epoch = 1

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/root/autodl-tmp/SwinJSCC/SwinJSCC/dataset/cifar-10-batches-py"
        test_data_dir = "/root/autodl-tmp/SwinJSCC/SwinJSCC/dataset/cifar-10-batches-py"
        batch_size = 128
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
    else:
        raise NotImplementedError("This script is currently written for CIFAR10 only.")


def load_weights(net, model_path):
    pretrained = torch.load(model_path, map_location="cuda:0")
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def main():
    seed_torch()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    config.logger = logger_configuration(config, save_log=False)

    net = SwinJSCC(args, config)
    load_weights(net, args.checkpoint)
    net = net.cuda()
    net.eval()

    train_loader, test_loader = get_loader(args, config)

    # Important: CPU MS-SSIM. Do NOT call .cuda() here.
    cpu_msssim = MS_SSIM(
        window_size=3,
        data_range=1.,
        levels=4,
        channel=3
    )
    cpu_msssim.eval()

    true_snr_list = [int(x) for x in args.multiple_snr.split(",")]
    hat_snr_list = [int(x) for x in args.multiple_snr.split(",")]
    channel_number = [int(x) for x in args.C.split(",")]

    results = []

    print("=================== CPU MS-SSIM Evaluation Starting ===================")

    with torch.no_grad():
        for snr_true in true_snr_list:
            for snr_hat in hat_snr_list:
                for rate in channel_number:

                    psnrs = AverageMeter()
                    msssim_vals = AverageMeter()
                    msssim_dbs = AverageMeter()

                    for batch_idx, (input, label) in enumerate(test_loader):
                        input_gpu = input.cuda()

                        recon_image, CBR, SNR, mse, loss_G = net(
                            input_gpu,
                            given_SNR=snr_true,
                            given_rate=rate,
                            mod_SNR=snr_hat
                        )

                        recon_gpu = recon_image.clamp(0., 1.)

                        if mse.item() > 0:
                            psnr = 10 * (torch.log(255. * 255. / mse) / math.log(10))
                            psnrs.update(psnr.item(), n=input.size(0))

                        # Move tensors to CPU before metric computation.
                        input_cpu = input.detach().cpu()
                        recon_cpu = recon_gpu.detach().cpu()

                        # Your MS_SSIM class returns 1 - MS_SSIM.
                        ms_ssim_loss = cpu_msssim(input_cpu, recon_cpu).mean().item()
                        ms_ssim_value = 1.0 - ms_ssim_loss
                        ms_ssim_value = max(0.0, min(1.0, ms_ssim_value))

                        ms_ssim_db = -10.0 * math.log10(max(1e-8, 1.0 - ms_ssim_value))

                        msssim_vals.update(ms_ssim_value, n=input.size(0))
                        msssim_dbs.update(ms_ssim_db, n=input.size(0))

                    row = {
                        "SNR_true": snr_true,
                        "SNR_hat": snr_hat,
                        "SNR_error": snr_hat - snr_true,
                        "C": rate,
                        "PSNR": psnrs.avg,
                        "MS_SSIM": msssim_vals.avg,
                        "MS_SSIM_dB": msssim_dbs.avg,
                    }
                    results.append(row)

                    msg = (
                        f"SNR_true={snr_true:2d} | "
                        f"SNR_hat={snr_hat:2d} | "
                        f"C={rate:3d} | "
                        f"PSNR={psnrs.avg:.4f} | "
                        f"MS-SSIM={msssim_vals.avg:.6f} | "
                        f"MS-SSIM(dB)={msssim_dbs.avg:.4f}"
                    )
                    print(msg)

    fieldnames = [
        "SNR_true",
        "SNR_hat",
        "SNR_error",
        "C",
        "PSNR",
        "MS_SSIM",
        "MS_SSIM_dB",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CPU MS-SSIM results to: {args.output}")
    print("=================== CPU MS-SSIM Evaluation Finished ===================")


if __name__ == "__main__":
    main()