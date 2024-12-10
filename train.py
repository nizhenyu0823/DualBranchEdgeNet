import sys
import os
import argparse
import time
import numpy as np
import glob
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from Data.data import get_loader, test_dataset
from Models import models
from Metrics import performance_metrics
from Metrics import losses


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx in range(test_loader.size):
        data, target, name = test_loader.load_data()
        data = data.to(device)
        target = np.asarray(target, np.float32)
        target /= (target.max() + 1e-8)
        output = model(data)
        # 确保 output 是 4D
        if output.dim() == 4:
            # 上采样处理
            output = F.interpolate(output, size=(target.shape[0], target.shape[1]), mode='bilinear',
                                   align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

        # 计算性能指标
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))


    train_dataloader = get_loader(args.train_img_dir,args.train_gt_dir, batchsize=args.batch_size, trainsize=352, num_workers=0)
    val_dataloader = test_dataset(args.test_img_dir, args.test_gt_dir, testsize=352)
    # train_dataloader, _, val_dataloader = data.get_loader(
    #     input_paths, target_paths, batchsize=args.batch_size, trainsize=352
    # )



    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    )


def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,

    ) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss
            )
            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch, perf
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                "Trained models/FCBFormer_" + args.dataset + ".pt",
            )
            prev_best_test = test_measure_mean


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, default= "Kvasir")
    parser.add_argument("--data-root", type=str, default="./dataset/Kvasir-SEG/",dest="root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    parser.add_argument('--train_img_dir', type=str, default='../data/polyp/TrainDataset/images/')
    parser.add_argument('--train_gt_dir', type=str, default='../data/polyp/TrainDataset/masks/')
    parser.add_argument('--test_img_dir', type=str, default='../data/polyp/TestDataset/CVC-300/images/')
    parser.add_argument('--test_gt_dir', type=str, default='../data/polyp/TestDataset/CVC-300/masks/')

    return parser.parse_args()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()