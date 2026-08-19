#!/usr/bin/env python3
"""
CIFAR-10 training with Exact-SNN convolutional SNN.

Usage:
    python train_cifar10.py
    python train_cifar10.py --epochs 30 --lr 0.005 --batch-size 64
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from exact_snn.extended import SNNConvNet, SpikeNorm
from exact_snn.optim import AdamTorch
from exact_snn.losses import latency_cross_entropy


def encode_ttfs(images: torch.Tensor, t_max: float = 40.0) -> torch.Tensor:
    """Convert [0,1] images to TTFS spike times: brighter = earlier spike.

    Args:
        images: (B, C, H, W) pixel intensities in [0, 1].
        t_max: maximum simulation time.

    Returns:
        (B, C, H, W) spike times.
    """
    return t_max * (1.0 - images.clamp(0.01, 0.99)) + 0.1


def accuracy(net: SNNConvNet, t_images: torch.Tensor, y: torch.Tensor,
             batch_size: int = 64) -> float:
    """Compute classification accuracy on a dataset."""
    correct = 0
    total = 0
    n = t_images.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            t_batch = t_images[start:end]
            y_batch = y[start:end]
            t_out = net.forward(t_batch)
            pred = t_out.argmin(dim=0)
            correct += (pred == y_batch).sum().item()
            total += y_batch.shape[0]
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Conv SNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--t-max", type=float, default=40.0)
    parser.add_argument("--tm", type=float, default=15.0)
    parser.add_argument("--ts", type=float, default=4.0)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    print(f"Device: {dev}")

    print("Loading CIFAR-10...")
    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    train_images = train_set.data.float().permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(train_set.targets, dtype=torch.long)
    test_images = test_set.data.float().permute(0, 3, 1, 2) / 255.0
    test_labels = torch.tensor(test_set.targets, dtype=torch.long)

    if args.max_samples is not None:
        train_images = train_images[:args.max_samples]
        train_labels = train_labels[:args.max_samples]
        print(f"  Limited to {args.max_samples} training samples")

    train_images = train_images.to(dev)
    train_labels = train_labels.to(dev)
    test_images = test_images.to(dev)
    test_labels = test_labels.to(dev)

    t_test = encode_ttfs(test_images, t_max=args.t_max)
    print(f"  Train: {train_images.shape[0]}, Test: {test_images.shape[0]}")

    net = SNNConvNet(
        in_channels=3, h_w=32, n_classes=10,
        tm=args.tm, ts=args.ts, theta=args.theta,
        t_max=args.t_max, grid_pts=301,
        dtype=torch.float32, device=dev, seed=args.seed,
    )

    params = [net.conv1.W, net.conv2.W] + list(net.fc.W)
    opt = AdamTorch(params, lr=args.lr, clip=args.clip)

    n_params = sum(p.numel() for p in [net.conv1.W, net.conv2.W])
    n_params += sum(p.numel() for p in net.fc.W)
    print(f"Conv+FC parameters: {n_params}")
    print(f"SpikeNorm params: {sum(p.numel() for p in net.norm1.parameters()) + sum(p.numel() for p in net.norm2.parameters())}")

    N = train_images.shape[0]
    print(f"\n{'Epoch':>5} {'Loss':>8} {'Test%':>6} {'Time':>6}")
    print("-" * 35)

    t_start_all = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(N, device=dev)
        train_images_shuffled = train_images[perm]
        train_labels_shuffled = train_labels[perm]

        epoch_loss = 0.0
        n_batches = 0
        t_start_ep = time.time()

        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch_images = train_images_shuffled[start:end]
            batch_labels = train_labels_shuffled[start:end]

            loss, grads, grads_R, stats = net.loss_and_grads(batch_images, batch_labels)

            all_params = [net.conv1.W, net.conv2.W] + list(net.fc.W)
            all_grads = grads + [None] * len(net.fc.R) if net.fc.R else grads
            opt.step(all_params, all_grads)

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start_ep

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            test_acc = accuracy(net, t_test, test_labels, batch_size=32)
            print(f"{epoch:5d} {avg_loss:8.4f} {test_acc*100:5.1f}% {elapsed:5.1f}s")
        else:
            print(f"{epoch:5d} {avg_loss:8.4f} {'--':>6} {elapsed:5.1f}s")

    total_time = time.time() - t_start_all
    print(f"\nTotal training time: {total_time:.1f}s")

    final_acc = accuracy(net, t_test, test_labels, batch_size=32)
    print(f"Final test accuracy: {final_acc*100:.2f}%")


if __name__ == "__main__":
    main()
