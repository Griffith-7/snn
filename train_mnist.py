#!/usr/bin/env python3
"""
MNIST training with Exact-SNN: end-to-end TTFS classification.

Usage:
    python train_mnist.py                      # defaults
    python train_mnist.py --epochs 20 --lr 0.01
    python train_mnist.py --arch 784-256-128-10
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from exact_snn import TTFSNet, AdamTorch
from exact_snn.losses import latency_cross_entropy


def encode_ttfs(images: torch.Tensor, t_max: float = 40.0) -> torch.Tensor:
    """Convert [0,1] images to TTFS spike times: brighter = earlier spike.

    Args:
        images: (B, C, H, W) pixel intensities in [0, 1].
        t_max: maximum simulation time.

    Returns:
        (n_features, B) spike times, one per pixel.
    """
    B, C, H, W = images.shape
    flat = images.view(B, -1)          # (B, n_features)
    t_in = t_max * (1.0 - flat) + 0.1  # brighter -> smaller t
    return t_in.t()                     # (n_features, B)


def accuracy(net: TTFSNet, t_in: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> float:
    """Compute classification accuracy on a dataset."""
    correct = 0
    total = 0
    n = t_in.shape[1]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            t_batch = t_in[:, start:end]
            y_batch = y[start:end]
            t_out = net.forward(t_batch)
            # predicted class = neuron with earliest spike time
            pred = t_out.argmin(dim=0)
            correct += (pred == y_batch).sum().item()
            total += y_batch.shape[0]
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train TTFS SNN on MNIST")
    parser.add_argument("--arch", type=str, default="784-256-10",
                        help="Network architecture, dash-separated sizes")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--t-max", type=float, default=40.0)
    parser.add_argument("--tm", type=float, default=15.0, help="Membrane time constant")
    parser.add_argument("--ts", type=float, default=4.0, help="Synaptic time constant")
    parser.add_argument("--theta", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--w-scale", type=float, default=0.2, help="Weight init scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training set size for quick tests")
    args = parser.parse_args()

    # --- Device ---
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    print(f"Device: {dev}")

    # --- Architecture ---
    sizes = [int(x) for x in args.arch.split("-")]
    assert sizes[0] == 784, f"Input size must be 784 (MNIST), got {sizes[0]}"
    print(f"Architecture: {sizes}")

    # --- Data ---
    print("Loading MNIST...")
    train_set = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    test_set = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    # Convert to tensors
    train_images = train_set.data.float().unsqueeze(1) / 255.0  # (60000, 1, 28, 28)
    train_labels = train_set.targets
    test_images = test_set.data.float().unsqueeze(1) / 255.0
    test_labels = test_set.targets

    if args.max_samples is not None:
        train_images = train_images[:args.max_samples]
        train_labels = train_labels[:args.max_samples]
        print(f"  Limited to {args.max_samples} training samples")

    # Move to device
    train_images = train_images.to(dev)
    train_labels = train_labels.to(dev)
    test_images = test_images.to(dev)
    test_labels = test_labels.to(dev)

    # Pre-encode test set (no randomness in TTFS encoding)
    t_test = encode_ttfs(test_images, t_max=args.t_max)
    print(f"  Train: {train_images.shape[0]}, Test: {test_images.shape[0]}")

    # --- Model ---
    net = TTFSNet(
        sizes=sizes,
        tm=args.tm, ts=args.ts, theta=args.theta,
        t_max=args.t_max, w_scale=args.w_scale,
        seed=args.seed, dtype=torch.float64, dev=dev,
    )

    # Optimizer: collect all weight tensors + readout heads
    params = list(net.W) + list(net.R)
    opt = AdamTorch(params, lr=args.lr, clip=args.clip)

    n_params = sum(p.numel() for p in net.W)
    print(f"Parameters: {n_params}")

    # --- Training ---
    N = train_images.shape[0]
    print(f"\n{'Epoch':>5} {'Batch':>6} {'Loss':>8} {'Train%':>7} {'Test%':>6} {'Time':>6}")
    print("-" * 48)

    t_start_all = time.time()
    for epoch in range(1, args.epochs + 1):
        # Shuffle
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
            B = batch_images.shape[0]

            # Encode to spike times
            t_in = encode_ttfs(batch_images, t_max=args.t_max)

            # Forward + loss + backward
            loss, grads, t_out = net.loss_and_grads(t_in, batch_labels)

            # Update weights
            all_grads = list(grads) + [None] * len(net.R)
            opt.step(params, all_grads)

            epoch_loss += loss
            n_batches += 1

        # Epoch stats
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start_ep

        # Accuracy (every epoch for small datasets, every 5 for large)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            train_acc = accuracy(net, t_test, train_labels[:1])  # placeholder
            test_acc = accuracy(net, t_test, test_labels)
            print(f"{epoch:5d} {'--':>6} {avg_loss:8.4f} {'--':>7} {test_acc*100:5.1f}% {elapsed:5.1f}s")
        else:
            print(f"{epoch:5d} {'--':>6} {avg_loss:8.4f} {'--':>7} {'--':>6} {elapsed:5.1f}s")

    total_time = time.time() - t_start_all
    print(f"\nTotal training time: {total_time:.1f}s")

    # Final evaluation
    final_acc = accuracy(net, t_test, test_labels)
    print(f"Final test accuracy: {final_acc*100:.2f}%")


if __name__ == "__main__":
    main()
