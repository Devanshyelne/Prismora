import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.utils import (
    ImageFolderDataset,
    get_transform,
    adaptive_instance_normalization,
    calc_mean_std,
)
from utils.models import VGGEncoder, Decoder

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an AdaIN neural style transfer decoder.")
    parser.add_argument("--content_dir", type=str,
                        default=str(PROJECT_ROOT / "content_data"))
    parser.add_argument("--style_dir", type=str,
                        default=str(PROJECT_ROOT / "style_data"))
    parser.add_argument("--vgg", type=str,
                        default=str(PROJECT_ROOT / "vgg_normalised.pth"))
    parser.add_argument("--experiment", type=str, default="big_dataset")
    parser.add_argument("--final_size", type=int, default=256)
    parser.add_argument("--content_size", type=int, default=512)
    parser.add_argument("--style_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of full passes through the paired batches.")
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--style_weight", type=float, default=10.0)
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save checkpoints every N epochs.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--optimizer_path", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Set when resuming old state_dict-only checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU acceleration
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA/MPS accelerator detected; training on CPU may be very slow.")

    root = PROJECT_ROOT / "experiment" / args.experiment
    root.mkdir(parents=True, exist_ok=True)

    content_transform = get_transform(args.content_size, True, args.final_size)
    style_transform = get_transform(args.style_size, True, args.final_size)
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)
    if len(content_dataset) < args.batch_size or len(style_dataset) < args.batch_size:
        raise ValueError(
            f"Not enough readable JPG/PNG images for batch_size={args.batch_size}. "
            f"Content={len(content_dataset)}, style={len(style_dataset)}."
        )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    content_loader = DataLoader(content_dataset, **loader_kwargs)
    style_loader = DataLoader(style_dataset, **loader_kwargs)
    # Pair batches by cycling the shorter loader. This makes every batch from the
    # larger set eligible to be seen within an epoch, unlike zip() which truncates.
    steps_per_epoch = max(len(content_loader), len(style_loader))
    print(f"Content images: {len(content_dataset):,}; batches: {len(content_loader):,}")
    print(f"Style images:   {len(style_dataset):,}; batches: {len(style_loader):,}")
    print(f"Steps per epoch: {steps_per_epoch:,}; epochs requested: {args.epochs}")

    encoder = VGGEncoder(args.vgg).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )
    start_epoch = args.start_epoch

    if args.resume:
        if not args.decoder_path:
            raise ValueError("--resume requires --decoder_path")
        decoder_state = torch.load(args.decoder_path, map_location=device)
        # Accept either plain decoder weights or a full training checkpoint.
        if isinstance(decoder_state, dict) and "decoder" in decoder_state:
            decoder.load_state_dict(decoder_state["decoder"])
            start_epoch = int(decoder_state.get("epoch", start_epoch))
            if "optimizer" in decoder_state:
                optimizer.load_state_dict(decoder_state["optimizer"])
            if "scheduler" in decoder_state:
                scheduler.load_state_dict(decoder_state["scheduler"])
        else:
            decoder.load_state_dict(decoder_state)
            if args.optimizer_path:
                optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))
        print(f"Resuming from epoch {start_epoch}.")
        # Old checkpoints did not save scheduler state; approximate its epoch.
        if start_epoch and not (isinstance(decoder_state, dict) and "scheduler" in decoder_state):
            for _ in range(start_epoch):
                scheduler.step()

    mse_loss = nn.MSELoss()
    content_iter = None
    style_iter = None

    for epoch in range(start_epoch, start_epoch + args.epochs):
        decoder.train()
        content_iter = iter(content_loader)
        style_iter = iter(style_loader)
        running_loss = running_closs = running_sloss = 0.0
        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}", unit="batch")

        for step in progress:
            try:
                content_batch = next(content_iter)
            except StopIteration:
                content_iter = iter(content_loader)
                content_batch = next(content_iter)
            try:
                style_batch = next(style_iter)
            except StopIteration:
                style_iter = iter(style_loader)
                style_batch = next(style_iter)

            # If the final partial batches differ in size, match batch dimensions.
            n = min(content_batch.size(0), style_batch.size(0))
            content_batch = content_batch[:n].to(device, non_blocking=True)
            style_batch = style_batch[:n].to(device, non_blocking=True)

            with torch.no_grad():
                c_feats = encoder(content_batch)
                s_feats = encoder(style_batch)
                target_feats = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            generated = decoder(target_feats)
            generated_feats = encoder(generated)

            loss_c = mse_loss(generated_feats[-1], target_feats) * args.content_weight
            loss_s = torch.zeros((), device=device)
            for gen_feat, style_feat in zip(generated_feats, s_feats):
                g_mean, g_std = calc_mean_std(gen_feat)
                s_mean, s_std = calc_mean_std(style_feat)
                loss_s = loss_s + mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()


            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                content=f"{loss_c.item():.3f}",
                style=f"{loss_s.item():.3f}",
            )

        
        denom = steps_per_epoch
        print(
            f"Epoch {epoch + 1}: loss={running_loss / denom:.5f}, "
            f"content={running_closs / denom:.5f}, style={running_sloss / denom:.5f}"
        )

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == start_epoch + args.epochs:
            # Save both a convenient weights file for app.py and a full resume checkpoint.
            torch.save(decoder.state_dict(), root / f"decoder_{epoch + 1}.pth")
            torch.save(optimizer.state_dict(), root / f"optimizer_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }, root / "checkpoint_latest.pth")

            # Save actual generated samples (content, style, generated), clamped for viewing.
            decoder.eval()
            with torch.no_grad():
                preview_c = content_batch[:min(2, len(content_batch))]
                preview_s = style_batch[:len(preview_c)]
                preview_t = adaptive_instance_normalization(
                    encoder(preview_c)[-1], encoder(preview_s)[-1]
                )
                preview_g = decoder(preview_t).clamp(0, 1)
                grid = torch.cat([preview_c.clamp(0, 1), preview_s.clamp(0, 1), preview_g], dim=0)
                save_image(grid, root / f"output_{epoch + 1}.png", nrow=len(preview_c))
            print(f"Saved checkpoint and preview in: {root.resolve()}")


if __name__ == "__main__":
    main()
