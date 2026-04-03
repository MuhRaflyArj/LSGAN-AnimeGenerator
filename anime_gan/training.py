"""Training loop utilities for Vanilla GAN and LSGAN experiments."""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .checkpointing import save_best_checkpoint, save_final_artifacts
from .models import Discriminator, Generator, weight_init_normal
from .visualization import sample_images


def create_training_components(cfg, device, gan_type):
    """Create the models, losses, optimizers, and schedulers for training."""

    if gan_type not in {"vanilla", "lsgan"}:
        raise ValueError("gan_type must be 'vanilla' or 'lsgan'")

    generator = Generator(latent_dim=cfg.latent_dim, channels=cfg.channels).to(device)
    discriminator = Discriminator(channels=cfg.channels).to(device)

    generator.apply(weight_init_normal)
    discriminator.apply(weight_init_normal)

    if gan_type == "vanilla":
        adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    else:
        adversarial_loss = nn.MSELoss().to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=cfg.lr_g, betas=(cfg.b1, cfg.b2)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.b1, cfg.b2)
    )

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode="max", factor=0.5, patience=8, threshold=1e-3, min_lr=1e-6
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode="min", factor=0.5, patience=8, threshold=1e-3, min_lr=1e-6
    )

    return {
        "generator": generator,
        "discriminator": discriminator,
        "adversarial_loss": adversarial_loss,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "scheduler_g": scheduler_g,
        "scheduler_d": scheduler_d,
    }


def train_gan(cfg, dataloader, device, gan_type="lsgan", run_name=None):
    """Train one GAN variant and save checkpoints and history."""

    if run_name is None:
        run_name = gan_type

    components = create_training_components(cfg=cfg, device=device, gan_type=gan_type)
    generator = components["generator"]
    discriminator = components["discriminator"]
    adversarial_loss = components["adversarial_loss"]
    optimizer_g = components["optimizer_g"]
    optimizer_d = components["optimizer_d"]
    scheduler_g = components["scheduler_g"]
    scheduler_d = components["scheduler_d"]

    global_step = 0
    g_losses, d_losses = [], []
    d_real_scores, d_fake_scores = [], []

    epoch_g_losses, epoch_d_losses = [], []
    epoch_d_real_scores, epoch_d_fake_scores = [], []
    epoch_margins = []
    epoch_pixel_mean, epoch_pixel_std = [], []
    epoch_lr_g, epoch_lr_d = [], []

    best_margin = -float("inf")
    best_epoch = -1

    fixed_noise = torch.randn(cfg.fixed_sample_size, cfg.latent_dim, device=device)

    for epoch in range(cfg.n_epochs):
        epoch_bar = tqdm(
            dataloader,
            desc=f"{run_name} | Epoch {epoch + 1}/{cfg.n_epochs}",
            leave=False,
        )

        g_epoch_vals, d_epoch_vals = [], []
        d_real_epoch_vals, d_fake_epoch_vals = [], []

        for imgs in epoch_bar:
            real_imgs = imgs.to(device, non_blocking=True)
            batch_size_now = real_imgs.size(0)

            real_targets = torch.full(
                (batch_size_now, 1), cfg.real_label, device=device
            )
            fake_targets = torch.full(
                (batch_size_now, 1), cfg.fake_label, device=device
            )

            if cfg.label_noise > 0:
                real_targets = (
                    real_targets + cfg.label_noise * torch.rand_like(real_targets)
                ).clamp(0.8, 1.0)
                fake_targets = (
                    fake_targets + cfg.label_noise * torch.rand_like(fake_targets)
                ).clamp(0.0, 0.2)

            optimizer_d.zero_grad(set_to_none=True)
            z = torch.randn((batch_size_now, cfg.latent_dim), device=device)
            gen_imgs = generator(z).detach()

            pred_real = discriminator(real_imgs)
            pred_fake = discriminator(gen_imgs)

            real_loss = adversarial_loss(pred_real, real_targets)
            fake_loss = adversarial_loss(pred_fake, fake_targets)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.grad_clip)
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            z = torch.randn((batch_size_now, cfg.latent_dim), device=device)
            gen_imgs = generator(z)
            g_targets = torch.full((batch_size_now, 1), cfg.real_label, device=device)

            pred_fake_for_g = discriminator(gen_imgs)
            g_loss = adversarial_loss(pred_fake_for_g, g_targets)

            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip)
            optimizer_g.step()

            g_val = g_loss.item()
            d_val = d_loss.item()
            d_real_val = pred_real.detach().mean().item()
            d_fake_val = pred_fake_for_g.detach().mean().item()
            margin = d_real_val - d_fake_val

            g_losses.append(g_val)
            d_losses.append(d_val)
            d_real_scores.append(d_real_val)
            d_fake_scores.append(d_fake_val)

            g_epoch_vals.append(g_val)
            d_epoch_vals.append(d_val)
            d_real_epoch_vals.append(d_real_val)
            d_fake_epoch_vals.append(d_fake_val)

            if global_step % cfg.sample_interval == 0:
                sample_images(
                    generator=generator,
                    device=device,
                    latent_dim=cfg.latent_dim,
                    tag=f"{global_step:07d}",
                    n_samples=25,
                    show=False,
                    out_prefix=f"{run_name}_step",
                    images_dir=str(cfg.images_dir),
                )

            epoch_bar.set_postfix(
                d_loss=f"{d_val:.6f}",
                g_loss=f"{g_val:.6f}",
                d_real=f"{d_real_val:.4f}",
                d_fake=f"{d_fake_val:.4f}",
                margin=f"{margin:.4f}",
                step=global_step,
            )

            global_step += 1

        mean_g = float(np.mean(g_epoch_vals))
        mean_d = float(np.mean(d_epoch_vals))
        mean_d_real = float(np.mean(d_real_epoch_vals))
        mean_d_fake = float(np.mean(d_fake_epoch_vals))
        mean_margin = mean_d_real - mean_d_fake

        sample_mean, sample_std = sample_images(
            generator=generator,
            device=device,
            latent_dim=cfg.latent_dim,
            tag=f"{epoch + 1:03d}",
            n_samples=25,
            show=False,
            out_prefix=f"{run_name}_epoch",
            images_dir=str(cfg.images_dir),
        )

        epoch_g_losses.append(mean_g)
        epoch_d_losses.append(mean_d)
        epoch_d_real_scores.append(mean_d_real)
        epoch_d_fake_scores.append(mean_d_fake)
        epoch_margins.append(mean_margin)
        epoch_pixel_mean.append(sample_mean)
        epoch_pixel_std.append(sample_std)
        epoch_lr_g.append(optimizer_g.param_groups[0]["lr"])
        epoch_lr_d.append(optimizer_d.param_groups[0]["lr"])

        if mean_margin > best_margin:
            best_margin = mean_margin
            best_epoch = epoch + 1
            save_best_checkpoint(
                checkpoints_dir=cfg.checkpoints_dir,
                run_name=run_name,
                epoch=best_epoch,
                mean_margin=best_margin,
                mean_g=mean_g,
                mean_d=mean_d,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
            )
            tqdm.write(
                f"{run_name} saved best checkpoint at epoch {best_epoch} (margin={best_margin:.4f})"
            )

        scheduler_g.step(mean_margin)
        scheduler_d.step(mean_d)

        tqdm.write(
            f"{run_name} | Epoch {epoch + 1}/{cfg.n_epochs} | "
            f"D: {mean_d:.6f} | G: {mean_g:.6f} | "
            f"D(real): {mean_d_real:.3f} | D(fake): {mean_d_fake:.3f} | "
            f"Margin: {mean_margin:.4f} | Pixel std: {sample_std:.4f}"
        )

    history = {
        "gan_type": gan_type,
        "run_name": run_name,
        "global_step": global_step,
        "g_losses": g_losses,
        "d_losses": d_losses,
        "d_real_scores": d_real_scores,
        "d_fake_scores": d_fake_scores,
        "epoch_g_losses": epoch_g_losses,
        "epoch_d_losses": epoch_d_losses,
        "epoch_d_real_scores": epoch_d_real_scores,
        "epoch_d_fake_scores": epoch_d_fake_scores,
        "epoch_margins": epoch_margins,
        "epoch_pixel_mean": epoch_pixel_mean,
        "epoch_pixel_std": epoch_pixel_std,
        "epoch_lr_g": epoch_lr_g,
        "epoch_lr_d": epoch_lr_d,
        "best_margin": best_margin,
        "best_epoch": best_epoch,
    }

    save_final_artifacts(
        checkpoints_dir=cfg.checkpoints_dir,
        run_name=run_name,
        generator=generator,
        discriminator=discriminator,
        history=history,
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_margin": best_margin,
        "fixed_noise": fixed_noise,
        "generator": generator,
        "discriminator": discriminator,
    }


def train_lsgan(cfg, dataloader, device, run_name="lsgan"):
    """Train the least-squares GAN variant."""

    return train_gan(
        cfg=cfg,
        dataloader=dataloader,
        device=device,
        gan_type="lsgan",
        run_name=run_name,
    )
