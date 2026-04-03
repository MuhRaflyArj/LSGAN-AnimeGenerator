"""Training loop utilities for Vanilla GAN and LSGAN experiments."""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .checkpointing import save_best_checkpoint, save_final_artifacts
from .models import Discriminator, Generator, weight_init_normal
from .visualization import sample_images


def _init_history_buffers():
    """Create empty step-level and epoch-level training buffers."""

    return {
        "g_losses": [],
        "d_losses": [],
        "d_real_scores": [],
        "d_fake_scores": [],
        "epoch_g_losses": [],
        "epoch_d_losses": [],
        "epoch_d_real_scores": [],
        "epoch_d_fake_scores": [],
        "epoch_margins": [],
        "epoch_pixel_mean": [],
        "epoch_pixel_std": [],
        "epoch_lr_g": [],
        "epoch_lr_d": [],
    }


def _make_targets(cfg, batch_size_now, device):
    """Create real and fake label targets, optionally with label noise."""

    real_targets = torch.full((batch_size_now, 1), cfg.real_label, device=device)
    fake_targets = torch.full((batch_size_now, 1), cfg.fake_label, device=device)

    if cfg.label_noise > 0:
        real_targets = (
            real_targets + cfg.label_noise * torch.rand_like(real_targets)
        ).clamp(0.8, 1.0)
        fake_targets = (
            fake_targets + cfg.label_noise * torch.rand_like(fake_targets)
        ).clamp(0.0, 0.2)

    return real_targets, fake_targets


def _train_discriminator_step(
    cfg,
    discriminator,
    generator,
    adversarial_loss,
    optimizer_d,
    real_imgs,
    real_targets,
    fake_targets,
):
    """Run one discriminator optimization step and return predictions and loss."""

    batch_size_now = real_imgs.size(0)
    optimizer_d.zero_grad(set_to_none=True)
    z = torch.randn((batch_size_now, cfg.latent_dim), device=real_imgs.device)
    gen_imgs = generator(z).detach()

    pred_real = discriminator(real_imgs)
    pred_fake = discriminator(gen_imgs)

    real_loss = adversarial_loss(pred_real, real_targets)
    fake_loss = adversarial_loss(pred_fake, fake_targets)
    d_loss = 0.5 * (real_loss + fake_loss)

    d_loss.backward()
    nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.grad_clip)
    optimizer_d.step()

    return pred_real, d_loss


def _train_generator_step(
    cfg, discriminator, generator, adversarial_loss, optimizer_g, batch_size_now, device
):
    """Run one generator optimization step and return fake predictions and loss."""

    optimizer_g.zero_grad(set_to_none=True)
    z = torch.randn((batch_size_now, cfg.latent_dim), device=device)
    gen_imgs = generator(z)
    g_targets = torch.full((batch_size_now, 1), cfg.real_label, device=device)

    pred_fake_for_g = discriminator(gen_imgs)
    g_loss = adversarial_loss(pred_fake_for_g, g_targets)

    g_loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip)
    optimizer_g.step()

    return pred_fake_for_g, g_loss


def _compute_step_metrics(g_loss, d_loss, pred_real, pred_fake_for_g):
    """Compute scalar metrics from a single training step."""

    g_val = g_loss.item()
    d_val = d_loss.item()
    d_real_val = pred_real.detach().mean().item()
    d_fake_val = pred_fake_for_g.detach().mean().item()

    return {
        "g_val": g_val,
        "d_val": d_val,
        "d_real_val": d_real_val,
        "d_fake_val": d_fake_val,
        "margin": d_real_val - d_fake_val,
    }


def _compute_epoch_averages(
    g_epoch_vals, d_epoch_vals, d_real_epoch_vals, d_fake_epoch_vals
):
    """Aggregate epoch-level means from per-step metric lists."""

    mean_g = float(np.mean(g_epoch_vals))
    mean_d = float(np.mean(d_epoch_vals))
    mean_d_real = float(np.mean(d_real_epoch_vals))
    mean_d_fake = float(np.mean(d_fake_epoch_vals))

    return {
        "mean_g": mean_g,
        "mean_d": mean_d,
        "mean_d_real": mean_d_real,
        "mean_d_fake": mean_d_fake,
        "mean_margin": mean_d_real - mean_d_fake,
    }


def _update_best_checkpoint(cfg, run_name, epoch, epoch_stats, state):
    """Save a checkpoint when margin improves and update best state."""

    if epoch_stats["mean_margin"] <= state["best_margin"]:
        return

    state["best_margin"] = epoch_stats["mean_margin"]
    state["best_epoch"] = epoch + 1
    save_best_checkpoint(
        checkpoints_dir=cfg.checkpoints_dir,
        run_name=run_name,
        epoch=state["best_epoch"],
        mean_margin=state["best_margin"],
        mean_g=epoch_stats["mean_g"],
        mean_d=epoch_stats["mean_d"],
        generator=state["generator"],
        discriminator=state["discriminator"],
        optimizer_g=state["optimizer_g"],
        optimizer_d=state["optimizer_d"],
    )
    tqdm.write(
        f"{run_name} saved best checkpoint at epoch {state['best_epoch']} "
        f"(margin={state['best_margin']:.4f})"
    )


def _build_training_history(
    gan_type, run_name, global_step, buffers, best_margin, best_epoch
):
    """Build the final history dictionary returned by training."""

    return {
        "gan_type": gan_type,
        "run_name": run_name,
        "global_step": global_step,
        "g_losses": buffers["g_losses"],
        "d_losses": buffers["d_losses"],
        "d_real_scores": buffers["d_real_scores"],
        "d_fake_scores": buffers["d_fake_scores"],
        "epoch_g_losses": buffers["epoch_g_losses"],
        "epoch_d_losses": buffers["epoch_d_losses"],
        "epoch_d_real_scores": buffers["epoch_d_real_scores"],
        "epoch_d_fake_scores": buffers["epoch_d_fake_scores"],
        "epoch_margins": buffers["epoch_margins"],
        "epoch_pixel_mean": buffers["epoch_pixel_mean"],
        "epoch_pixel_std": buffers["epoch_pixel_std"],
        "epoch_lr_g": buffers["epoch_lr_g"],
        "epoch_lr_d": buffers["epoch_lr_d"],
        "best_margin": best_margin,
        "best_epoch": best_epoch,
    }


def _run_training_loop(
    cfg,
    dataloader,
    device,
    run_name,
    generator,
    discriminator,
    adversarial_loss,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    buffers,
    state,
):
    """Run the epoch and batch training loop and return the final global step."""

    global_step = 0
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

            real_targets, fake_targets = _make_targets(
                cfg=cfg, batch_size_now=batch_size_now, device=device
            )
            pred_real, d_loss = _train_discriminator_step(
                cfg=cfg,
                discriminator=discriminator,
                generator=generator,
                adversarial_loss=adversarial_loss,
                optimizer_d=optimizer_d,
                real_imgs=real_imgs,
                real_targets=real_targets,
                fake_targets=fake_targets,
            )
            pred_fake_for_g, g_loss = _train_generator_step(
                cfg=cfg,
                discriminator=discriminator,
                generator=generator,
                adversarial_loss=adversarial_loss,
                optimizer_g=optimizer_g,
                batch_size_now=batch_size_now,
                device=device,
            )
            step_metrics = _compute_step_metrics(
                g_loss=g_loss,
                d_loss=d_loss,
                pred_real=pred_real,
                pred_fake_for_g=pred_fake_for_g,
            )

            buffers["g_losses"].append(step_metrics["g_val"])
            buffers["d_losses"].append(step_metrics["d_val"])
            buffers["d_real_scores"].append(step_metrics["d_real_val"])
            buffers["d_fake_scores"].append(step_metrics["d_fake_val"])

            g_epoch_vals.append(step_metrics["g_val"])
            d_epoch_vals.append(step_metrics["d_val"])
            d_real_epoch_vals.append(step_metrics["d_real_val"])
            d_fake_epoch_vals.append(step_metrics["d_fake_val"])

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
                d_loss=f"{step_metrics['d_val']:.6f}",
                g_loss=f"{step_metrics['g_val']:.6f}",
                d_real=f"{step_metrics['d_real_val']:.4f}",
                d_fake=f"{step_metrics['d_fake_val']:.4f}",
                margin=f"{step_metrics['margin']:.4f}",
                step=global_step,
            )

            global_step += 1

        epoch_stats = _compute_epoch_averages(
            g_epoch_vals=g_epoch_vals,
            d_epoch_vals=d_epoch_vals,
            d_real_epoch_vals=d_real_epoch_vals,
            d_fake_epoch_vals=d_fake_epoch_vals,
        )

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

        buffers["epoch_g_losses"].append(epoch_stats["mean_g"])
        buffers["epoch_d_losses"].append(epoch_stats["mean_d"])
        buffers["epoch_d_real_scores"].append(epoch_stats["mean_d_real"])
        buffers["epoch_d_fake_scores"].append(epoch_stats["mean_d_fake"])
        buffers["epoch_margins"].append(epoch_stats["mean_margin"])
        buffers["epoch_pixel_mean"].append(sample_mean)
        buffers["epoch_pixel_std"].append(sample_std)
        buffers["epoch_lr_g"].append(optimizer_g.param_groups[0]["lr"])
        buffers["epoch_lr_d"].append(optimizer_d.param_groups[0]["lr"])

        _update_best_checkpoint(
            cfg=cfg,
            run_name=run_name,
            epoch=epoch,
            epoch_stats=epoch_stats,
            state=state,
        )

        scheduler_g.step(epoch_stats["mean_margin"])
        scheduler_d.step(epoch_stats["mean_d"])

        tqdm.write(
            f"{run_name} | Epoch {epoch + 1}/{cfg.n_epochs} | "
            f"D: {epoch_stats['mean_d']:.6f} | G: {epoch_stats['mean_g']:.6f} | "
            f"D(real): {epoch_stats['mean_d_real']:.3f} | "
            f"D(fake): {epoch_stats['mean_d_fake']:.3f} | "
            f"Margin: {epoch_stats['mean_margin']:.4f} | Pixel std: {sample_std:.4f}"
        )

    return global_step


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

    buffers = _init_history_buffers()

    state = {
        "best_margin": -float("inf"),
        "best_epoch": -1,
        "generator": generator,
        "discriminator": discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
    }

    fixed_noise = torch.randn(cfg.fixed_sample_size, cfg.latent_dim, device=device)
    global_step = _run_training_loop(
        cfg=cfg,
        dataloader=dataloader,
        device=device,
        run_name=run_name,
        generator=generator,
        discriminator=discriminator,
        adversarial_loss=adversarial_loss,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        buffers=buffers,
        state=state,
    )

    history = _build_training_history(
        gan_type=gan_type,
        run_name=run_name,
        global_step=global_step,
        buffers=buffers,
        best_margin=state["best_margin"],
        best_epoch=state["best_epoch"],
    )

    save_final_artifacts(
        checkpoints_dir=cfg.checkpoints_dir,
        run_name=run_name,
        generator=generator,
        discriminator=discriminator,
        history=history,
    )

    return {
        "history": history,
        "best_epoch": state["best_epoch"],
        "best_margin": state["best_margin"],
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
