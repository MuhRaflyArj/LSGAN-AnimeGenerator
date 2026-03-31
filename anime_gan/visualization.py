import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from torchvision.utils import make_grid, save_image

from .utils import denorm, smooth_curve


def sample_images(
    generator,
    device,
    latent_dim,
    tag,
    n_samples=25,
    fixed_noise=None,
    show=True,
    out_prefix="epoch",
    images_dir="images/lsgan",
):
    generator.eval()
    with torch.no_grad():
        if fixed_noise is not None:
            z = fixed_noise
        else:
            z = torch.randn(n_samples, latent_dim, device=device)

        gen_imgs = generator(z)
        gen_denorm = denorm(gen_imgs).clamp(0.0, 1.0)

        side = int(np.sqrt(gen_denorm.shape[0]))
        if side * side != gen_denorm.shape[0]:
            side = 5 if gen_denorm.shape[0] >= 25 else max(1, gen_denorm.shape[0])

        save_image(gen_denorm, f"{images_dir}/{out_prefix}_{tag}.png", nrow=side)

    pixel_mean = gen_denorm.mean().item()
    pixel_std = gen_denorm.std().item()

    if show:
        clear_output(wait=True)
        show_n = min(16, gen_denorm.shape[0])
        grid = make_grid(gen_denorm[:show_n].cpu(), nrow=4)
        plt.figure(figsize=(5, 5))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.axis("off")
        plt.title(f"Generated samples: {out_prefix}_{tag}")
        plt.show()
        plt.close()

    generator.train()
    return pixel_mean, pixel_std


def plot_training_dashboard(
    g_losses,
    d_losses,
    d_real_scores,
    d_fake_scores,
    epoch_margins,
    epoch_lr_g,
    epoch_lr_d,
    best_margin,
    save_prefix="",
    title_prefix="",
    images_dir="images/lsgan",
):
    title_head = f"{title_prefix} " if title_prefix else ""
    save_head = f"{save_prefix}_" if save_prefix else ""

    plt.figure(figsize=(12, 5))
    plt.plot(g_losses, alpha=0.3, label="G loss (raw)", color="tab:blue")
    plt.plot(d_losses, alpha=0.3, label="D loss (raw)", color="tab:orange")
    if len(g_losses) >= 25:
        plt.plot(np.arange(24, len(g_losses)), smooth_curve(g_losses, window=25), label="G loss (smooth)", color="tab:blue", linewidth=2)
    if len(d_losses) >= 25:
        plt.plot(np.arange(24, len(d_losses)), smooth_curve(d_losses, window=25), label="D loss (smooth)", color="tab:orange", linewidth=2)
    plt.title(f"{title_head}Training Loss Curves")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{images_dir}/{save_head}loss_curves.png", dpi=140)
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(d_real_scores, label="D(real)", color="tab:green", alpha=0.6)
    plt.plot(d_fake_scores, label="D(fake)", color="tab:red", alpha=0.6)
    plt.title(f"{title_head}Discriminator Scores Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{images_dir}/{save_head}discriminator_scores.png", dpi=140)
    plt.show()
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epoch_margins, marker="o", color="tab:purple")
    axes[0].axhline(best_margin, linestyle="--", color="tab:gray", label=f"Best margin: {best_margin:.4f}")
    axes[0].set_title(f"{title_head}Epoch Margin (D(real) - D(fake))")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Margin")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epoch_lr_g, label="LR_G", color="tab:blue")
    axes[1].plot(epoch_lr_d, label="LR_D", color="tab:orange")
    axes[1].set_title(f"{title_head}Learning Rate Schedule")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{images_dir}/{save_head}margin_and_lr.png", dpi=140)
    plt.show()
    plt.close()


def plot_final_samples(generator, fixed_noise, latent_dim, device, save_prefix="", title_prefix="", images_dir="images/lsgan"):
    title_head = f"{title_prefix} " if title_prefix else ""
    save_head = f"{save_prefix}_" if save_prefix else ""

    generator.eval()
    with torch.no_grad():
        random_noise = torch.randn(100, latent_dim, device=device)
        random_gen = denorm(generator(random_noise)).clamp(0.0, 1.0)
        fixed_gen = denorm(generator(fixed_noise)).clamp(0.0, 1.0)

    save_image(random_gen, f"{images_dir}/{save_head}final_random_10x10.png", nrow=10)
    save_image(fixed_gen, f"{images_dir}/{save_head}final_fixed_8x8.png", nrow=8)

    plt.figure(figsize=(8, 8))
    grid_random = make_grid(random_gen[:64].cpu(), nrow=8)
    plt.imshow(np.transpose(grid_random.numpy(), (1, 2, 0)))
    plt.title(f"{title_head}Final random samples (8x8 preview)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 8))
    grid_fixed = make_grid(fixed_gen[:64].cpu(), nrow=8)
    plt.imshow(np.transpose(grid_fixed.numpy(), (1, 2, 0)))
    plt.title(f"{title_head}Final fixed-noise samples (8x8)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_gan_comparison(vanilla_history, lsgan_history, images_dir="images/lsgan"):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(vanilla_history["epoch_g_losses"], label="Vanilla G", color="tab:blue")
    plt.plot(vanilla_history["epoch_d_losses"], label="Vanilla D", color="tab:orange")
    plt.plot(lsgan_history["epoch_g_losses"], "--", label="LSGAN G", color="tab:green")
    plt.plot(lsgan_history["epoch_d_losses"], "--", label="LSGAN D", color="tab:red")
    plt.title("Epoch Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(vanilla_history["epoch_margins"], label="Vanilla margin", color="tab:purple")
    plt.plot(lsgan_history["epoch_margins"], label="LSGAN margin", color="tab:brown")
    plt.title("Epoch Margin Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Margin")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{images_dir}/vanilla_vs_lsgan_comparison.png", dpi=140)
    plt.show()
    plt.close()


def plot_side_by_side_fixed_samples(vanilla_generator, lsgan_generator, fixed_noise, images_dir="images/lsgan"):
    vanilla_generator.eval()
    lsgan_generator.eval()

    with torch.no_grad():
        vanilla_imgs = denorm(vanilla_generator(fixed_noise)).clamp(0.0, 1.0)
        lsgan_imgs = denorm(lsgan_generator(fixed_noise)).clamp(0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.transpose(make_grid(vanilla_imgs[:64].cpu(), nrow=8).numpy(), (1, 2, 0)))
    axes[0].set_title("Vanilla GAN (fixed noise)")
    axes[0].axis("off")

    axes[1].imshow(np.transpose(make_grid(lsgan_imgs[:64].cpu(), nrow=8).numpy(), (1, 2, 0)))
    axes[1].set_title("LSGAN (fixed noise)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{images_dir}/fixed_noise_side_by_side.png", dpi=140)
    plt.show()
    plt.close()
