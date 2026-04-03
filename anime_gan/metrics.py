import random
from typing import Dict

import matplotlib.pyplot as plt
import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from lpips.lpips import LPIPS


def _to_uint8_from_tanh(x: torch.Tensor) -> torch.Tensor:
    x = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0
    return x.to(torch.uint8)


def _compute_fid_kid(generator, dataloader, device, latent_dim: int, max_samples: int) -> Dict[str, float]:
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=50, normalize=False).to(device)

    seen = 0
    generator.eval()

    with torch.no_grad():
        for real_imgs in dataloader:
            if seen >= max_samples:
                break

            real_imgs = real_imgs.to(device)
            bsz = real_imgs.shape[0]
            take = min(bsz, max_samples - seen)
            real_imgs = real_imgs[:take]

            z = torch.randn((take, latent_dim), device=device)
            fake_imgs = generator(z)

            fid.update(_to_uint8_from_tanh(real_imgs), real=True)
            fid.update(_to_uint8_from_tanh(fake_imgs), real=False)

            kid.update(_to_uint8_from_tanh(real_imgs), real=True)
            kid.update(_to_uint8_from_tanh(fake_imgs), real=False)

            seen += take

    fid_val = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()

    return {
        "fid": fid_val,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
    }


def _compute_lpips_diversity(generator, device, latent_dim: int, sample_count: int = 128, pair_count: int = 64) -> float:
    pair_count = max(1, pair_count)
    sample_count = max(2, sample_count)

    metric = LPIPS(net="alex").to(device)
    metric.eval()

    generator.eval()
    with torch.no_grad():
        z = torch.randn((sample_count, latent_dim), device=device)
        samples = generator(z).clamp(-1.0, 1.0)

    indices = list(range(sample_count))
    scores = []

    with torch.no_grad():
        for _ in range(pair_count):
            i, j = random.sample(indices, 2)
            score = metric(samples[i : i + 1], samples[j : j + 1]).item()
            scores.append(score)

    return float(sum(scores) / len(scores))


def evaluate_gan_models(
    vanilla_generator,
    lsgan_generator,
    dataloader,
    device,
    latent_dim: int,
    max_samples: int = 2048,
    lpips_samples: int = 128,
    lpips_pairs: int = 64,
) -> Dict[str, Dict[str, float]]:
    """Compute comparable FID, KID, and LPIPS diversity for vanilla and LSGAN."""

    vanilla_scores = _compute_fid_kid(
        generator=vanilla_generator,
        dataloader=dataloader,
        device=device,
        latent_dim=latent_dim,
        max_samples=max_samples,
    )
    vanilla_scores["lpips_diversity"] = _compute_lpips_diversity(
        generator=vanilla_generator,
        device=device,
        latent_dim=latent_dim,
        sample_count=lpips_samples,
        pair_count=lpips_pairs,
    )

    lsgan_scores = _compute_fid_kid(
        generator=lsgan_generator,
        dataloader=dataloader,
        device=device,
        latent_dim=latent_dim,
        max_samples=max_samples,
    )
    lsgan_scores["lpips_diversity"] = _compute_lpips_diversity(
        generator=lsgan_generator,
        device=device,
        latent_dim=latent_dim,
        sample_count=lpips_samples,
        pair_count=lpips_pairs,
    )

    return {
        "vanilla": vanilla_scores,
        "lsgan": lsgan_scores,
    }


def print_metrics_table(metrics: Dict[str, Dict[str, float]]) -> None:
    print("\nGAN Metric Summary")
    print("Model      | FID (lower) | KID mean (lower) | KID std | LPIPS diversity (higher)")
    print("-" * 80)

    for model_name in ["vanilla", "lsgan"]:
        m = metrics[model_name]
        print(
            f"{model_name:<10} | {m['fid']:<11.4f} | {m['kid_mean']:<16.6f} | {m['kid_std']:<7.6f} | {m['lpips_diversity']:<.6f}"
        )


def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]], images_dir: str = "images/lsgan") -> None:
    labels = ["FID", "KID", "LPIPS"]

    vanilla_vals = [
        metrics["vanilla"]["fid"],
        metrics["vanilla"]["kid_mean"],
        metrics["vanilla"]["lpips_diversity"],
    ]
    lsgan_vals = [
        metrics["lsgan"]["fid"],
        metrics["lsgan"]["kid_mean"],
        metrics["lsgan"]["lpips_diversity"],
    ]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width / 2 for i in x], vanilla_vals, width=width, label="Vanilla")
    plt.bar([i + width / 2 for i in x], lsgan_vals, width=width, label="LSGAN")
    plt.xticks(list(x), labels)
    plt.title("Metric Comparison (Vanilla vs LSGAN)")
    plt.legend()
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(f"{images_dir}/fid_kid_lpips_comparison.png", dpi=140)
    plt.show()
    plt.close()
