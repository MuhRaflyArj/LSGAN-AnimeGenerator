from pathlib import Path

import torch


def save_best_checkpoint(
    checkpoints_dir: Path,
    run_name: str,
    epoch: int,
    mean_margin: float,
    mean_g: float,
    mean_d: float,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
):
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_name": run_name,
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G_state_dict": optimizer_g.state_dict(),
            "optimizer_D_state_dict": optimizer_d.state_dict(),
            "best_margin": mean_margin,
            "epoch_g_loss": mean_g,
            "epoch_d_loss": mean_d,
        },
        checkpoints_dir / f"{run_name}_best.pt",
    )
    torch.save(generator.state_dict(), checkpoints_dir / f"{run_name}_generator_best.pt")
    torch.save(discriminator.state_dict(), checkpoints_dir / f"{run_name}_discriminator_best.pt")


def save_final_artifacts(checkpoints_dir: Path, run_name: str, generator, discriminator, history: dict):
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), checkpoints_dir / f"{run_name}_generator_final.pt")
    torch.save(discriminator.state_dict(), checkpoints_dir / f"{run_name}_discriminator_final.pt")
    torch.save(history, checkpoints_dir / f"{run_name}_training_history.pt")
