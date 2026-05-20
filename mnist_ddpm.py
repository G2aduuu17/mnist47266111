
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def require_dependencies():
    try:
        import matplotlib
        import numpy
        import torch
        import torchvision
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing dependency: {missing}\n\n"
            "Install the required packages, for example:\n"
            "  pip install torch torchvision matplotlib numpy\n\n"
            "If you use conda, install PyTorch from the official PyTorch command "
            "for your CUDA/CPU setup, then install matplotlib and numpy."
        ) from exc

    matplotlib.use("Agg")
    return torch, torchvision, matplotlib, numpy


torch, torchvision, matplotlib, np = require_dependencies()
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal MNIST DDPM.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=200, help="Number of diffusion steps.")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for generated images/checkpoint. Default: this script's folder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for quick tests. Default uses all batches.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use cuda if available by default.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = F.silu(self.norm1(h))
        h = self.conv2(h)
        h = F.silu(self.norm2(h))
        return h + self.skip(x)


class TinyTimeUNet(nn.Module):
    """Small time-conditioned denoiser for 28x28 grayscale MNIST."""

    def __init__(self, base_channels: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c = base_channels
        self.input_conv = nn.Conv2d(1, c, kernel_size=3, padding=1)

        self.down1 = TimeBlock(c, c, time_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)

        self.down2 = TimeBlock(c * 2, c * 2, time_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)

        self.middle = TimeBlock(c * 4, c * 4, time_dim)

        self.upsample2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = TimeBlock(c * 4, c * 2, time_dim)

        self.upsample1 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up1 = TimeBlock(c * 2, c, time_dim)

        self.output = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        x = self.input_conv(x)

        h1 = self.down1(x, time_emb)
        h2 = self.downsample1(h1)
        h2 = self.down2(h2, time_emb)

        h3 = self.downsample2(h2)
        h3 = self.middle(h3, time_emb)

        u2 = self.upsample2(h3)
        u2 = self.up2(torch.cat([u2, h2], dim=1), time_emb)

        u1 = self.upsample1(u2)
        u1 = self.up1(torch.cat([u1, h1], dim=1), time_emb)
        return self.output(u1)


class Diffusion:
    def __init__(self, T: int, beta_start: float, beta_end: float, device: torch.device):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self.extract(self.sqrt_alpha_bars, t, x_start.shape)
        sqrt_omab = self.extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_omab * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, step: int) -> torch.Tensor:
        beta_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_ab = self.extract(self.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alpha = self.extract(self.sqrt_recip_alphas, t, x.shape)

        pred_noise = model(x, t)
        mean = sqrt_recip_alpha * (x - beta_t * pred_noise / sqrt_one_minus_ab)
        if step == 0:
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(x)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        n: int,
        image_size: int = 28,
        channels: int = 1,
        keep_steps: int = 8,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        model.eval()
        x = torch.randn(n, channels, image_size, image_size, device=self.device)
        keep = set(torch.linspace(self.T - 1, 0, keep_steps).long().tolist())
        trajectory = []

        for step in reversed(range(self.T)):
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, step)
            x = x.clamp(-1.5, 1.5)
            if step in keep:
                trajectory.append(x.detach().cpu())

        return x.detach().cpu(), trajectory


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def save_image_grid(x: torch.Tensor, path: Path, nrow: int = 8) -> None:
    utils.save_image(denorm(x), path, nrow=nrow)


def save_forward_grid(
    diffusion: Diffusion,
    images: torch.Tensor,
    out_path: Path,
    steps: int = 8,
) -> None:
    image = images[:1].repeat(steps, 1, 1, 1).to(diffusion.device)
    ts = torch.linspace(0, diffusion.T - 1, steps, device=diffusion.device).long()
    noise = torch.randn_like(image)
    noised = diffusion.q_sample(image, ts, noise).detach().cpu()
    save_image_grid(noised, out_path, nrow=steps)


def save_reverse_trajectory(trajectory: list[torch.Tensor], out_path: Path) -> None:
    if not trajectory:
        return
    first_sample_steps = torch.cat([state[:1] for state in trajectory], dim=0)
    save_image_grid(first_sample_steps, out_path, nrow=len(trajectory))


def save_loss_curve(losses: list[float], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1)
    plt.xlabel("Optimization step")
    plt.ylabel("MSE noise-prediction loss")
    plt.title("MNIST DDPM training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_dataloader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Diffusion steps T: {args.T}")

    dataloader = build_dataloader(args.data_dir, args.batch_size, args.num_workers)
    model = TinyTimeUNet().to(device)
    diffusion = Diffusion(args.T, args.beta_start, args.beta_end, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    first_batch = next(iter(dataloader))[0]
    save_forward_grid(diffusion, first_batch, out_dir / "forward_noising_grid.png")

    losses: list[float] = []
    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        seen = 0
        for batch_idx, (x0, _) in enumerate(dataloader, start=1):
            if args.max_train_batches is not None and batch_idx > args.max_train_batches:
                break

            x0 = x0.to(device)
            batch_size = x0.shape[0]
            t = torch.randint(0, args.T, (batch_size,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            value = float(loss.item())
            losses.append(value)
            running += value
            seen += 1
            global_step += 1

            if batch_idx % 100 == 0:
                print(
                    f"epoch {epoch:02d} batch {batch_idx:04d} "
                    f"step {global_step:05d} loss {value:.4f}"
                )

        avg = running / max(seen, 1)
        print(f"epoch {epoch:02d} average loss {avg:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "losses": losses,
        }
        torch.save(checkpoint, out_dir / "checkpoint.pt")

    save_loss_curve(losses, out_dir / "loss_curve.png")

    samples, trajectory = diffusion.sample(model, n=args.sample_count)
    save_image_grid(samples, out_dir / "generated_digits_grid.png", nrow=int(math.sqrt(args.sample_count)))
    save_reverse_trajectory(trajectory, out_dir / "reverse_denoising_trajectory.png")

    print("\nSaved artifacts:")
    for name in [
        "forward_noising_grid.png",
        "reverse_denoising_trajectory.png",
        "generated_digits_grid.png",
        "loss_curve.png",
        "checkpoint.pt",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    train(parse_args())
