import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import CHECKPOINT_DIR, RUNS_DIR, TrainConfig
from src.data.dataset import LEVIRCDDataset
from src.losses import BCEDiceLoss
from src.metrics import SegMetricsAccumulator
from src.pipeline import build_model, forward_pass, load_checkpoint


def parse_args() -> tuple[TrainConfig, Path | None]:
    defaults = TrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "snn"], default=defaults.model)
    p.add_argument("--encoding", choices=["rate", "difference", "spectral"], default=defaults.encoding.kind)
    p.add_argument("--time-steps", type=int, default=defaults.encoding.time_steps)
    p.add_argument("--diff-threshold", type=float, default=defaults.encoding.diff_threshold)
    p.add_argument("--beta", type=float, default=defaults.model_cfg.beta)
    p.add_argument("--patch-size", type=int, default=defaults.data.patch_size)
    p.add_argument("--crops-per-image", type=int, default=defaults.data.crops_per_image)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full epoch")
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--min-lr", type=float, default=defaults.min_lr)
    p.add_argument("--change-class-weight", type=float, default=defaults.change_class_weight)
    p.add_argument("--num-workers", type=int, default=defaults.data.num_workers)
    p.add_argument("--data-root", type=str, default=defaults.data.root, help="LEVIR-CD root (with train/val/test)")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--use-checkpointing", action="store_true", help="per-time-step gradient checkpointing (SNN)")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint (e.g. checkpoints/<run>/last.pt) to continue; its saved config is used, "
        "other flags are ignored except --num-workers and --data-root",
    )
    args = p.parse_args()

    cfg = TrainConfig(
        model=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        change_class_weight=args.change_class_weight,
        seed=args.seed,
        use_checkpointing=args.use_checkpointing,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )
    cfg.data.patch_size = args.patch_size
    cfg.data.crops_per_image = args.crops_per_image
    cfg.data.num_workers = args.num_workers
    cfg.data.root = str(args.data_root)
    cfg.encoding.kind = args.encoding
    cfg.encoding.time_steps = args.time_steps
    cfg.encoding.diff_threshold = args.diff_threshold
    cfg.model_cfg.beta = args.beta
    cfg.run_name = args.run_name or f"{cfg.model}_{cfg.encoding.kind}_{int(time.time())}"
    return cfg, Path(args.resume) if args.resume else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomic(obj: dict, path: Path) -> None:
    # write-then-rename: an instance killed mid-save keeps the previous file intact
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> None:
    cfg, resume_path = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = None
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path)
        # machine-specific settings come from this invocation, not the checkpoint
        num_workers, data_root = cfg.data.num_workers, cfg.data.root
        cfg = ckpt["config"]
        cfg.data.num_workers, cfg.data.root = num_workers, data_root
    seed_everything(cfg.seed)

    train_ds = LEVIRCDDataset(
        cfg.data.root, "train", patch_size=cfg.data.patch_size, train=True,
        crops_per_image=cfg.data.crops_per_image,
    )
    val_ds = LEVIRCDDataset(cfg.data.root, "val", patch_size=cfg.data.patch_size, train=False)
    persistent = cfg.data.num_workers > 0
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
        drop_last=True, pin_memory=True, persistent_workers=persistent,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=True, persistent_workers=persistent,
    )

    model = build_model(cfg).to(device)
    loss_fn = BCEDiceLoss(
        pos_weight=cfg.change_class_weight, bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    start_epoch, global_step, best_f1 = 0, 0, -1.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_f1 = ckpt["best_f1"]
        set_rng_state(ckpt["rng_state"])
        print(f"resumed {cfg.run_name} from epoch {ckpt['epoch']} (best val f1 {best_f1:.4f})")
        del ckpt

    run_dir = RUNS_DIR / cfg.run_name
    ckpt_dir = CHECKPOINT_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # purge_step drops events logged after the last checkpoint of an interrupted run
    writer = SummaryWriter(log_dir=str(run_dir), purge_step=global_step if start_epoch else None)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        for step, (img1, img2, label) in enumerate(train_dl):
            if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
                break
            img1, img2, label = (t.to(device, non_blocking=True) for t in (img1, img2, label))

            logits = forward_pass(model, img1, img2, cfg)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()

        model.eval()
        val_metrics = SegMetricsAccumulator()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for step, (img1, img2, label) in enumerate(val_dl):
                if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
                    break
                img1, img2, label = (t.to(device, non_blocking=True) for t in (img1, img2, label))
                logits = forward_pass(model, img1, img2, cfg)
                val_loss_total += loss_fn(logits, label).item()
                val_batches += 1
                val_metrics.update(logits, label)

        metrics = val_metrics.compute()
        writer.add_scalar("val/loss", val_loss_total / max(1, val_batches), epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        is_best = metrics["f1"] > best_f1
        if is_best:
            best_f1 = metrics["f1"]
        print(
            f"epoch {epoch}: val_loss={val_loss_total / max(1, val_batches):.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            + (" *best*" if is_best else "")
        )

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_f1": best_f1,
            "val_metrics": metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": rng_state(),
            "config": cfg,
        }
        save_atomic(state, ckpt_dir / "last.pt")
        if is_best:
            save_atomic(state, ckpt_dir / "best.pt")
        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
