"""Train, resume, or evaluate LACA and LACA-L."""
import argparse
import logging
from pathlib import Path

import torch

from LACA.checkpoint import load_checkpoint
from LACA.config import export_model_config
from LACA.config.runtime import build_config
from LACA.dataloader import build_train_loader
from LACA.laca import LACA
from LACA.loss import build_criterion
from LACA.utils import dist_utils as comm, misc
from LACA.utils.eval_disp import eval_disp
from LACA.utils.launch import launch
from LACA.utils.logger import setup_logger


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default="", help="Experiment YAML file")
    parser.add_argument("--checkpoint-dir", default="checkpoints/laca", help="Logs and saved weights")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--from-pretrained", help="Checkpoint path or sf/l_sf/kitti; initialize weights or evaluate")
    parser.add_argument("--data-root", help="Parent directory containing SceneFlow, KITTI, ETH3D, etc.")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs per machine; 0 for CPU")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[], help="Configuration KEY VALUE pairs")
    return parser


def build_optimizer(model, cfg):
    base_lr = cfg.SOLVER.BASE_LR
    backbone_weight_decay = cfg.SOLVER.BACKBONE_WEIGHT_DECAY
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    norm_module_types = (
        torch.nn.BatchNorm2d,
        torch.nn.InstanceNorm2d,
        torch.nn.LayerNorm,
    )
    params = []
    params_norm = []
    params_backbone = []
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            if f"{module_name}.{module_param_name}".startswith("stereo_encoder.backbone"):
                params_backbone.append(value)
            elif isinstance(module, norm_module_types) and weight_decay_norm is not None:
                params_norm.append(value)
            else:
                params.append(value)
    ret = []
    if len(params) > 0:
        ret.append({"params": params, "lr": base_lr})
    if len(params_norm) > 0:
        ret.append({"params": params_norm, "lr": base_lr, "weight_decay": weight_decay_norm})
    if len(params_backbone) > 0:
        ret.append({"params": params_backbone, "lr": base_lr, "weight_decay": backbone_weight_decay})
    adamw_args = {
        "params": ret,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY
    }
    return torch.optim.AdamW(**adamw_args)

def make_scheduler(optimizer, cfg, start_step=0):
    if start_step:
        # PyTorch needs these group fields when reconstructing an old schedule.
        for group in optimizer.param_groups:
            group.update(initial_lr=cfg.SOLVER.BASE_LR / 25,
                         max_lr=cfg.SOLVER.BASE_LR, min_lr=cfg.SOLVER.BASE_LR / 25 / 10000)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg.SOLVER.BASE_LR, total_steps=cfg.SOLVER.MAX_ITER + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy="cos",
        last_epoch=start_step - 1,
    )
    return scheduler


def write_checkpoint(path, model, cfg, optimizer=None, scheduler=None, step=0, epoch=0):
    state = {"model": model.state_dict(), "model_config": export_model_config(cfg)}
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step, epoch=epoch)
    # Replace only after serialization finishes, so an interrupted save stays recoverable.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def log_scalars(writer, values, step, prefix):
    if writer is None:
        return
    for name, value in values.items():
        tag = f"{prefix}/{name}"
        if isinstance(value, dict):
            log_scalars(writer, value, step, tag)
        else:
            writer.add_scalar(tag, value, step)


def main(args):
    cfg = build_config(args.config_file, args.opts, data_root=args.data_root)
    resume = cfg.SOLVER.RESUME if not args.eval_only else ""
    source = resume or args.from_pretrained
    if args.eval_only and not source:
        raise ValueError("--eval-only requires --from-pretrained CHECKPOINT.")
    checkpoint = load_checkpoint(source) if source else None
    if checkpoint is not None and "model_config" in checkpoint:
        cfg = build_config(args.config_file, args.opts, checkpoint["model_config"], args.data_root)

    output_dir = Path(args.checkpoint_dir)
    if comm.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(output_dir), distributed_rank=comm.get_rank(), name="LACA")
    misc.seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + comm.get_rank())
    if comm.is_main_process():
        (output_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    device = torch.device("cuda", comm.get_local_rank()) if args.num_gpus else torch.device("cpu")
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK and not args.eval_only
    # Full model checkpoints already contain the frozen monocular weights.
    model = LACA(cfg, mono_pretrained=checkpoint is None, backbone_pretrained=checkpoint is None).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=cfg.SOLVER.STRICT_RESUME if resume else True)
    model_without_ddp = model
    if comm.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    if args.eval_only:
        del checkpoint
        return eval_disp(model, cfg)

    train_loader, train_sampler = build_train_loader(cfg)
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty: check dataset paths and batch size.")
    optimizer = build_optimizer(model_without_ddp, cfg)
    criterion = build_criterion(cfg).to(device)
    start_step = start_epoch = 0
    scheduler = make_scheduler(optimizer, cfg)
    if resume and not cfg.SOLVER.NO_RESUME_OPTIMIZER and "optimizer" in checkpoint:
        start_step, start_epoch = checkpoint.get("step", 0), checkpoint.get("epoch", 0)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            if checkpoint["scheduler"]["total_steps"] != cfg.SOLVER.MAX_ITER + 100:
                raise ValueError("Resume MAX_ITER must match the saved schedule; use SOLVER.NO_RESUME_OPTIMIZER True for fine-tuning.")
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            scheduler = make_scheduler(optimizer, cfg, start_step)
        logger.info("Resumed optimizer and schedule at step %d.", start_step)
    del checkpoint
    logger.info("Parameters: %.2fM total, %.2fM trainable.",
                sum(p.numel() for p in model_without_ddp.parameters()) / 1e6,
                sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1e6)
    writer = None
    if comm.is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output_dir))
        except ImportError:
            logger.info("TensorBoard is unavailable; training metrics will be written to the log.")
    total_steps, epoch = start_step, start_epoch
    metrics = misc.MetricLogger(delimiter="  ")
    try:
        while total_steps < cfg.SOLVER.MAX_ITER:
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            for sample in metrics.log_every(train_loader, 20, f"Epoch [{epoch}]", logger=logger):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(sample)
                loss_dict = criterion(outputs, sample, log=True)
                loss = sum(value * criterion.weight_dict[key] for key, value in loss_dict.items()
                           if key in criterion.weight_dict)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {total_steps}: {loss_dict}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                total_steps += 1
                metrics.update(lr=scheduler.get_last_lr()[0], loss=loss.detach())
                metrics.update(**{k: v.detach() for k, v in loss_dict.items() if not k.endswith("_aux")})
                log_scalars(writer, loss_dict, total_steps, "train")

                final_step = total_steps == cfg.SOLVER.MAX_ITER
                if comm.is_main_process():
                    period = cfg.SOLVER.CHECKPOINT_PERIOD
                    if period > 0 and (total_steps % period == 0 or final_step):
                        write_checkpoint(output_dir / f"step_{total_steps:06d}.pth", model_without_ddp, cfg)
                    period = cfg.SOLVER.LATEST_CHECKPOINT_PERIOD
                    if final_step or (period > 0 and total_steps % period == 0):
                        write_checkpoint(output_dir / "checkpoint_latest.pth", model_without_ddp, cfg,
                                         optimizer, scheduler, total_steps, epoch)
                # Validation runs only at the requested interval, never every batch.
                if cfg.TEST.EVAL_PERIOD > 0 and total_steps % cfg.TEST.EVAL_PERIOD == 0:
                    log_scalars(writer, eval_disp(model, cfg), total_steps, "val")
                    model.train()
                if final_step:
                    break
            epoch += 1
        logger.info("Training completed at step %d.", total_steps)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.num_gpus < 0 or args.num_gpus > torch.cuda.device_count():
        parser.error(f"--num-gpus must be between 0 and {torch.cuda.device_count()} on this machine.")
    if args.num_machines < 1 or not 0 <= args.machine_rank < args.num_machines:
        parser.error("Invalid machine count/rank.")
    if args.num_gpus == 0 and args.num_machines != 1:
        parser.error("CPU execution currently supports a single machine.")
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
