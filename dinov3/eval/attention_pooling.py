# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    cleanup_checkpoint,
    find_latest_checkpoint,
    keep_last_n_checkpoints,
)
from dinov3.data import SamplerType, make_data_loader
from dinov3.eval.data import create_train_dataset_dict, get_num_classes
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.linear import (
    AllClassifiers,
    EvalConfig,
    FewShotConfig,
    TrainConfig,
    TransformConfig,
    make_evaluators,
    make_train_dataset,
    scale_lr,
)
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.eval.utils import LossType, ModelWithIntermediateLayers, average_metrics
from dinov3.eval.utils import save_results as default_save_results_func
from dinov3.logging import MetricLogger, SmoothedValue
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-attention-pooling.csv"
MAIN_METRICS = [".*_accuracy(_mean)?"]

_DEFAULT_WD_LIST: Tuple[float, ...] = (0.0,)


def sliding_window_coords(h: int, w: int, crop: int, stride: int):
    coords = []
    for y in range(0, max(h - crop, 0) + 1, stride):
        for x in range(0, max(w - crop, 0) + 1, stride):
            coords.append((y, x))
    if coords and coords[-1][0] + crop < h:
        for x in range(0, max(w - crop, 0) + 1, stride):
            coords.append((h - crop, x))
    if coords and coords[-1][1] + crop < w:
        for y in range(0, max(h - crop, 0) + 1, stride):
            coords.append((y, w - crop))
    if h > crop and w > crop:
        coords.append((h - crop, w - crop))
    coords = list(dict.fromkeys(coords))
    # If image is smaller than crop, just use (0, 0)
    if not coords:
        coords = [(0, 0)]
    return coords


@dataclass
class AttnPoolTrainConfig(TrainConfig):
    weight_decays: Tuple[float, ...] = _DEFAULT_WD_LIST  # weight decay values to grid search
    cls_patch: bool = True
    warmup_iterations: int = 0
    intermediate_layers: bool = False
    embed_dim: int = 512
    num_heads: int = 8
    attn_dropout: float = 0.0
    l2_normalize_input: bool = False
    nb_curriculum_epoch: int = 0  # number of epochs to train on single-label samples only (0 = disabled)
    reload_best_for_eval: bool = False  # if True, reload best (val) checkpoint before final eval


@dataclass
class AttnPoolEvalConfig:
    model: ModelConfig
    train: AttnPoolTrainConfig = field(default_factory=AttnPoolTrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    save_results: bool = False
    output_dir: str = ""


def create_attn_pool_input(x_tokens_list, use_n_blocks, use_cls_patch=False, l2_normalize_cls=False):
    """Extract patch tokens from the last use_n_blocks layers, concatenated along the feature dim.

    If use_cls_patch is True, prepends the CLS token(s) to the patch token sequence.
    If l2_normalize_cls is True, L2-normalizes the CLS tokens before prepending (only effective when use_cls_patch is True).
    """
    intermediate_output = x_tokens_list[-use_n_blocks:]
    patch_tokens = torch.cat([patch for patch, _ in intermediate_output], dim=-1)  # [B, N, n*D]
    if use_cls_patch:
        cls_tokens = torch.cat([cls for _, cls in intermediate_output], dim=-1)  # [B, n*D]
        if l2_normalize_cls:
            cls_tokens = F.normalize(cls_tokens, dim=-1, p=2)
        patch_tokens = torch.cat([cls_tokens.unsqueeze(1), patch_tokens], dim=1)  # [B, 1+N, n*D]
    return patch_tokens.float()


def extract_multicrop_features(feature_model, images, crop_size, stride):
    """Run backbone on sliding-window crops and merge features across crops.

    Args:
        feature_model: ``ModelWithIntermediateLayers`` (frozen backbone wrapper).
        images: ``[B, C, H, W]`` batch.
        crop_size: sliding-window crop size (must match backbone's expected resolution).
        stride: sliding-window stride.

    Returns:
        List of ``(merged_patches, avg_cls)`` tuples — one per intermediate layer —
        in the same format that ``feature_model()`` produces, so existing AP heads
        can consume the output unchanged.
    """
    B, C, H, W = images.shape

    pad_h = max(crop_size - H, 0)
    pad_w = max(crop_size - W, 0)
    if pad_h > 0 or pad_w > 0:
        images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)
        _, _, H, W = images.shape

    coords = sliding_window_coords(H, W, crop_size, stride)
    N_crops = len(coords)

    crops = torch.stack(
        [images[:, :, y : y + crop_size, x : x + crop_size] for y, x in coords],
        dim=1,
    )  # [B, N_crops, C, crop_size, crop_size]
    crops = crops.reshape(B * N_crops, C, crop_size, crop_size)

    with torch.no_grad():
        features = feature_model(crops)
    # features: list of (patch_tokens, cls_token) per intermediate layer
    # patch_tokens: [B*N_crops, P, D],  cls_token: [B*N_crops, D]

    merged = []
    for patch_tokens, cls_token in features:
        P, D = patch_tokens.shape[1], patch_tokens.shape[2]
        patch_tokens = patch_tokens.reshape(B, N_crops, P, D).reshape(B, N_crops * P, D)
        cls_token = cls_token.reshape(B, N_crops, D).mean(dim=1)
        merged.append((patch_tokens, cls_token))

    return merged


class AttnPoolClassifier(nn.Module):
    """Attention pooling classifier to train on top of frozen patch features."""

    def __init__(
        self,
        out_dim,
        use_n_blocks,
        use_cls_patch=False,
        num_classes=1000,
        embed_dim=512,
        num_heads=8,
        dropout=0.0,
        l2_normalize_input=False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_cls_patch = use_cls_patch
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.l2_normalize_input = l2_normalize_input
        self.input_proj = nn.Linear(out_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.query_token = nn.Parameter(torch.empty(embed_dim))
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x_tokens_list):
        feat_tokens = create_attn_pool_input(
            x_tokens_list, self.use_n_blocks, self.use_cls_patch, l2_normalize_cls=self.l2_normalize_input
        )
        B, N, _ = feat_tokens.shape
        feat_tokens = self.ln(self.input_proj(feat_tokens))
        D = self.embed_dim

        q = self.query_token.expand(B, 1, -1)
        q = q.reshape(B, 1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(feat_tokens).reshape(B, N, 2, self.num_heads, D // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = torch.unbind(kv, dim=0)

        x = F.scaled_dot_product_attention(q, k, v)  # [B, heads, head_dim]
        x = self.drop(x.reshape(B, D))

        return self.linear(x)


def setup_attn_pool_classifiers(
    sample_output,
    n_last_blocks_list,
    learning_rates,
    weight_decays,
    batch_size,
    num_classes=1000,
    cls_patch=True,
    embed_dim=512,
    num_heads=8,
    dropout=0.0,
    l2_normalize_input=False,
):
    cls_patch_values = [False, True] if cls_patch else [False]
    classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        out_dim = create_attn_pool_input(sample_output, use_n_blocks=n).shape[-1]
        for use_cls_patch in cls_patch_values:
            for _lr in learning_rates:
                for _wd in weight_decays:
                    lr = scale_lr(_lr, batch_size)
                    classifier = AttnPoolClassifier(
                        out_dim,
                        use_n_blocks=n,
                        use_cls_patch=use_cls_patch,
                        num_classes=num_classes,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        l2_normalize_input=l2_normalize_input,
                    )
                    classifier = classifier.cuda()
                    classifier_name = (
                        f"classifier_{n}_blocks_cls_patch_{use_cls_patch}_lr_{lr:.2e}_wd_{_wd:.2e}".replace(".", "_")
                    )
                    assert classifier_name not in classifiers_dict, f"Classifier name {classifier_name} duplicated!"
                    classifiers_dict[classifier_name] = classifier
                    optim_param_groups.append({"params": classifier.parameters(), "lr": lr, "weight_decay": _wd})

    all_classifiers = AllClassifiers(classifiers_dict)
    first_classifier = next(iter(classifiers_dict.values()))
    num_params = sum(p.numel() for p in first_classifier.parameters() if p.requires_grad)
    logger.info(f"Attention pooling classifier parameters: {num_params:,}")
    if distributed.is_enabled():
        all_classifiers = nn.parallel.DistributedDataParallel(all_classifiers)

    return all_classifiers, optim_param_groups


def setup_attn_pool_training(
    *,
    config: AttnPoolTrainConfig,
    sample_output,
    training_num_classes: int,
    checkpoint_output_dir: str,
):
    classifiers, optim_param_groups = setup_attn_pool_classifiers(
        sample_output,
        config.n_last_blocks_list,
        config.learning_rates,
        config.weight_decays,
        config.batch_size,
        training_num_classes,
        config.cls_patch,
        config.embed_dim,
        config.num_heads,
        config.attn_dropout,
        config.l2_normalize_input,
    )
    max_iter = config.epochs * config.epoch_length
    optimizer = config.optimizer_type.get_optimizer(optim_param_groups=optim_param_groups)
    warmup_iterations = config.warmup_iterations
    scheduler = config.scheduler_type.get_scheduler(
        optimizer=optimizer,
        optim_param_groups=optim_param_groups,
        epoch_length=config.epoch_length,
        epochs=config.epochs,
        max_iter=max_iter - warmup_iterations,
    )
    if warmup_iterations > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_iterations
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_iterations]
        )

    start_iter = 0
    best_accuracy = -1
    if config.resume and (
        last_checkpoint_dir := find_latest_checkpoint(config.classifier_fpath or checkpoint_output_dir)
    ):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        checkpoint = torch.load(last_checkpoint_dir / "checkpoint.pth")
        start_iter = checkpoint.get("iteration", -1) + 1
        best_accuracy = checkpoint.get("best_accuracy", -1)
        classifiers.load_state_dict(checkpoint["classifiers"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if config.loss_type == LossType.BINARY_CROSS_ENTROPY:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return (
        classifiers,
        start_iter,
        max_iter,
        criterion,
        optimizer,
        scheduler,
        best_accuracy,
    )


def train_attn_pool_classifiers(
    *,
    feature_model,
    train_dataset,
    train_config: AttnPoolTrainConfig,
    training_num_classes: int,
    val_evaluator,
    checkpoint_output_dir: str,
):
    (
        classifiers,
        start_iter,
        max_iter,
        criterion,
        optimizer,
        scheduler,
        best_accuracy,
    ) = setup_attn_pool_training(
        config=train_config,
        sample_output=feature_model(train_dataset[0][0].unsqueeze(0).cuda()),
        training_num_classes=training_num_classes,
        checkpoint_output_dir=checkpoint_output_dir,
    )
    checkpoint_period = train_config.save_checkpoint_iterations or train_config.epoch_length
    eval_period = train_config.eval_period_iterations or train_config.epoch_length

    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=SamplerType.INFINITE,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )

    curriculum_end_iter = train_config.nb_curriculum_epoch * train_config.epoch_length
    if curriculum_end_iter > 0:
        logger.info(
            f"Curriculum learning enabled: single-label samples only for the first "
            f"{train_config.nb_curriculum_epoch} epochs ({curriculum_end_iter} iterations)"
        )

    iteration = start_iter
    logger.info(f"Starting attention pooling training from iteration {start_iter}")
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6g}"))
    metric_logger.add_meter("wd", SmoothedValue(window_size=1, fmt="{value:.6g}"))
    header = "Training"

    ckpt_dir = Path(checkpoint_output_dir).expanduser()

    def _save_checkpoint(sub_dir: str, iteration: int, best_accuracy: float):
        if distributed.is_subgroup_main_process():
            (ckpt_dir / sub_dir).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "iteration": iteration,
                "classifiers": classifiers.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
            }
            torch.save(checkpoint, ckpt_dir / sub_dir / "checkpoint.pth")

    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)
        if isinstance(labels, (list, tuple)):
            labels = labels[0]
            if isinstance(labels, list):
                if len(labels) == 1:
                    labels = labels[0] if isinstance(labels[0], torch.Tensor) else torch.stack(labels).squeeze(0)
                else:
                    labels = torch.stack(labels)
        labels = labels.cuda(non_blocking=True)

        if iteration < curriculum_end_iter and len(labels.shape) > 1:
            single_label_mask = labels.sum(dim=-1) == 1
            if single_label_mask.any():
                data = data[single_label_mask]
                labels = labels[single_label_mask]
            else:
                scheduler.step()
                iteration = iteration + 1
                continue
            if iteration == curriculum_end_iter - 1:
                logger.info("Curriculum phase complete, switching to full dataset")

        features = feature_model(data)
        outputs = classifiers(features)

        if len(labels.shape) > 1:
            labels = labels.float()
        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        # Checkpointing
        is_last_iteration = (iteration + 1) == max_iter
        is_ckpt_iteration = ((iteration + 1) % checkpoint_period == 0) or is_last_iteration
        if is_ckpt_iteration:
            ckpt_sub_dir = "final" if is_last_iteration else str(iteration)
            _save_checkpoint(ckpt_sub_dir, iteration, best_accuracy)
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, train_config.checkpoint_retention_policy.max_to_keep)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            val_results_dict = val_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=classifiers,
                prefixstring=f"ITER: {iteration}",
                iteration=iteration,
            )
            val_accuracy = val_results_dict[val_evaluator.main_metric_name]
            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                _save_checkpoint("best", iteration, best_accuracy)
            torch.distributed.barrier()

        iteration = iteration + 1

    return feature_model, classifiers, iteration, best_accuracy


def eval_attention_pooling_with_model(*, model: torch.nn.Module, autocast_dtype, config: AttnPoolEvalConfig):
    start = time.time()
    cudnn.benchmark = True

    train_dataset = make_train_dataset(config.train.dataset, config.transform)
    training_num_classes = get_num_classes(train_dataset)
    train_dataset_dict = create_train_dataset_dict(
        train_dataset,
        few_shot_eval=config.few_shot.enable,
        few_shot_k_or_percent=config.few_shot.k_or_percent,
        few_shot_n_tries=config.few_shot.n_tries,
    )
    n_last_blocks = max(config.train.n_last_blocks_list)
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    if config.train.intermediate_layers:
        total_blocks = len(model.blocks)
        layer_indices = [(i + 1) * total_blocks // n_last_blocks - 1 for i in range(n_last_blocks)]
        logger.info(
            f"Using {n_last_blocks} intermediate layers (evenly spaced) "
            f"out of {total_blocks} total blocks: {layer_indices}"
        )
        feature_model = ModelWithIntermediateLayers(model, layer_indices, autocast_ctx)

    save_results_func = None
    if config.save_results:
        save_results_func = partial(default_save_results_func, output_dir=config.output_dir)

    metrics_file_path = os.path.join(config.output_dir, "results_eval_attention_pooling.json")
    val_evaluator, test_evaluators = make_evaluators(
        eval_config=config.eval,
        val_metric_type=config.train.val_metric_type,
        val_dataset=config.train.val_dataset,
        transform_config=config.transform,
        metrics_file_path=metrics_file_path,
        training_num_classes=training_num_classes,
        save_results_func=save_results_func,
    )
    results_dict = {}
    checkpoint_output_dirs: list = []
    for _try in train_dataset_dict.keys():
        if len(train_dataset_dict) > 1:
            checkpoint_output_dir = os.path.join(config.output_dir, f"checkpoints_{_try}")
            save_filename_suffix = f"_{_try}"
        else:
            checkpoint_output_dir = os.path.join(config.output_dir, "checkpoints")
            save_filename_suffix = ""
        os.makedirs(checkpoint_output_dir, exist_ok=True)

        feature_model, classifiers, iteration, best_accuracy = train_attn_pool_classifiers(
            feature_model=feature_model,
            train_dataset=train_dataset_dict[_try],
            train_config=config.train,
            training_num_classes=training_num_classes,
            val_evaluator=val_evaluator,
            checkpoint_output_dir=checkpoint_output_dir,
        )
        checkpoint_output_dirs.append(checkpoint_output_dir)

        final_val_results = val_evaluator.evaluate_and_maybe_save(
            feature_model=feature_model,
            linear_classifiers=classifiers,
            prefixstring=f"ITER: {iteration} (final)",
            iteration=iteration,
        )
        final_val_accuracy = final_val_results[val_evaluator.main_metric_name]
        if final_val_accuracy >= best_accuracy:
            best_accuracy = final_val_accuracy
            if distributed.is_subgroup_main_process():
                best_ckpt_dir = Path(checkpoint_output_dir).expanduser() / "best"
                best_ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "iteration": iteration,
                        "classifiers": classifiers.state_dict(),
                        "best_accuracy": best_accuracy,
                    },
                    best_ckpt_dir / "checkpoint.pth",
                )
        if distributed.is_enabled():
            torch.distributed.barrier()

        # `reload_best_for_eval=True` reloads the best-on-val classifier checkpoint
        # before the final test pass. When False, the test pass runs on the
        # last-iteration weights.
        if config.train.reload_best_for_eval:
            best_ckpt_path = Path(checkpoint_output_dir).expanduser() / "best" / "checkpoint.pth"
            if best_ckpt_path.exists():
                logger.info(f"Reloading best checkpoint from {best_ckpt_path}")
                best_ckpt = torch.load(best_ckpt_path)
                classifiers.load_state_dict(best_ckpt["classifiers"])
            else:
                logger.warning("No best checkpoint found, using last-iteration weights for evaluation")
        else:
            logger.info("reload_best_for_eval=False; using last-iteration weights for evaluation")

        results_dict[_try] = val_evaluator.evaluate_and_maybe_save(
            feature_model=feature_model,
            linear_classifiers=classifiers,
            iteration=iteration,
            save_filename_suffix=save_filename_suffix,
        )
        for test_evaluator in test_evaluators:
            eval_results_dict = test_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=classifiers,
                iteration=iteration,
                best_classifier_on_val=results_dict[_try]["best_classifier"],
                save_filename_suffix=save_filename_suffix,
            )
            results_dict[_try] = {**eval_results_dict, **results_dict[_try]}

    if len(train_dataset_dict) > 1:
        results_dict = average_metrics(results_dict, ignore_keys=["best_classifier"])
    else:
        results_dict = {**results_dict[_try]}

    for checkpoint_output_dir in checkpoint_output_dirs:
        if distributed.is_subgroup_main_process():
            cleanup_checkpoint(checkpoint_output_dir, config.train.checkpoint_retention_policy)

    logger.info("Test Results Dict " + str(results_dict))
    logger.info(f"Attention pooling evaluation done in {int(time.time() - start)}s")
    return results_dict


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=AttnPoolEvalConfig)
    model, model_context = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    results_dict = eval_attention_pooling_with_model(
        model=model, config=dataclass_config, autocast_dtype=model_context["autocast_dtype"]
    )
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"], distributed_timeout=timedelta(hours=1)):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    main()
