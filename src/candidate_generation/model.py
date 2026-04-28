# Extends sentence_transformers CrossEncoder with gradient accumulation and NaN guard.
# https://github.com/UKPLab/sentence-transformers
import logging
import os
from typing import Callable, Dict, Type

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def _log_step(model, loss_value, logits, labels, global_step):
    flagged = False

    if torch.isnan(loss_value) or torch.isinf(loss_value):
        tqdm.write(f"\n[step {global_step}] loss={loss_value.item()} (NaN/Inf)")
        tqdm.write(
            f"  logits: min={logits.min().item():.4f}  max={logits.max().item():.4f}  "
            f"has_nan={torch.isnan(logits).any().item()}"
        )
        tqdm.write(
            f"  labels: min={labels.min().item():.4f}  max={labels.max().item():.4f}"
        )
        flagged = True

    nan_grad_params, max_grad = [], 0.0
    total_norm_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad
        total_norm_sq += g.norm(2).item() ** 2
        max_grad = max(max_grad, g.abs().max().item())
        if torch.isnan(g).any() or torch.isinf(g).any():
            nan_grad_params.append(name)

    total_norm = total_norm_sq**0.5

    if nan_grad_params:
        logger.warning(f"[step {global_step}] NaN/Inf gradients in {len(nan_grad_params)} params")
        for n in nan_grad_params[:5]:
            logger.warning(f"  {n}")
        flagged = True

    if global_step % 10 == 0 or flagged:
        logger.info(
            f"[step {global_step}] loss={loss_value.item():.5f}  "
            f"grad_norm={total_norm:.4f}  max_grad={max_grad:.4f}"
        )

    return flagged


class CrossEncoder(CrossEncoder):
    def fit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=num_train_steps,
            )

        if loss_fct is None:
            loss_fct = (
                nn.BCEWithLogitsLoss()
                if self.config.num_labels == 1
                else nn.CrossEntropyLoss()
            )

        nan_steps = []
        global_step = 0

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                train_dataloader,
                desc="Iteration",
                smoothing=0.05,
                disable=not show_progress_bar,
            ):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                    scaler.scale(loss_value).backward()
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)

                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning(f"[step {global_step}] NaN in logits, skipping step")
                        optimizer.zero_grad()
                        training_steps += 1
                        global_step += 1
                        continue

                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()

                if (
                    training_steps + 1
                ) % gradient_accumulation_steps == 0 or training_steps + 1 == len(
                    train_dataloader
                ):
                    found_nan = _log_step(
                        self.model, loss_value, logits, labels, global_step
                    )

                    if found_nan:
                        nan_steps.append(global_step)
                        optimizer.zero_grad()
                        training_steps += 1
                        global_step += 1
                        continue

                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                training_steps += 1

                if (
                    evaluator is not None
                    and evaluation_steps > 0
                    and training_steps % evaluation_steps == 0
                ):
                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        training_steps,
                        callback,
                    )
                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(
                    evaluator, output_path, save_best_model, epoch, -1, callback
                )

        if nan_steps:
            logger.warning(f"NaN steps: {len(nan_steps)}/{global_step} total")
