# From sentence_transformers/cross_encoder/CrossEncoder.py
# https://github.com/UKPLab/sentence-transformers
# Add grad accumulation
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


def _nan_diag(model, loss_value, logits, labels, global_step):
    flagged = False

    if torch.isnan(loss_value) or torch.isinf(loss_value):
        tqdm.write(f"\n[step {global_step}] ❌ LOSS = {loss_value.item()}")
        tqdm.write(
            f"  logits : min={logits.min().item():.4f}  max={logits.max().item():.4f}  "
            f"has_nan={torch.isnan(logits).any().item()}"
        )
        tqdm.write(
            f"  labels : min={labels.min().item():.4f}  max={labels.max().item():.4f}"
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
        print(
            f"[step {global_step}] ❌ NaN/Inf gradients in {len(nan_grad_params)} params:"
        )
        for n in nan_grad_params[:5]:
            print(f"  {n}")
        flagged = True

    if global_step % 10 == 0 or flagged:
        # ✅ added LR — passed in now
        print(
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
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
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

        for epoch in trange(epochs, desc="Epoch", disable=show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                train_dataloader,
                desc="Iteration",
                smoothing=0.05,
                disable=not show_progress_bar,
            ):
                print(f"use_amp is ======>{use_amp}")
                if use_amp:
                    for key, val in features.items():
                        if isinstance(val, torch.Tensor) and torch.isnan(val).any():
                            print(f"[step {global_step}] ❌ NaN in INPUT: {key}")
                    with autocast(enabled=False):
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scaler.scale(loss_value).backward()
                else:
                    print("Logits finite:*****", torch.isfinite(logits).all().item())
                    print("Labels finite:******", torch.isfinite(labels).all().item())
                    print(
                        "Labels min:*****",
                        labels.min().item(),
                        "Labels max:",
                        labels.max().item(),
                    )
                    print("Loss finite:*****", torch.isfinite(loss_value).item())
                    # Check input features for NaN before forward pass
                    for k, v in features.items():
                        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                            print(f"[step {global_step}] ❌ NaN in INPUT: {k}")
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    print("Logits finite:", torch.isfinite(logits).all().item())
                    if isinstance(labels, torch.Tensor):
                        print("Labels finite:", torch.isfinite(labels).all().item())
                        print(
                            "Labels min:",
                            labels.min().item(),
                            "Labels max:",
                            labels.max().item(),
                        )

                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(
                            f"*****[step {global_step}] ⚠️ NaN in logits, skipping backward*******"
                        )
                        optimizer.zero_grad()
                        training_steps += 1
                        global_step += 1
                        continue
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    print("Loss finite:", torch.isfinite(loss_value).item())
                    loss_value.backward()

                if (
                    training_steps + 1
                ) % gradient_accumulation_steps == 0 or training_steps + 1 == len(
                    train_dataloader
                ):
                    found_nan = _nan_diag(
                        self.model, loss_value, logits, labels, global_step
                    )

                    if found_nan:
                        nan_steps.append(global_step)

                        # log LR at every NaN
                        current_lr = optimizer.param_groups[0]["lr"]
                        print(
                            f"[step {global_step}] ⚠️  NaN #{len(nan_steps)} — lr={current_lr:.2e}"
                        )

                        # every 20 NaNs, print distribution
                        if len(nan_steps) % 20 == 0:
                            print(
                                f"  NaN distribution: first={nan_steps[0]}  last={nan_steps[-1]}  total={len(nan_steps)}"
                            )

                        optimizer.zero_grad()
                        training_steps += 1
                        global_step += 1
                        continue
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
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
        # after the epoch loop ends
        print(f"\n=== NaN Summary ===")
        print(f"Total NaN steps: {len(nan_steps)} / {global_step}")
        print(f"First 10: {nan_steps[:10]}")
        print(f"Last 10:  {nan_steps[-10:]}")
