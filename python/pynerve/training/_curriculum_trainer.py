"""Topological curriculum trainer."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
from torch import nn

from .._validation import validate_finite_scalar as _finite_scalar
from .curriculum import (
    CurriculumConfig,
    TopologicalComplexityCalculator,
    TopologicalCurriculumSampler,
)


class TopologicalCurriculumTrainer:
    """Train a model with a topology-based curriculum learning schedule.

    Manages stage advancement, dataloader creation, and training/validation loops.

    :param model: The PyTorch model to train.
    :param config: Curriculum configuration.
    :param stage_advancement_criterion: How stages advance (``"epoch"``, ``"performance"``, or ``"manual"``).
    :param seed: Random seed for reproducibility.
    :raises ValueError: If ``stage_advancement_criterion`` is invalid.
    """

    def __init__(
        self,
        model: nn.Module,
        config: CurriculumConfig,
        stage_advancement_criterion: str = "epoch",
        seed: int | None = None,
    ):
        """Initialize the curriculum trainer.

        :param model: The PyTorch model to train.
        :param config: Curriculum configuration.
        :param stage_advancement_criterion: How stages advance: ``"epoch"``,
            ``"performance"``, or ``"manual"``.
        :param seed: Random seed for reproducibility.
        :raises ValueError: If ``stage_advancement_criterion`` is invalid.
        """
        if stage_advancement_criterion not in {"epoch", "performance", "manual"}:
            raise ValueError("stage_advancement_criterion is invalid")

        self.model = model
        self.config = config
        self.criterion = stage_advancement_criterion
        self.seed = seed
        self.current_stage = 0
        self.epoch = 0
        self.validation_scores: list[float] = []

    def create_dataloader(
        self,
        dataset: Dataset[Any],
        diagrams: list[torch.Tensor],
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> DataLoader[Any]:
        """Create a DataLoader with curriculum-ordered sampling.

        :param dataset: The PyTorch dataset.
        :param diagrams: Persistence diagrams for each sample.
        :param batch_size: Number of samples per batch.
        :param num_workers: Number of data loader workers.
        :returns: A DataLoader with ``TopologicalCurriculumSampler``.
        :raises ValueError: If ``batch_size`` is not positive or ``num_workers`` is negative.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        sampler = TopologicalCurriculumSampler(
            dataset, diagrams, self.config, self.current_stage, seed=self.seed
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def should_advance_stage(self, validation_score: float | None = None) -> bool:
        """Determine whether the curriculum should advance to the next stage.

        :param validation_score: Optional validation score for performance-based advancement.
        :returns: ``True`` if stage advancement criteria are met.
        """
        if self.current_stage >= self.config.num_stages - 1:
            return False
        if self.criterion == "epoch":
            return self.epoch > 0 and self.epoch % self.config.warmup_epochs == 0
        if self.criterion == "performance":
            if validation_score is not None:
                validation_score = _finite_scalar(validation_score, "validation_score")
                self.validation_scores.append(validation_score)
            if len(self.validation_scores) < 3:
                return False
            recent = self.validation_scores[-3:]
            return max(recent) - min(recent) < 0.01
        return False

    def advance_stage(self) -> None:
        """Advance to the next curriculum stage (clamped to max)."""
        self.current_stage = min(self.current_stage + 1, self.config.num_stages - 1)

    def train_epoch(self, dataloader: DataLoader[Any], optimizer: Any, loss_fn: Any) -> float:
        """Train the model for one epoch.

        :param dataloader: A DataLoader providing training batches.
        :param optimizer: The PyTorch optimizer.
        :param loss_fn: The loss function.
        :returns: The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for data, target in dataloader:
            optimizer.zero_grad(set_to_none=True)
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

        self.epoch += 1
        return total_loss / max(num_batches, 1)

    def fit(
        self,
        train_dataset: Dataset[Any],
        train_diagrams: list[torch.Tensor],
        val_dataset: Dataset[Any] | None = None,
        val_diagrams: list[torch.Tensor] | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: Any = None,
        loss_fn: Any = None,
    ) -> list[dict[str, Any]]:
        """Run the full curriculum training loop.

        :param train_dataset: The training dataset.
        :param train_diagrams: Persistence diagrams for the training set.
        :param val_dataset: Optional validation dataset.
        :param val_diagrams: Optional validation persistence diagrams.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param optimizer: PyTorch optimizer (defaults to ``Adam``).
        :param loss_fn: Loss function (defaults to ``CrossEntropyLoss``).
        :returns: A list of per-epoch training records.
        :raises ValueError: If ``epochs`` is negative.
        """
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters())
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        history = []
        for epoch in range(epochs):
            train_loader = self.create_dataloader(train_dataset, train_diagrams, batch_size)
            train_loss = self.train_epoch(train_loader, optimizer, loss_fn)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "stage": self.current_stage,
            }

            validation_score = None
            if val_dataset is not None and val_diagrams is not None:
                validation_score = self.evaluate(val_dataset, val_diagrams)
                record["validation_score"] = validation_score

            if self.should_advance_stage(validation_score):
                self.advance_stage()
                record["advanced_stage"] = self.current_stage

            history.append(record)

        return history

    def evaluate(self, dataset: Dataset[Any], diagrams: list[torch.Tensor]) -> float:
        """Evaluate the model on a validation dataset.

        Computes the mean topological complexity of validation diagrams.

        :param dataset: The validation dataset.
        :param diagrams: Validation persistence diagrams.
        :returns: Mean complexity score across the validation set.
        """
        del dataset

        self.model.eval()
        calculator = TopologicalComplexityCalculator(self.config.persistence_threshold)
        complexities = calculator.compute_batch_complexity(diagrams, self.config.complexity_measure)
        if not complexities:
            return 0.0
        return float(np.mean(complexities))
