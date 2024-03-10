# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Base Solver class. Specific solver should inherit this class.
Solver takes care of various things, like setting up logging?
As well as running stages.
"""
import logging
from pathlib import Path
import time
import typing as tp

import torch

from .formatter import Formatter
from .logging import LogProgressBar, ResultLogger
from .state import StateManager, AttributeWrapper
from .utils import write_and_rename


StageCallable = tp.Callable
logger = logging.getLogger(__name__)


class BaseSolver:
    def __init__(self, cfg, signature, experiment_name: str, local_folder: str) -> None:
        self.train_cfg = cfg.TRAIN
        self.optim_cfg = self.train_cfg.OPTIMIZER
        self.val_cfg = cfg.VAL
        self.signature = signature
        
        self.stateful = StateManager()
        self.register_stateful('signature', write_only=True)
        self.logger = logger
        self.folder = Path(local_folder)
        self.result_logger = ResultLogger(
            experiment_name, self.logger, local_folder
        )
        
        self.load_from_epoch = cfg.LOAD_FROM_EPOCH
        self.save_interval = self.train_cfg.SAVE_INTERVAL

        self._current_stage: tp.Optional[str] = None
        self._current_formatter: tp.Optional[Formatter] = None
        self._start_epoch()
        self.history: tp.List[tp.Dict[str, tp.Any]] = []
        self.checkpoints_list: tp.Dict[str, str] = {}
        self.register_stateful('history', 'checkpoints_list')
        
        self.max_trial_nums = cfg.FAULT_TOLERANCE
        self._retry_count = 0

    def _start_epoch(self) -> None:
        self._pending_metrics: tp.Dict[str, tp.Any] = {}
        
    @property
    def load_path(self) -> tp.Optional[str]:
        key = str(self.load_from_epoch)
        if key in self.checkpoints_list:
            return self.checkpoints_list[key]
        
    @property
    def checkpoint_metrics_repr(self) -> tp.Optional[str]:
        raise NotImplementedError()

    @property
    def checkpoint_path(self) -> Path:
        if self.checkpoint_metrics_repr is not None:
            path = self.folder / f'epoch_{self.epoch - 1}_{self.checkpoint_metrics_repr}.th'
        else:
            path = self.folder / f'epoch_{self.epoch - 1}.th'
        self.checkpoints_list[str(self.epoch - 1)] = str(path)
        return path

    @property
    def epoch(self) -> int:
        return len(self.history) + 1

    def init_wandb(self, **kwargs):
        """Initialize Wandb logging from Dora xp.
        See `flashy.logging.ResultLogger.init_wandb` for details

        Args:
            with_media_logging (bool): Whether to also log media to Wandb. Default: True
            project (str): Optional wandb project name
            name (str): Optional name for the experiment
            group (str): Optional group for the experiment
            kwargs: Additional arguments for :class:`~flashy.loggers.wandb.WandbLogger` initialization
        """
        self.result_logger.init_wandb(**kwargs)

    def _check_in_stage(self):
        if self._current_stage is None:
            raise RuntimeError("This function can only be called from inside a stage.")

    def log_progress(self, stage_name: str, iterable: tp.Iterable,
                     total: tp.Optional[int] = None, updates: int = 5, **kwargs) -> LogProgressBar:
        """See `flashy.logging.ResultLogger.get_log_progress_bar` for details"""
        return self.result_logger.get_log_progress_bar(
            stage_name, iterable, total=total, updates=updates,
            step=self.epoch, step_name='epoch', formatter=self.formatter, **kwargs)

    def log_hyperparams(self, params: dict, metrics: tp.Optional[dict] = None):
        """See `flashy.logging.ResultLogger.log_hyperparams` for details"""
        self.result_logger.log_hyperparams(params, metrics)

    def log_metrics(self, stage_name: str, metrics: dict, formatter: tp.Optional[Formatter] = None):
        """
        Log metrics for a given stage. Note that the overall metrics for a stage ran
        with `run_stage` are automatically logged from the returned dict of metrics.
        You might want however to log other metrics with a different stage name.
        If called from outside a stage, you must pass the Formatter explicitely.

        See `flashy.logging.ResultLogger.log_metrics` for details"""
        if stage_name in self._pending_metrics:
            raise RuntimeError(f"Stage {stage_name} already exist for epoch {self.epoch}")
        self._pending_metrics[stage_name] = metrics
        if formatter is None:
            formatter = self.formatter
        self.result_logger.log_metrics(stage_name, metrics, step=self.epoch, step_name='epoch',
                                       formatter=formatter)

    def log_audio(self, stage_name: str, key: str, audio: tp.Any, sample_rate: int, **kwargs: tp.Any):
        """See `flashy.logging.ResultLogger.log_audio` for details"""
        self.result_logger.log_audio(stage_name, key, audio, sample_rate, self.epoch, **kwargs)

    def log_image(self, stage_name: str, key: str, image: tp.Any, **kwargs: tp.Any):
        """See `flashy.logging.ResultLogger.log_image` for details"""
        self.result_logger.log_image(stage_name, key, image, self.epoch, **kwargs)

    def log_text(self, stage_name: str, key: str, text: str, **kwargs: tp.Any):
        """See `flashy.logging.ResultLogger.log_text` for details"""
        self.result_logger.log_text(stage_name, key, text, self.epoch, **kwargs)

    def register_stateful(self, *args: str, write_only: bool = False):
        """Shortcut around `StateManager.register` method. You can pass any number of
        attribute, included nested attributes and those will be included into the checkpoints
        and automatically restored when `BaseSolver.restore` is called.

        If `write_only` is True, state is only stored and not restored.
        """
        for name in args:
            owner = self
            *path, leaf = name.split(".")
            for part in path:
                owner = getattr(owner, part)
            state_source = AttributeWrapper(owner, leaf)
            self.stateful.register(name, state_source, write_only)

    def state_dict(self):
        return self.stateful.state_dict()

    def load_state_dict(self, state):
        self.stateful.load_state_dict(state)

    def commit(self):
        save_checkpoint = self.epoch % self.save_interval == 0
        self.history.append(self._pending_metrics)
        self._start_epoch()
        if save_checkpoint:
            state = self.state_dict()
            with write_and_rename(self.checkpoint_path) as f:
                torch.save(state, f)
            self.logger.debug("Checkpoint saved to %s", self.checkpoint_path)

    def restore(self) -> bool:
        if self.load_path is None:
            return False
        load_path = Path(self.load_path)
        if not load_path.exists():
            return False
        state = torch.load(load_path, 'cpu')
        self.load_state_dict(state)
        # TODO: Move to StandardSolver when it exists
        # if len(self.history) > 0:
        #     logger.info("Replaying past metrics...")
        #     for epoch, stages in enumerate(self.history):
        #         for stage_name, metrics in stages.items():
        #             formatted_metrics = self.formatter(metrics)
        #             logger.info("%s", default_format_summary(stage_name, formatted_metrics, epoch))

        self.logger.debug("Checkpoint loaded from %s", load_path)
        return True

    def get_formatter(self, stage_name: str) -> Formatter:
        return Formatter()

    @property
    def formatter(self) -> Formatter:
        self._check_in_stage()
        assert self._current_formatter is not None
        return self._current_formatter

    @property
    def current_stage(self) -> str:
        self._check_in_stage()
        assert self._current_stage is not None
        return self._current_stage

    def run_stage(self, stage_name, method, *args, **kwargs):
        assert self._current_stage is None
        self._current_stage = stage_name
        self._current_formatter = self.get_formatter(stage_name)

        begin = time.time()
        try:
            metrics = method(*args, **kwargs)
            if metrics is None:
                metrics = {}
            metrics["duration"] = time.time() - begin
            self.log_metrics(stage_name, metrics)
        except Exception as e:
            self._retry_count += 1
            self.logger.exception(f"TRIAL {self._retry_count}. "
                                  f"Exception encountered at epoch {self.epoch}: {e}")
            metrics = {}
            if self._retry_count > self.max_trial_nums:
                raise RuntimeError("Max number of trials reached!")
        finally:
            self._current_stage = None
            self._current_formatter = None

        return metrics

    def run(self):
        raise NotImplementedError()
