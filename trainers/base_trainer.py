from __future__ import annotations

import ctypes
import functools
import gc
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import schedulefree
import torch
import torch.utils.data
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.augmentation import Augmentation
from trainers import utils
from trainers.base_config import BaseConfig
from trainers.utils import ShuffledDataset

print = functools.partial(print, flush=True)


class BaseTrainer:
    def __init__(self, config: BaseConfig):
        self.config = config

        self.accelerator = Accelerator(gradient_accumulation_steps=config.batch_accum,
                                       mixed_precision=config.mixed_precision)
        self.device = self.accelerator.device
        print(f"Using device {self.device}")

        self.augmentation = Augmentation(rand_linear_scale=config.rand_linear_scale,
                                         rand_const_offset=config.rand_const_offset)

        train_dataset, val_dataset = self._initialize_datasets()
        self._initialize_dataloaders(train_dataset, val_dataset)

        self._initialize_loss()

        self.model = self.config.get_model()

        if config.compile_model:
            self.model = torch.compile(self.model)

        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()

        self.train_loader, self.val_loader, self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.train_loader, self.val_loader, self.model, self.optimizer, self.scheduler)

        self.tensorboard = SummaryWriter(log_dir=self.config.tensorboard_dir)

        self.temp_save = defaultdict(list)

    def _initialize_loss(self):
        raise NotImplementedError

    def _initialize_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.schedule_milestones,
                                                    gamma=self.config.schedule_gamma)


    def _initialize_datasets(self) -> tuple[torch.utils.data.Dataset]:
        raise NotImplementedError

    def _initialize_dataloaders(self, train_dataset, val_dataset):
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.num_workers,
                                       persistent_workers=True if self.config.num_workers > 0 else False,
                                       pin_memory=True)

        self.val_loader = DataLoader(ShuffledDataset(val_dataset), batch_size=self.config.batch_size, shuffle=False,
                                     num_workers=self.config.num_workers,
                                     persistent_workers=True if self.config.num_workers > 0 else False,
                                     pin_memory=True)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validate_epoch(self, epoch):
        raise NotImplementedError

    def _initialize_optimizer(self):
        samples_per_epoch = len(
            self.train_loader) if self.config.limit_train_steps is None else self.config.limit_train_steps
        if self.config.scheduler == "AdamWScheduleFree":
            return schedulefree.AdamWScheduleFree(
                self.model.parameters(),
                lr=self.config.lr * self.accelerator.num_processes,
                warmup_steps=self.config.schedule_warmup_epochs * samples_per_epoch,
            )
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

    def _log_metrics(self, metrics_dict, epoch, step, time):
        print_str = f"{epoch} {step}:"
        for k, m in metrics_dict.items():
            if len(m) > 0:
                print_str += f" {k}:{np.mean(m):.5f} "
                self.tensorboard.add_scalar(f'{k}/{step}', np.mean(m), epoch)

        self.tensorboard.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
        print(print_str + f"in {time}")
        self.tensorboard.flush()

    def train(self):

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        print("Training {}".format(self.config.checkpoint_dir))
        self._save_metadata()
        utils.seed_everything()

        if self.config.resume_checkpoint:
            print(f"resuming from {self.config.resume_checkpoint}")
            print(f"starting from epoch {self.config.start_epoch}")
            self.accelerator.load_state(self.config.resume_checkpoint)

        for epoch in range(self.config.start_epoch, self.config.epochs):


            start_time = time.time()
            metrics_dict = self._train_epoch(epoch)
            self.scheduler.step()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._log_metrics(metrics_dict, epoch, 'train', time.time() - start_time)

            start_time = time.time()
            metrics_dict = self._validate_epoch(epoch)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._log_metrics(metrics_dict, epoch, 'val', time.time() - start_time)
                self._save_checkpoint(epoch)

    def _prepare_augmentation(self, l1c):

        # check l1c number of axis
        if len(l1c.shape) == 5:
            l1c_augm, undo = self.augmentation.do(l1c[:, 0, ...])
            l1c_augm = l1c_augm.unsqueeze(1)

            for j in range(1, l1c.shape[1]):
                new_aug, _ = self.augmentation.do(l1c[:, j, ...])
                l1c_augm = torch.cat((l1c_augm, new_aug.unsqueeze(1)), dim=1)
        else:
            l1c_augm, undo = self.augmentation.do(l1c)

        return l1c_augm, undo

    def _check_step_abort(self, i):
        if self.config.limit_train_steps and i > self.config.limit_train_steps:
            return True
        return False

    def _print_loss(self, i, loss_value):
        if self.config.print_step_loss:
            print(f"step {i} loss: {loss_value}")

    def _save_metadata(self):
        self.config.save_config()

    def _save_checkpoint(self, epoch):
        if epoch % self.config.save_checkpoint_every == 0:
            folder = Path(self.config.checkpoint_dir, f"check_{epoch:03d}").__str__()
            os.makedirs(folder, exist_ok=True)
            self.accelerator.save_state(folder)

    def _save_samples(self, epoch, step, key, sample):
        if self.accelerator.is_main_process and (epoch % self.config.save_images_every == 0):
            if step < self.config.save_images_len:
                # Accumlate samples
                self.temp_save[key].append(sample.cpu().detach().numpy().transpose((0, 2, 3, 1)))
            elif step == self.config.save_images_len:
                # Save samples
                for key, value in self.temp_save.items():
                    np.save(f"{self.config.checkpoint_dir}/{key}", np.concatenate(value))

                self.temp_save = defaultdict(list)