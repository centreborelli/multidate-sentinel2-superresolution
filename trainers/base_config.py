import inspect
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from local import EnvironmentSettings
from models import MDSelfS2Model


class BaseConfig:
    def __init__(
            self,
            exp_name=None,

            model_class=MDSelfS2Model,
            model_params=None,

            input_bands='10m_20m',
            output_bands='10m_20m',
            dataset_train_params=None,
            dataset_val_params=None,
            augmentation=True,

            resume_checkpoint_date=None,
            rand_const_offset=0.0,
            rand_linear_scale=0.4,
            warp_padding_mode="zeros",
            loss_expr_20m="none",
            loss_expr_10m="accurate",
            use_alignment=False,
            epochs=100,
            lr=0.0001,
            scheduler="AdamWScheduleFree",
            schedule_warmup_epochs=2,
            schedule_steps_milestones=None,
            schedule_gamma=0.1,
            num_workers=0,
            batch_size=4,
            batch_accum=1,
            mixed_precision=None,
            compile_model=False,

            save_checkpoint_every=1,
            save_images_len=4,
            save_images_every=5,

            print_step_loss=False,
            limit_train_steps=None,
            limit_val_steps=None,
    ):

        if exp_name is None:
            frame = inspect.currentframe()
            exp_name = frame.f_back.f_code.co_name

        ## Model parameters
        self.exp_name = exp_name
        self.model_class = model_class
        self.model_params = model_params or dict()

        # Dataset / input output parameters
        self.input_bands = input_bands
        self.output_bands = output_bands
        self.augmentation = augmentation

        if dataset_train_params is None:
            dataset_train_params = dict()
        self.dataset_train_params = dataset_train_params

        if dataset_val_params is None:
            dataset_val_params = dict()
        self.dataset_val_params = dataset_val_params

        self.rand_const_offset = rand_const_offset
        self.rand_linear_scale = rand_linear_scale
        self.warp_padding_mode = warp_padding_mode
        self.loss_expr_20m = loss_expr_20m
        self.loss_expr_10m = loss_expr_10m
        self.epochs = epochs
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_accum = batch_accum
        self.mixed_precision = mixed_precision
        self.compile_model = compile_model
        self.use_alignment = use_alignment

        # constants
        self.upsampling_20m = "nearest"
        self.scheduler = scheduler
        self.schedule_warmup_epochs = schedule_warmup_epochs
        self.schedule_milestones = schedule_steps_milestones
        if isinstance(schedule_steps_milestones, int):
            self.schedule_milestones = [x for x in range(schedule_steps_milestones, self.epochs, schedule_steps_milestones)]
        if self.schedule_milestones is None:
            self.schedule_milestones = []

        self.schedule_gamma = schedule_gamma

        self.start_epoch = 0
        self.checkpoint_dir = None
        self.tensorboard_dir = None
        self.resume_checkpoint = None

        if resume_checkpoint_date:
            folder_name = f"{resume_checkpoint_date}_{self.exp_name}"
            self.checkpoint_dir = EnvironmentSettings.WORK_DIR.__str__() + '/trainings/' + folder_name
            self.tensorboard_dir = EnvironmentSettings.WORK_DIR.__str__() + '/tensorboard/' + folder_name
            epoch, chk_epoch = find_highest_checkpoint_number(self.checkpoint_dir)

            if epoch >= 0:
                self.start_epoch = epoch + 1
                self.resume_checkpoint = self.checkpoint_dir + '/' + chk_epoch
            else:
                raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")

        else:
            folder_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{self.exp_name}"
            self.checkpoint_dir = EnvironmentSettings.WORK_DIR.__str__() + '/trainings/' + folder_name
            self.tensorboard_dir = EnvironmentSettings.WORK_DIR.__str__() + '/tensorboard/' + folder_name

            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.save_checkpoint_every = save_checkpoint_every
        self.save_images_len = save_images_len * self.batch_accum
        self.save_images_every = save_images_every

        # debug constants
        self.print_step_loss = print_step_loss
        self.limit_train_steps = limit_train_steps
        self.limit_val_steps = limit_val_steps

    def get_model(self):
        return self.model_class(**self.model_params)

    def save_config(self):

        metadata = dict()
        for k, v in self.__dict__.items():
            # special cases
            if k == 'bands':
                metadata[k] = list(v)
            elif k == 'model_class':
                metadata['model_class'] = v.__name__
                metadata['model_module'] = v.__module__
            else:
                metadata[k] = v

        metadata["env"] = {
            # "rev": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip(),
            # "dirty": subprocess.check_output(["git", "describe", "--dirty", "--always"]).decode("ascii").strip(),
            "hostname": platform.node(),
        }

        with open(f"{self.checkpoint_dir}/metadata.json", "w") as f:
            f.write(json.dumps(metadata, indent=4))


def find_highest_checkpoint_number(checkpoint_dir):
    checkpoint_files = list(Path(checkpoint_dir).glob('check_*'))

    highest_n = -1
    folder = None
    for file in checkpoint_files:
        if file.is_dir():
            n = int(file.name.split('_')[-1])
            if n > highest_n:
                highest_n = n
                folder = file.name

    return highest_n, folder
