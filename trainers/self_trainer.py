from __future__ import annotations

from collections import defaultdict

import torch
import torch.utils.data

from common.loss import get_loss
from data.l1c_l1b_dataset import L1CToL1BDataset
from trainers.base_config import BaseConfig
from trainers.base_trainer import BaseTrainer


class SelfTrainer(BaseTrainer):
    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _initialize_loss(self):
        self.loss_10m_fn = get_loss(self.config.loss_expr_10m, self.config.warp_padding_mode,
                                    bands="only-10m").to(self.device)
        self.loss_20m_fn = get_loss(self.config.loss_expr_20m, self.config.warp_padding_mode,
                                    bands="only-20m").to(self.device)

    def _initialize_datasets(self) -> tuple[torch.utils.data.Dataset]:
        train_dataset = L1CToL1BDataset(**self.config.dataset_train_params)
        val_dataset = L1CToL1BDataset(**self.config.dataset_val_params)
        return train_dataset, val_dataset

    def _prepare_loss(self, y, band_l1b, band_grid, band_mask):

        band_loss = {}
        band_yy = {}

        if self.config.output_bands == '10m':
            band_loss['10m'], band_yy['10m'] = self.loss_10m_fn(y, band_l1b['10m'], band_grid['10m'], band_mask['10m'])
            loss = band_loss['10m']
        elif self.config.output_bands == '20m':
            band_loss['20m'], band_yy['20m'] = self.loss_20m_fn(y, band_l1b['20m'], band_grid['20m'], band_mask['20m'])
            loss = band_loss['20m']
        elif self.config.output_bands == '10m_20m':
            band_loss['10m'], band_yy['10m'] = self.loss_10m_fn(y[:, :4, ...], band_l1b['10m'],
                                                                band_grid['10m'], band_mask['10m'])
            band_loss['20m'], band_yy['20m'] = self.loss_20m_fn(y[:, 4:, ...], band_l1b['20m'],
                                                                band_grid['20m'], band_mask['20m'])
            #loss = (band_loss['10m'] + band_loss['20m']) / 2.0
            loss = (0.75 * band_loss['10m']) + (0.25 * band_loss['20m'])

        return loss, band_loss, band_yy

    ## Assumes that y is a list of outputs, and y[-1] is the single output, and y[0] is the multi date output
    def _prepare_loss_multi(self, y, band_l1b, band_grid, band_mask):
        loss_accum = []
        band_loss_accum = []
        band_yy_accum = []

        for _y in y:
            loss, band_loss, band_yy = self._prepare_loss(_y, band_l1b, band_grid, band_mask)
            loss_accum.append(loss)
            band_loss_accum.append(band_loss)
            band_yy_accum.append(band_yy)

        loss = sum(loss_accum) / len(loss_accum)

        band_yy = band_yy_accum[-1]
        band_loss = band_loss_accum[-1]
        y_single = y[-1]

        band_yy_multi = band_yy_accum[0]
        band_loss_multi = band_loss_accum[0]
        y_multi = y[0]

        return loss, band_loss, band_yy, y_single, band_loss_multi, band_yy_multi, y_multi

    def _prepare_data(self, sample):

        band_l1c, band_l1b, band_grid, band_mask = {}, {}, {}, {}

        # prepare data
        band_l1c['10m'], band_l1b['10m'], band_grid['10m'], band_mask['10m'] = sample[:4]
        band_l1c['20m'], band_l1b['20m'], band_grid['20m'], band_mask['20m'] = sample[4:8]

        return band_l1c, band_l1b, band_grid, band_mask

    def _internal_save_sample(self, phase, epoch, step, y, band_l1c, band_l1b, band_yy, band_mask, y_multi=None,
                              band_yy_multi=None):

        def _save_output(_y, name):
            if self.config.output_bands == '10m_20m':
                self._save_samples(epoch, step, f'{phase}-{epoch}-{name}_10m', _y[:, :4, ...])
                self._save_samples(epoch, step, f'{phase}-{epoch}-{name}_20m', _y[:, 4:, ...])
            else:
                self._save_samples(epoch, step, f'{phase}-{epoch}-{name}_{self.config.output_bands}', _y)

        _save_output(y, "imr")
        if y_multi is not None:
            _save_output(y_multi, "imr_multi")

        def _get_single_image(image):
            if len(image.shape) == 5:
                return image[:, 0, ...]
            return image


        if '10m' in self.config.input_bands:
            self._save_samples(epoch, step, f'{phase}-{epoch}-im1_10m', _get_single_image(band_l1c['10m']))
        if '10m' in self.config.input_bands:
            self._save_samples(epoch, step, f'{phase}-{epoch}-im1_20m', _get_single_image(band_l1c['20m']))

        def _save_band_info_if_available(dict_tensor, name, key):
            if dict_tensor is not None and key in dict_tensor and dict_tensor[key] is not None:
                self._save_samples(epoch, step, f'{phase}-{epoch}-{name}_{key}', _get_single_image(dict_tensor[key]))

        out_bands = self.config.output_bands.split("_")
        for band in out_bands:
            _save_band_info_if_available(band_l1b, "im2", band)
            _save_band_info_if_available(band_mask, "mask", band)
            _save_band_info_if_available(band_yy, "imw", band)
            _save_band_info_if_available(band_yy_multi, "imw_multi", band)

    def _prepare_augmentation(self, band_l1c):

        if not self.config.augmentation:
            return band_l1c, lambda x: x

        if self.config.output_bands not in self.config.input_bands:
            raise ValueError("Invalid configuration for augmentation")

        band_undo = {}
        band_l1c_aug = {}
        for k, v in band_l1c.items():
            band_l1c_aug[k], band_undo[k] = super()._prepare_augmentation(v)

        if self.config.output_bands == '10m_20m':
            undo = lambda x: torch.cat((band_undo['10m'](x[:, :4]), band_undo['20m'](x[:, 4:])), dim=1)
        elif self.config.output_bands == '10m':
            undo = lambda x: band_undo['10m'](x)
        else:
            undo = lambda x: band_undo['20m'](x)

        return band_l1c_aug, undo

    @staticmethod
    def _append_metrics(metrics_track, loss, band_loss, band_loss_multi):

        # general loss
        metrics_track['loss'].append(loss.item())
        # losses components
        for k, v in band_loss.items():
            metrics_track[f'loss_{k}'].append(v.item())

        if band_loss_multi is not None:
            for k, v in band_loss_multi.items():
                metrics_track[f'loss_{k}_multi'].append(v.item())

    def _forward(self, band_l1c, grid_temp_align=None):
        with self.accelerator.autocast():
            if self.config.input_bands == '10m':
                y = self.model(band_l1c['10m'], grid_temp_align=grid_temp_align)
            elif self.config.input_bands == '20m':
                y = self.model(band_l1c['20m'], grid_temp_align=grid_temp_align)
            elif self.config.input_bands == '10m_20m':
                y = self.model(band_l1c['10m'], band_l1c['20m'], grid_temp_align=grid_temp_align)
            return y

    def _train_epoch(self, epoch):

        metrics_track = defaultdict(list)
        self.model.train()
        self.optimizer.train()

        for i, sample in enumerate(self.train_loader):

            if self._check_step_abort(i):
                break

            band_l1c, band_l1b, band_grid, band_mask = self._prepare_data(sample)

            grid_temp_align = sample[8]

            with self.accelerator.accumulate(self.model):

                band_l1c, undo = self._prepare_augmentation(band_l1c)

                if self.config.use_alignment:
                    y = self._forward(band_l1c, grid_temp_align)
                else:
                    y = self._forward(band_l1c)

                y_multi, band_yy_multi, band_loss_multi = None, None, None
                if isinstance(y, tuple) or isinstance(y, list):
                    y = [undo(_y) for _y in y]
                    loss, band_loss, band_yy, y, band_loss_multi, band_yy_multi, y_multi = \
                        self._prepare_loss_multi(y, band_l1b, band_grid, band_mask)
                else:
                    y = undo(y)
                    loss, band_loss, band_yy = self._prepare_loss(y, band_l1b, band_grid, band_mask)

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self._print_loss(i, loss.item())
                self._append_metrics(metrics_track, loss, band_loss, band_loss_multi)
                self._internal_save_sample("train", epoch, i, y, band_l1c, band_l1b, band_yy, band_mask, y_multi,
                                           band_yy_multi)

        return metrics_track

    def _validate_epoch(self, epoch):

        metrics_track = defaultdict(list)
        self.model.eval()
        self.optimizer.eval()

        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):

                if self._check_step_abort(i):
                    break

                # prepare data
                band_l1c, band_l1b, band_grid, band_mask = self._prepare_data(sample)
                grid_temp_align = sample[8]

                if self.config.use_alignment:
                    y = self._forward(band_l1c, grid_temp_align)
                else:
                    y = self._forward(band_l1c)

                y_multi, band_yy_multi, band_loss_multi = None, None, None
                if isinstance(y, tuple) or isinstance(y, list):
                    loss, band_loss, band_yy, y, band_loss_multi, band_yy_multi, y_multi = \
                        self._prepare_loss_multi(y, band_l1b, band_grid, band_mask)
                else:
                    loss, band_loss, band_yy = self._prepare_loss(y, band_l1b, band_grid, band_mask)

                self._print_loss(i, loss.item())
                self._append_metrics(metrics_track, loss, band_loss, band_loss_multi)
                self._internal_save_sample("val", epoch, i, y, band_l1c, band_l1b, band_yy, band_mask, y_multi,
                                           band_yy_multi)

        return metrics_track
