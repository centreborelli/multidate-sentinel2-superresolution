import random
from pathlib import Path
from typing import Literal

import torch.nn.functional as F
import h5py
import hdf5plugin  # noqa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from local import EnvironmentSettings
from scipy.ndimage import binary_erosion



# Past frames
class L1CToL1BDataset(Dataset):

    def __init__(self, data_type: Literal['train', 'val', 'test', 'train-val'] = 'train',
                 normalize: int = 256,
                 aux_frames=-1,
                 aux_fill: Literal[None, 'zeros', 'random', 'mirror'] = None,
                 aux_cap_range: bool = True,
                 aux_selection_strategy: Literal['random', 'random_fixed', 'closest_dates'] = 'random_fixed',
                 randomness_closest_dates: float = 0,
                 aux_source: Literal['past', 'future', 'both'] = 'both',
                 aux_orbit_source: Literal['mixed', 'nonmixed'] = 'mixed',
                 limit_samples: int = None,
                 shift_aux=False):

        self.normalize = normalize
        self.aux_frames = aux_frames
        self.shift_aux = shift_aux

        if data_type in ['train', 'val', 'train-val']:
            self.path = Path(EnvironmentSettings.MAIN_DATASET / 'train.h5')
        elif data_type == 'test':
            self.path = Path(EnvironmentSettings.MAIN_DATASET / 'test.h5')
        else:
            raise ValueError(f'Unknown data type {data_type}')

        precomputed_path = EnvironmentSettings.MAIN_DATASET / 'info'

        # Assumes a ref dataframe already randomized
        self.ref_df = pd.read_parquet(precomputed_path / f"ref-{data_type}.parquet", engine='pyarrow')

        if limit_samples is not None:
            self.ref_df = self.ref_df[:limit_samples]

        if data_type == 'train' or data_type == 'val':
            self.aux_df = pd.read_parquet(precomputed_path / f"aux-random-train-val.parquet", engine='pyarrow')
            self.aux_df = self.aux_df[self.aux_df['sample_id'].isin(self.ref_df['sample_id'].unique())]
        else:
            self.aux_df = pd.read_parquet(precomputed_path / f"random-{data_type}2.parquet", engine='pyarrow')

        if aux_source == 'past':
            self.aux_df = self.aux_df[self.aux_df['source'] == 'p']
        elif aux_source == 'future':
            self.aux_df = self.aux_df[self.aux_df['source'] == 'f']

        if aux_orbit_source == 'nonmixed':
            ref_df_subset = self.ref_df[['sample_id', 'orbit']].drop_duplicates()
            self.aux_df = self.aux_df.merge(ref_df_subset, on=['sample_id', 'orbit'], how='inner').reset_index(drop=True)

        if aux_selection_strategy == 'closest_dates':
            merged_df = self.aux_df.merge(self.ref_df[['sample_id', 'date']], on='sample_id', suffixes=('', '_ref'))
            merged_df['date_diff'] = (merged_df['date'] - merged_df['date_ref']).abs()
            self.aux_df = merged_df.sort_values(by=['sample_id', 'date_diff'])

        self.aux_df.set_index(['sample_id', 'source', 'date_index'], inplace=True)
        self.randomness_closest_dates = randomness_closest_dates
        self.aux_fill = aux_fill
        self.aux_cap_range = aux_cap_range
        self.aux_selection_strategy = aux_selection_strategy
        self.data_h5 = None
        self.length = len(self.ref_df)

        if aux_frames == -1:
            with h5py.File(self.path, 'r') as f:
                self.aux_frames = f['pastl1c'].shape[1] + f['futl1c'].shape[1]

    def _ensure_open(self):
        if self.data_h5 is None:
            self.data_h5 = h5py.File(self.path, 'r')

    def __getitem__(self, index):
        self._ensure_open()

        sample_index = self.ref_df.loc[index]['sample_id']

        l1c = torch.from_numpy(self.data_h5['l1c'][sample_index])
        l1b = torch.from_numpy(self.data_h5['l1b'][sample_index])
        grid = torch.from_numpy(self.data_h5['grid'][sample_index])
        mask = torch.from_numpy(self.data_h5['mask'][sample_index])

        l1c_20m = torch.from_numpy(self.data_h5['l1c_20m'][sample_index])
        l1b_20m = torch.from_numpy(self.data_h5['l1b_20m'][sample_index])
        grid_20m = torch.from_numpy(self.data_h5['grid_20m'][sample_index])
        mask_20m = torch.from_numpy(self.data_h5['mask_20m'][sample_index])

        alignment_info = np.array([0, 0, 1])
        date_info = int(pd.Timestamp(self.ref_df.loc[index]['date']).tz_localize(None).timestamp())
        orbit_info = [self.ref_df.loc[index]['orbit']]

        target_frames = 0
        if self.aux_frames is not None:

            l1c = l1c.unsqueeze(0)
            l1c_20m = l1c_20m.unsqueeze(0)
            alignment_info = [alignment_info]
            date_info = [date_info]

            aux_df_selected = self.aux_df.loc[sample_index]
            indexes_aux = aux_df_selected.index.values

            if self.aux_selection_strategy == 'random':
                np.random.shuffle(indexes_aux)
            elif self.aux_selection_strategy == 'closest_dates' and self.randomness_closest_dates > 0:
                indexes_aux = shuffle_with_control(indexes_aux, self.randomness_closest_dates)

            if isinstance(self.aux_frames, list):
                if self.aux_cap_range:
                    target_frames = random.choice([v for v in self.aux_frames if v <= len(indexes_aux)])
                else:
                    target_frames = min(random.choice(self.aux_frames), len(indexes_aux))
            else:
                target_frames = min(self.aux_frames, len(indexes_aux))

            fill_frames = 0
            if self.aux_fill:
                fill_frames = max(self.aux_frames, target_frames) - target_frames

            if target_frames > 0:
                past_l1c = torch.from_numpy(self.data_h5['pastl1c'][sample_index])
                fut_l1c = torch.from_numpy(self.data_h5['futl1c'][sample_index])
                past_l1c_20m = torch.from_numpy(self.data_h5['pastl1c_20m'][sample_index])
                fut_l1c_20m = torch.from_numpy(self.data_h5['futl1c_20m'][sample_index])

                merge = [l1c[0]]
                merge_20m = [l1c_20m[0]]

                def _merge_aux(source, date_index):

                    selected = aux_df_selected.loc[(source, date_index)]
                    alignment_info.append(selected[['dy', 'dx', 'score']].values)
                    date_info.append(int(selected['date'].tz_localize(None).timestamp()))
                    orbit_info.append(selected['orbit'])

                    if source == 'p':
                        merge.append(past_l1c[date_index])
                        merge_20m.append(past_l1c_20m[date_index])
                    elif source == 'f':
                        merge.append(fut_l1c[date_index])
                        merge_20m.append(fut_l1c_20m[date_index])

                for i in range(target_frames):
                    _merge_aux(*indexes_aux[i])

                if self.aux_fill == 'zeros':
                    for i in range(fill_frames):
                        merge.append(torch.zeros_like(l1c[0]))
                        merge_20m.append(torch.zeros_like(l1c_20m[0]))
                elif self.aux_fill == 'random':
                    for i in range(fill_frames):
                        random_aux = np.random.randint(0, len(indexes_aux))
                        _merge_aux(*indexes_aux[random_aux])
                elif self.aux_fill == 'mirror':
                    s_aux = 0
                    for i in range(fill_frames):
                        _merge_aux(*indexes_aux[s_aux % len(indexes_aux)])
                        s_aux += 1

                l1c = torch.stack(merge, axis=0)
                l1c_20m = torch.stack(merge_20m, axis=0)

        l1c = (l1c / self.normalize).float()
        l1b = (l1b / self.normalize).float()
        l1c_20m = (l1c_20m / self.normalize).float()
        l1b_20m = (l1b_20m / self.normalize).float()

        mask = apply_erosion(mask[0], iterations=5)
        mask = mask.unsqueeze(0).repeat(4, 1, 1)
        mask_20m = apply_erosion(mask_20m[0], iterations=3)
        mask_20m = mask_20m.unsqueeze(0).repeat(6, 1, 1)

        # mask = mask.repeat(4, 1, 1)
        # mask_20m = mask_20m.repeat(6, 1, 1)

        grid_temp_align = []
        for i, info in enumerate(alignment_info):
            shift_y, shift_x, score = info
            theta = torch.tensor([
                [1, 0, -2 * shift_x / l1c[i].shape[2]],
                [0, 1, -2 * shift_y / l1c[i].shape[1]]
            ], dtype=torch.float32).unsqueeze(0)
            grid_temp_align.append(F.affine_grid(theta, l1c[i].unsqueeze(0).size(), align_corners=False))

        grid_temp_align = torch.cat(grid_temp_align, dim=0)

        metadata = {
            'target_frames': target_frames,
            'date_info': np.array(date_info),
            'alignment_info': np.array(alignment_info, dtype=np.float32),
            'orbit_info': np.array(orbit_info)
        }


        if self.shift_aux:
            l1c = integer_shift(l1c, metadata['alignment_info'])
            l1c_20m = integer_shift(l1c_20m, metadata['alignment_info']/2)

        return l1c, l1b, grid, mask, l1c_20m, l1b_20m, grid_20m, mask_20m, grid_temp_align, metadata


    def __len__(self):
        return self.length

    def __del__(self):
        if hasattr(self, 'data_h5') and self.data_h5 is not None:
            try:
                self.data_h5.close()
            except Exception:
                pass
            self.data_h5 = None


def shift_tensor(x, dy, dx):
    C, H, W = x.shape

    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)

    start_y = max(0, -dy)
    start_x = max(0, -dx)

    out = x_padded[:, start_y:start_y + H, start_x:start_x + W]

    return out


def integer_shift(image_series, alignment_info):
    shifted_series = []
    for i in range(len(alignment_info)):
        dy, dx, _ = alignment_info[i]
        dy = int(dy)
        dx = int(dx)
        shifted_image = shift_tensor(image_series[i], dy, dx)
        shifted_series.append(shifted_image)
    return torch.stack(shifted_series)


def shuffle_with_control(array, randomness=0.5):
    n = len(array)

    num_swaps = int(randomness * n)

    for _ in range(num_swaps):
        i, j = np.random.randint(0, n, size=2)
        array[i], array[j] = array[j], array[i]

    return array

def apply_erosion(mask, iterations=1):
    structure = np.ones((3, 3))

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    eroded_mask = binary_erosion(mask, structure=structure, iterations=iterations)
    eroded_mask = torch.tensor(eroded_mask, dtype=torch.bool)
    return eroded_mask

