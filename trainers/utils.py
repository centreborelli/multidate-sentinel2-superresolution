import os
import random
from collections import OrderedDict

import numpy as np
import safetensors.torch
import torch
from torch.utils.data import Dataset


def load_weights(model, checkpoint_path):
    checkpoint = safetensors.torch.load_file(checkpoint_path)
    updated_dict = OrderedDict()
    for key, value in checkpoint.items():
        new_key = key.replace('_orig_mod.', '', 1)
        updated_dict[new_key] = value

    checkpoint = updated_dict
    model.load_state_dict(checkpoint)
    return model


class ShuffledDataset(Dataset):
    def __init__(self, data_source):
        self.data_source = data_source
        np.random.seed(42)
        self.indices = np.random.permutation(len(data_source))

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        return self.data_source[self.indices[idx]]


#### General purpose
def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
