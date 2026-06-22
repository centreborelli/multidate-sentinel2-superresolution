from pathlib import Path

import torch


class EnvironmentSettings:
    WORK_DIR = Path(Path(__file__).parent, "./_workspace")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAIN_DATASET = Path(Path(__file__).parent, "_data/md_l1c_l1b_data")
    OPENSR_PRED = Path(Path(__file__).parent, "_workspace/opensr_pred")
    OPENSR_METRICS = Path(Path(__file__).parent, "./_workspace/opensr_metrics")
    OPENSR_SDDATA = Path(Path(__file__).parent, "./_data/opensr_original")
    OPENSR_MDDATA = Path(Path(__file__).parent, "./_data/opensr_md_v2")
