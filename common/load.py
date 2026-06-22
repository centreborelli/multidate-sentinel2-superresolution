from models import MDSelfS2Model
from trainers.utils import load_weights
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio


def read_image(folder, filename):
    """Read a single band image from a TIFF file."""
    with rasterio.open(folder / filename) as src:
        crop = src.read()
        return crop


def load_roi_temporal_bands(roi_path, bands_10m=True, bands_20m=True, reverse_order=True):
    """
    Load 10m and 20m bands from ROI folders in temporal order.

    Parameters:
    -----------
    roi_path : str or Path
        Path to the ROI folder
    bands_10m : bool
        If True, load 10m bands (B02, B03, B04, B08)
    bands_20m : bool
        If True, load 20m bands (B05, B06, B07, B8A, B11, B12)
    reverse_order : bool
        If True, sort by date descending (newer first). Default: True

    Returns:
    --------
    temporal_10m : np.ndarray or None
        Array of shape (num_dates, 4, height_10m, width_10m) if bands_10m=True, else None
    temporal_20m : np.ndarray or None
        Array of shape (num_dates, 6, height_20m, width_20m) if bands_20m=True, else None
    """

    roi_path = Path(roi_path)

    # Get all subdirectories in the ROI folder
    folders = list(roi_path.iterdir())
    folders = [f for f in folders if f.is_dir()]

    # Extract sensing dates from folder names
    sensing_dates = []
    valid_folders = []

    for folder in folders:
        try:
            # Extract date from filename pattern: YYYYMMDDTHHMMSS
            matches = re.findall(r"(\d{8}T\d{6})", folder.name)
            if matches:
                sensing_date = datetime.strptime(matches[0], "%Y%m%dT%H%M%S")
                sensing_dates.append(sensing_date)
                valid_folders.append(folder)
        except ValueError:
            continue

    # Sort by date
    sorted_data = sorted(zip(sensing_dates, valid_folders))
    sensing_dates, valid_folders = zip(*sorted_data)
    sensing_dates = list(sensing_dates)
    valid_folders = list(valid_folders)

    # Reverse if needed (descending order - newest first)
    if reverse_order:
        sensing_dates.reverse()
        valid_folders.reverse()

    # Define bands to load
    bands_10m_list = ["B04.tif", "B03.tif", "B02.tif", "B08.tif"] if bands_10m else []
    bands_20m_list = ["B05.tif", "B06.tif", "B07.tif", "B8A.tif", "B11.tif", "B12.tif"] if bands_20m else []

    # Load temporal data separately for 10m and 20m
    temporal_10m_samples = []
    temporal_20m_samples = []

    for folder, date in zip(valid_folders, sensing_dates):
        images_10m = []
        images_20m = []

        # Load 10m bands
        if bands_10m:
            for band in bands_10m_list:
                img = read_image(folder, band)
                images_10m.append(img[0])
            temporal_10m_samples.append(np.array(images_10m))

        # Load 20m bands
        if bands_20m:
            for band in bands_20m_list:
                img = read_image(folder, band)
                images_20m.append(img[0])
            temporal_20m_samples.append(np.array(images_20m))

    temporal_10m = np.array(temporal_10m_samples) if temporal_10m_samples else None
    temporal_20m = np.array(temporal_20m_samples) if temporal_20m_samples else None

    return temporal_10m, temporal_20m



class LoadModels:

    @staticmethod
    def mdselfsen2():
        model = MDSelfS2Model(input_bands="10m_20m", output_bands="10m_20m", multi_pred=None, padding_mode='reflect')
        file_path = Path(Path(__file__).parent.parent, "_trained_models/mdselfsen2.safetensors")
        load_weights(model, file_path.__str__())
        return model


    @staticmethod
    def mdselfsen2_only10m():
        model = MDSelfS2Model(input_bands="10m", output_bands="10m", multi_pred=None, padding_mode='reflect')
        file_path = Path(Path(__file__).parent.parent, "_trained_models/mdselfsen2_only10m.safetensors")
        load_weights(model, file_path.__str__())
        return model


    @staticmethod
    def sdselfsen2():
        model = MDSelfS2Model(input_bands="10m_20m", output_bands="10m_20m", multi_pred=None, padding_mode='reflect')
        file_path = Path(Path(__file__).parent.parent, "_trained_models/sdselfsen2.safetensors")
        load_weights(model, file_path.__str__())
        return model

    @staticmethod
    def sdselfsen2_olny10m():
        model = MDSelfS2Model(input_bands="10m", output_bands="10m", multi_pred=None, padding_mode='reflect')
        file_path = Path(Path(__file__).parent.parent, "_trained_models/sdselfsen2_only10m.safetensors")
        load_weights(model, file_path.__str__())
        return model