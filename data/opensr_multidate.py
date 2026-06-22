import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from skimage.transform import resize


def resize_images_to_same_shape(images, target_shape):
    return [resize(img, target_shape, order=0, preserve_range=True, anti_aliasing=False) for img in images]


def read_image(folder, filename):
    with rasterio.open(folder / filename) as src:
        crop = src.read()
        return crop


def load_opensr_multidate(root, dname, roi, mode='l1c', selected_bands=None):

    path = Path(root, f"{mode}/{dname}/{roi}")

    temporal_samples = []

    folders = list(path.iterdir())

    if (path / "ref.txt").exists():
        with open(path / "ref.txt", 'r') as f:
            reference_name = f.read().strip()
        folders = [f for f in folders if f.name != reference_name]

    folders = [f for f in folders if f.is_dir() and any(f.iterdir())]  # Check if folder is not empty

    sensing_dates = []
    for folder in folders:
        matches = re.findall(r"\d{8}T\d{6}", folder.stem)
        sensing_date = datetime.strptime(matches[0], "%Y%m%dT%H%M%S")
        sensing_dates.append(sensing_date)

    # Sort folders by sensing date
    folders = [folder for _, folder in sorted(zip(sensing_dates, folders))]

    folders = folders[::-1]  # Reverse to have most recent date first

    for folder in folders:
        bands = ["B01.tif", "B02.tif", "B03.tif", "B04.tif", "B05.tif", "B06.tif", "B07.tif", "B08.tif", "B8A.tif",
                 "B09.tif", "B11.tif", "B12.tif"]

        if selected_bands is not None:
            bands = [bands[i] for i in selected_bands]

        images = [read_image(folder, band)[0] for band in bands]

        if selected_bands is None:
            result = resize_images_to_same_shape(images, images[1].shape)
        else:
            result = images

        temporal_samples.append(np.array(result))

    if len(temporal_samples) == 0:
        return None

    return np.array(temporal_samples)
