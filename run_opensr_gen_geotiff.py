
###
## Gen geotiff opensr_pred predictions, adapted from https://huggingface.co/isp-uv-es/superIX


from pathlib import Path

from data.opensr_multidate import load_opensr_multidate
from common.load import LoadModels
import functools
import rasterio as rio
import opensr_test
import matplotlib.pyplot as plt
from typing import Callable, Union
import torch
from skimage.measure import block_reduce
import numpy as np

from local import EnvironmentSettings


def create_geotiff(
        model: Callable,
        fn: Callable,
        datasets: Union[str, list],
        output_path: str,
        force: bool = False,
        **kwargs
) -> None:
    """Create all the GeoTIFFs for a specific dataset snippet
    Args:
        model (Callable): The model to use to run the fn function.
        fn (Callable): A function that return a dictionary with the following keys:
            - "lr": Low resolution image
            - "sr": Super resolution image
            - "hr": High resolution image
        datasets (list): A list of dataset snippets to use to run the fn function.
        output_path (str): The output path to save the GeoTIFFs.
        force (bool, optional): If True, the dataset is redownloaded. Defaults
            to False.
    """

    if datasets == "all":
        datasets = opensr_test.datasets

    for snippet in datasets:
        create_geotiff_batch(
            model=model,
            fn=fn,
            snippet=snippet,
            output_path=output_path,
            force=force,
            **kwargs
        )

    return None


def create_geotiff_batch(
        model: Callable,
        fn: Callable,
        snippet: str,
        output_path: str,
        force: bool = False,
        **kwargs
) -> Path:
    """Create all the GeoTIFFs for a specific dataset snippet
    Args:
        model (Callable): The model to use to run the fn function.
        fn (Callable): A function that return a dictionary with the following keys:
            - "lr": Low resolution image
            - "sr": Super resolution image
            - "hr": High resolution image
        snippet (str): The dataset snippet to use to run the fn function.
        output_path (str): The output path to save the GeoTIFFs.
        force (bool, optional): If True, the dataset is redownloaded. Defaults
            to False.
    Returns:
        Path: The output path where the GeoTIFFs are saved.
    """

    # Create folders to save results
    output_path = EnvironmentSettings.OPENSR_PRED /output_path / "results" / "SR"
    output_path.mkdir(parents=True, exist_ok=True)

    output_path_dataset_geotiff = output_path / snippet / "geotiff"
    output_path_dataset_geotiff.mkdir(parents=True, exist_ok=True)

    output_path_dataset_npy = output_path / snippet / "npy"
    output_path_dataset_npy.mkdir(parents=True, exist_ok=True)

    output_path_dataset_png = output_path / snippet / "png"
    output_path_dataset_png.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    dataset = opensr_test.load(snippet, force=force, model_dir=EnvironmentSettings.OPENSR_SDDATA.__str__())
    lr_dataset, l1c_dataset, hr_dataset, metadata = dataset["L2A"],  dataset["L1C"], dataset["HRharm"], dataset[
        "metadata"]
    for index in range(len(lr_dataset)):
        print(f"Processing {index}/{len(lr_dataset)}")

        # Run the model
        results = fn(
            model=model,
            lr=lr_dataset[index],
            lr_l1c=l1c_dataset[index],
            hr=hr_dataset[index],
            dname=snippet,
            roi=metadata["roi"][index],
            **kwargs
        )

        # Get the image name
        image_name = metadata.iloc[index]["hr_file"]

        # Get the CRS and transform
        crs = metadata.iloc[index]["crs"]
        transform_str = metadata.iloc[index]["affine"]
        transform_list = [float(x) for x in transform_str.split(",")]
        transform_rio = rio.transform.from_origin(
            transform_list[2],
            transform_list[5],
            transform_list[0],
            transform_list[4] * -1
        )

        # Create rio dict
        meta_img = {
            "driver": "GTiff",
            "count": 3,
            "dtype": "uint16",
            "height": results["hr"].shape[1],
            "width": results["hr"].shape[2],
            "crs": crs,
            "transform": transform_rio,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True
        }

        if "all" in results:
            np.save(output_path_dataset_npy / (image_name + ".npy"), results["all"].transpose(1, 2, 0))

        # Save the GeoTIFF
        with rio.open(output_path_dataset_geotiff / (image_name + ".tif"), "w", **meta_img) as dst:
            dst.write(np.clip(results["sr"], 0, None))

        # Save the PNG
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow((results["lr"].transpose(1, 2, 0) / 3000).clip(0, 1))
        ax[0].set_title("LR")
        ax[0].axis("off")
        ax[1].imshow((results["sr"].transpose(1, 2, 0) / 3000).clip(0, 1))
        ax[1].set_title("SR")
        ax[1].axis("off")
        ax[2].imshow((results["hr"].transpose(1, 2, 0) / 3000).clip(0, 1))
        ax[2].set_title("HR")
        # remove whitespace around the image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        plt.savefig(output_path_dataset_png / (image_name + ".png"))
        plt.close()
        plt.clf()

    return output_path_dataset_geotiff

# Predict using only 10m bands
def _pred_multi(model, lr, lr_l1c, hr, dname=None, roi=None, frames=9):

    auxiliary_data = load_opensr_multidate(EnvironmentSettings.OPENSR_MDDATA, dname, roi, 'l2a')
    lr_input = np.expand_dims(lr, 0)

    lr_input = lr_input[:, [3, 2, 1, 7]]

    if auxiliary_data is not None:
        lr_input = np.concatenate([lr_input, auxiliary_data[:, [3, 2, 1, 7]]], axis=0)

    lr_input = lr_input[:frames]

    lr_input = torch.from_numpy(lr_input).to(EnvironmentSettings.DEVICE).float()
    lr_input = lr_input / 256.0

    with torch.autocast(device_type=EnvironmentSettings.DEVICE, dtype=torch.float16):
        sr = model(lr_input.unsqueeze(0), None)
    sr = sr * 256.0
    sr = sr[0].cpu().detach().numpy()

    return {
        'lr': lr[[3, 2, 1],],
        'sr': sr[:3],
        'all': sr,
        'hr': hr[:3]
    }


# Predict with all the bands and auxiliary data, including 20m bands and auxiliary data, and with the fix of input size for some datasets
def _pred_multi_all(model, lr, lr_l1c, hr, dname=None, roi=None, frames=9):

    lr_input = lr[[3, 2, 1, 7, 4, 5, 6, 8, 10, 11]]

    intput_size_fix = lr_input.shape[1] == 121

    input_10m = lr_input[:4]
    input_20m = lr_input[4:]
    if intput_size_fix:
        input_10m = input_10m[:, :120, :120]
        input_20m = input_20m[:, :120, :120]

    input_20m = block_reduce(input_20m, (1, 2, 2), np.median)

    input_10m = np.expand_dims(input_10m, 0)
    input_20m = np.expand_dims(input_20m, 0)

    auxiliary_data_10m = load_opensr_multidate(EnvironmentSettings.OPENSR_MDDATA, dname, roi, 'l2a', selected_bands=[3, 2, 1, 7])
    auxiliary_data_20m = load_opensr_multidate(EnvironmentSettings.OPENSR_MDDATA, dname, roi, 'l2a', selected_bands=[4, 5, 6, 8, 10, 11])

    if auxiliary_data_10m is not None and auxiliary_data_20m is not None:
        if intput_size_fix:
            auxiliary_data_10m = auxiliary_data_10m[:, :, :120, :120]

        input_10m = np.concatenate([input_10m, auxiliary_data_10m], axis=0)
        input_20m = np.concatenate([input_20m, auxiliary_data_20m], axis=0)

    input_10m = input_10m[:frames]
    input_20m = input_20m[:frames]

    input_10m = torch.from_numpy(input_10m).to(EnvironmentSettings.DEVICE).float()
    input_10m = input_10m / 256.0

    input_20m = torch.from_numpy(input_20m).to(EnvironmentSettings.DEVICE).float()
    input_20m = input_20m / 256.0

    with torch.autocast(device_type=EnvironmentSettings.DEVICE, dtype=torch.float16):
        sr = model(input_10m.unsqueeze(0), input_20m.unsqueeze(0))


    sr = sr * 256.0
    sr = sr[0].cpu().detach().numpy()

    if intput_size_fix:
        sr = np.pad(sr, ((0, 0), (0, 2), (0, 2)), mode='constant', constant_values=0)

    return {
        'lr': lr[[3, 2, 1],],
        'sr': sr[:3],
        'all': sr,
        'hr': hr[:3]
    }


def main():
    model = LoadModels.sdselfsen2()
    model = model.to(EnvironmentSettings.DEVICE)
    create_geotiff(model, functools.partial(_pred_multi_all, frames=1), "all", output_path="singleall", force=False)

    model = LoadModels.sdselfsen2_olny10m()
    model = model.to(EnvironmentSettings.DEVICE)
    create_geotiff(model, functools.partial(_pred_multi, frames=1), "all", output_path="single10m", force=False)

    model = LoadModels.mdselfsen2()
    model = model.to(EnvironmentSettings.DEVICE)
    create_geotiff(model, functools.partial(_pred_multi_all, frames=4), "all", output_path=f"multiall", force=False)

    model = LoadModels.mdselfsen2_only10m()
    model = model.to(EnvironmentSettings.DEVICE)
    create_geotiff(model, functools.partial(_pred_multi, frames=4), "all", output_path=f"multi10m", force=False)



if __name__ == '__main__':
    main()
