###
## Compute opensr_pred metrics from geotiff predictions, adapted from https://huggingface.co/isp-uv-es/superIX
from pathlib import Path
from typing import Optional

import numpy as np
import opensr_test
import pandas as pd
import rioxarray as rxr
import torch
from matplotlib import pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from local import EnvironmentSettings


def image_resize(image: np.ndarray, size: int) -> np.ndarray:
    image = torch.from_numpy(image)
    image = torch.nn.functional.interpolate(
        image / 10000,
        size=size,
        mode="bilinear",
        antialias=True
    ) * 10000
    return image.squeeze().numpy()


def run(
        model_id: str,
        sr_in_bands: list,
        output_dir: Path,
        scale: int = 4,
        version: str = "v2",
        dataset_ids: Optional[str] = None,
        experiment: Optional[dict] = None,
        compute_plots: Optional[dict] = {
            "triplets": True,
            "summary": True,
            "tc": True,
            "histogram": True,
            "ternary": True
        },
) -> None:
    """Run an experiment with opensr_pred-test.

    Args:
        model_id (str, optional): The model to experiment with.
        dataset_ids (list, optional): The datasets to evaluate the model on.
        experiment (dict, optional): The experiment configuration.
        scale (int, optional): The scale factor to resize the SR images, if
            needed.
    """

    # Global parameters
    exp_object = opensr_test.Metrics(**experiment)
    dmetric = experiment["correctness_distance"]
    columns = ["model", "dataset", "reflectance", "spectral", "spatial", "synthesis", "ha_metric", "om_metric",
               "im_metric", "psnr", "ssim", "lpips"]
    condition = (dmetric == "clip") or (dmetric == "lpips")
    df = pd.DataFrame(columns=columns)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to("cuda")

    for dataset in dataset_ids:

        output_dir.mkdir(parents=True, exist_ok=True)

        output_dir_triplets = output_dir / "triplets"
        output_dir_triplets.mkdir(parents=True, exist_ok=True)

        output_dir_summary = output_dir / f"summary_{dmetric}"
        output_dir_summary.mkdir(parents=True, exist_ok=True)

        output_dir_tc = output_dir / f"triplets_tc_{dmetric}"
        output_dir_tc.mkdir(parents=True, exist_ok=True)

        output_dir_histogram = output_dir / f"histogram_{dmetric}"
        output_dir_histogram.mkdir(parents=True, exist_ok=True)

        output_dir_ternary = output_dir / f"ternary_{dmetric}"
        output_dir_ternary.mkdir(parents=True, exist_ok=True)

        # Load RGB LR & HR data
        data = opensr_test.load(dataset, version=version, force=False, model_dir=EnvironmentSettings.OPENSR_SDDATA.__str__())
        lr, hr = data["L2A"][:, [3, 2, 1]], data["HRharm"][:, 0:3]
        metadata = data["metadata"]

        # Load SR data
        sr_path = Path(EnvironmentSettings.OPENSR_PRED, f"{model_id}/results/SR/{dataset}/geotiff")
        sr_files = list(sr_path.glob("*"))
        sr_files.sort()
        sr = np.stack([rxr.open_rasterio(file)[sr_in_bands] for file in sr_files])
        sr_scale = sr.shape[2] // lr.shape[2]
        hr_scale = hr.shape[2] // lr.shape[2]

        # Resize SR images if needed
        if sr_scale > hr_scale:
            sr = image_resize(sr, hr.shape[3])

        # Resize HR and LR images if needed
        if scale < 4:
            hr = image_resize(hr, lr.shape[3] * scale)
            sr = image_resize(sr, lr.shape[3] * scale)

        # Compute metrics
        for i in range(len(lr)):
            print(f"Processing {dataset}: image {i + 1}/{len(lr) + 1}")
            # Load image by image
            lr_img, hr_img, sr_img = lr[i], hr[i], sr[i]
            lr_img = torch.from_numpy(lr_img) / 10000
            hr_img = torch.from_numpy(hr_img) / 10000
            sr_img = torch.from_numpy(sr_img) / 10000

            # Add metadata
            hr_name = metadata.loc[i, "hr_file"]

            if condition:
                # From reflectance to true color
                # IMPORTANT ESTIMATE PERCEPTUAL METRICS IN TRUE COLOR IMAGE
                # TO CONVERT FROM REFLECTANCE TO TRUE COLOR WE FOLLOW TCI PIPELINE
                # https://sentiwiki.copernicus.eu/web/s2-products
                lr_img = (lr_img * 3).clip(0, 1)
                hr_img = (hr_img * 3).clip(0, 1)
                sr_img = (sr_img * 3).clip(0, 1)

            # Execute the experiment
            results = exp_object.compute(lr_img.float(), sr_img.float(), hr_img.float())
            results["model"] = model_id
            results["dataset"] = dataset

            # PSNR, SSIM and LPIPS (remove borders)
            sr_b = sr_img.unsqueeze(0).float().to("cuda")[:, :, 50:-50, 50:-50]
            hr_b = hr_img.unsqueeze(0).float().to("cuda")[:, :, 50:-50, 50:-50]

            psnr_val = psnr_metric(sr_b, hr_b).item()

            ssim_val = ssim_metric(sr_b, hr_b).item()

            # LPIPS expects input in [-1, 1]
            sr_lp = sr_b * 2 - 1
            hr_lp = hr_b * 2 - 1
            lpips_val = lpips_metric(sr_lp, hr_lp).item()

            results["psnr"] = psnr_val
            results["ssim"] = ssim_val
            results["lpips"] = lpips_val

            df.loc[len(df)] = results

            # Save the figures
            if compute_plots["triplets"]:
                fig, axs = exp_object.plot_triplets()
                fig.savefig(output_dir_triplets / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["summary"]:
                fig, axs = exp_object.plot_summary()
                fig.savefig(output_dir_summary / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["tc"]:
                fig, axs = exp_object.plot_tc()
                fig.savefig(output_dir_tc / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["histogram"]:
                fig, axs = exp_object.plot_histogram()
                fig.savefig(output_dir_histogram / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["ternary"]:
                fig, axs = exp_object.plot_ternary()
                fig.savefig(output_dir_ternary / f"{dataset}__{hr_name}.png")
                plt.close(fig)

    # Save the results
    df.to_csv(output_dir / f"results.csv", index=False)

    return df


if __name__ == '__main__':


    config = {"device": "cuda", "agg_method": "patch", "patch_size": 1, "correctness_distance": "fd"}

    datasets = ["spot", "naip", "spain_crops", "spain_urban", "venus"]

    for model in Path(EnvironmentSettings.OPENSR_PRED).glob("*"):
        if model.is_dir():
            print("Model: ", model)

            run(
                model_id=model.stem,
                sr_in_bands=[0, 1, 2],
                output_dir=Path(EnvironmentSettings.OPENSR_METRICS, f"{model}/"),
                dataset_ids=datasets,
                version='v2',
                experiment=config,
                compute_plots={
                    "triplets": True,
                    "summary": True,
                    "tc": True,
                    "histogram": True,
                    "ternary": True
                }
            )

            print("--------------------------------")
