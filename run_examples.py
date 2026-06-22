from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from common.load import LoadModels, load_roi_temporal_bands


def main():

    INPUT_FOLDER = Path('_examples')
    model = LoadModels.mdselfsen2()
    model.cuda()

    modelsd = LoadModels.sdselfsen2()
    modelsd.cuda()

    OUTPUT_FOLDER = Path('_workspace/run_examples')

    for folder in INPUT_FOLDER.iterdir():
        print("Processing folder: ", folder)

        out_folder = OUTPUT_FOLDER / folder.stem
        out_folder.mkdir(exist_ok=True, parents=True)

        input10m, input20m = load_roi_temporal_bands(folder, bands_10m=True, bands_20m=True, reverse_order=True)

        input10m = input10m / 255.
        input20m = input20m / 255.

        input10m = torch.from_numpy(input10m).cuda().float()
        input20m = torch.from_numpy(input20m).cuda().float()

        sr_out = model(input10m[:6].unsqueeze(0), input20m[:6].unsqueeze(0))
        sr_out_single = modelsd(input10m[:1].unsqueeze(0), input20m[:1].unsqueeze(0))

        np.save(out_folder / f'in_10m.npy', input10m.permute(0, 2, 3, 1).cpu().numpy())
        np.save(out_folder / f'in_20m.npy', input20m.permute(0, 2, 3, 1).cpu().numpy())
        np.save(out_folder / f'out.npy', sr_out.permute(0, 2, 3, 1).detach().cpu().numpy())
        np.save(out_folder / f'out_sd.npy', sr_out_single.permute(0, 2, 3, 1).detach().cpu().numpy())

        in_rgb = input10m[0, :3].permute(1, 2, 0).detach().cpu().numpy()
        out_rgb = sr_out[0, :3].permute(1, 2, 0).detach().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.clip((in_rgb-np.min(in_rgb))/np.max(in_rgb), 0, 1))
        ax[0].set_title('IN')
        ax[0].axis('off')
        ax[1].imshow(np.clip((out_rgb-np.min(out_rgb))/np.max(out_rgb), 0, 1))
        ax[1].set_title('OUT')
        ax[1].axis('off')
        fig.tight_layout(pad=0.1)
        fig.savefig(out_folder / 'compare_in_out.png', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

if __name__ == "__main__":
    main()