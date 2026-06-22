import itertools
from typing import Literal

import torch
import torch.nn as nn

import torch.nn.functional as F

from common.sr_backbone_utils import ResidualBlocksWithInputConv


class MDSelfS2Model(nn.Module):
    def __init__(self,
                 input_bands: Literal['10m', '20m', '10m_20m'] = '10m_20m',
                 output_bands: Literal['10m', '20m', '10m_20m'] = '10m_20m',
                 multi_pred=None, padding_mode: str = "zeros"):

        super().__init__()

        # Networks parameters
        encoder_10m_blocks = 10
        encoder_20m_blocks = 10
        encoder_blocks = 60
        mid_channels = 80

        n_channels = {
            '10m': 4,
            '20m': 6,
            '10m_20m': 10
        }

        self.input_bands = input_bands
        self.output_bands = output_bands

        assert output_bands in input_bands

        # encoder module
        self.encoder_10m = ResidualBlocksWithInputConv(n_channels['10m'], mid_channels, encoder_10m_blocks, padding_mode)
        self.encoder_20m = ResidualBlocksWithInputConv(n_channels['20m'], mid_channels, encoder_20m_blocks, padding_mode)

        fusion_encoder_channels = 2 * mid_channels if input_bands == '10m_20m' else mid_channels
        self.encoder_band_fusion = ResidualBlocksWithInputConv(fusion_encoder_channels, mid_channels, encoder_blocks, padding_mode)

        self.temporal_fusion = TemporalFusion(multi_pred=multi_pred, input_channels=mid_channels,
                                              mid_channels=mid_channels, fs_blocks=8)

        self.final_conv_block_10m = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, 4, 3, 1, 1, bias=True)
        )
        self.final_conv_block_20m = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, 6, 3, 1, 1, bias=True)
        )

    def decoder(self, x, ref_image_10m, ref_image_20m):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        output = []
        if '10m' in self.output_bands:
            x10m = self.final_conv_block_10m(x)
            x10m = x10m + ref_image_10m
            output.append(x10m)
        if '20m' in self.output_bands:
            x20m = self.final_conv_block_20m(x)
            x20m = x20m + ref_image_20m
            output.append(x20m)

        return torch.cat(output, dim=1)


    def forward(self, *args, grid_temp_align=None):

        ref_image_10m, ref_image_20m = None, None
        if self.input_bands == '10m':
            input_10m = args[0]
            batch_size, num_images, _, height, width = input_10m.size()
            encoded_10m = self.encoder_10m(input_10m.view(-1, 4, height, width))
            final_encoded = self.encoder_band_fusion(encoded_10m)
            ref_image_10m = torch.nn.functional.interpolate(input_10m[:, 0, ...], scale_factor=2, mode="bilinear",
                                                            align_corners=False)
        elif self.input_bands == '20m':
            input_20m = args[0]
            batch_size, num_images, _, height_20m, width_20m = input_20m.size()
            height, width = height_20m * 2, width_20m * 2
            encoded_20m = self.encoder_20m(input_20m.view(-1, 6, height_20m, width_20m))
            encoded_20m = torch.nn.functional.interpolate(encoded_20m, scale_factor=2, mode="nearest")
            final_encoded = self.encoder_band_fusion(encoded_20m)
            ref_image_20m = torch.nn.functional.interpolate(input_20m[:, 0, ...], scale_factor=4, mode="bilinear",
                                                            align_corners=False)
        else:
            input_10m, input_20m = args
            batch_size, num_images, _, height_20m, width_20m = input_20m.size()
            _, _, _, height, width = input_10m.size()
            encoded_10m = self.encoder_10m(input_10m.view(-1, 4, height, width))
            encoded_20m = self.encoder_20m(input_20m.view(-1, 6, height_20m, width_20m))
            encoded_20m = torch.nn.functional.interpolate(encoded_20m, scale_factor=2, mode="nearest")
            encoder_fusion_input = torch.cat([encoded_10m, encoded_20m], dim=1)
            final_encoded = self.encoder_band_fusion(encoder_fusion_input)

            ref_image_10m = torch.nn.functional.interpolate(input_10m[:, 0, ...], scale_factor=2, mode="bilinear",
                                                            align_corners=False)
            ref_image_20m = torch.nn.functional.interpolate(input_20m[:, 0, ...], scale_factor=4, mode="bilinear",
                                                            align_corners=False)

        if grid_temp_align is not None:
            grid_temp_align = grid_temp_align.view(-1, height, width, 2)
            final_encoded = F.grid_sample(final_encoded, grid_temp_align, mode='bilinear', padding_mode='reflection',
                                          align_corners=False)
        final_encoded = final_encoded.view(batch_size, num_images, -1, height, width)

        fused = self.temporal_fusion(final_encoded)

        if isinstance(fused, tuple) or isinstance(fused, list):
            out = []
            for i in range(len(fused)):
                out.append(self.decoder(fused[i], ref_image_10m, ref_image_20m))
            return out
        else:
            out = self.decoder(fused,  ref_image_10m, ref_image_20m)
            return out


class TemporalFusion(nn.Module):
    def __init__(self, input_channels=64, mid_channels=64, pre_fusion_blocks=10, weight_pred_blocks=4, fm_blocks=3,
                 fs_blocks=3, multi_pred=True):
        super(TemporalFusion, self).__init__()

        self.pre_fusion = ResidualBlocksWithInputConv(2*input_channels, mid_channels, pre_fusion_blocks)
        self.weight_pred = ResidualBlocksWithInputConv((2*input_channels) + mid_channels, mid_channels, weight_pred_blocks)

        self.final_fusion_multiple = ResidualBlocksWithInputConv(input_channels + mid_channels, mid_channels, fm_blocks)
        self.final_fusion_single = ResidualBlocksWithInputConv(mid_channels, mid_channels, fs_blocks)
        self.multi_pred = multi_pred

    def forward(self, x):

        num_images = x.size(1)

        if num_images == 1:
            fused = self.final_fusion_single(x[:, 0, ...])
            return fused
        else:
            ref_feat = x[:, :1, :, :, :]
            oth_feat = x[:, 1:, :, :, :]

            ref_feat = ref_feat.repeat(1, oth_feat.shape[1], 1, 1, 1)

            # Concatenate the reference and other features
            features = torch.cat((ref_feat, oth_feat), dim=2)
            # Reshape the features for the encoder
            batch_size, num_bursts, num_features, height, width = features.size()
            all_feat = features.view(-1, num_features, height, width)
            # Pass the features through the fusion convolution
            all_feat = self.pre_fusion(all_feat)
            all_feat = all_feat.view(batch_size, num_bursts, -1, height, width)

            # Predict weights
            diff_feat = oth_feat - ref_feat
            features_w = torch.cat((ref_feat, diff_feat, all_feat), dim=2)
            batch_size, num_bursts, num_features, height, width = features_w.size()
            features_w = features_w.view(-1, num_features, height, width)
            weights = self.weight_pred(features_w)
            weights = weights.view(batch_size, -1, *weights.shape[-3:])

            weights = torch.sigmoid(weights)
            weights_norm = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
            fused = (all_feat * weights_norm).sum(dim=1)

            fused = torch.cat([fused, x[:, 0, ...]], dim=1)

            fused = self.final_fusion_multiple(fused)

            if self.multi_pred:
                fused_list = [fused]
                for i in self.multi_pred:
                    if i == 0:
                        single_pred = self.final_fusion_single(x[:, 0, ...])
                        fused_list.append(single_pred)
                    else:
                        weights_norm = weights[:, :i] / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
                        fused = (all_feat[:, :i] * weights_norm).sum(dim=1)
                        fused = torch.cat([fused, x[:, 0, ...]], dim=1)
                        fused = self.final_fusion_multiple(fused)
                        fused_list.append(fused)

                return fused_list

            return fused


if __name__ == '__main__':
    input_tensor_10m = torch.randn(4, 6, 4, 50, 50)
    input_tensor_20m = torch.randn(4, 6, 6, 25, 25)

    input_bands = ['10m', '20m', '10m_20m']
    out_bands = ['10m', '20m', '10m_20m']

    model = MDSelfS2Model(input_bands='10m_20m', output_bands='10m_20m', multi_pred=[2, 1, 0])
    output_tensor = model(input_tensor_10m, input_tensor_20m)
    for v in output_tensor:
        print(v.shape)
    for ib, ob in itertools.product(input_bands, out_bands):
        print(f"{ib} -> {ob} test")
        try:
            model = MDSelfS2Model(input_bands=ib, output_bands=ob)
            if ib == '10m':
                output_tensor = model(input_tensor_10m)
            elif ib == '20m':
                output_tensor = model(input_tensor_20m)
            else:
                output_tensor = model(input_tensor_10m, input_tensor_20m)
            print("Output tensor shape:", output_tensor.shape)
        except:
            print("Failed")

