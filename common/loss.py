from __future__ import annotations

from typing import Literal, Optional, Protocol, assert_never

import torch
import torch.utils.data
from typing_extensions import override


def get_loss(
        loss: bool | float | str,
        warp_padding_mode: str,
        bands: Literal["only-10m", "only-20m"],
) -> ILoss:
    if loss == "none":
        return NoLoss()

    if isinstance(loss, str) and "+" in loss:
        losses = [get_loss(l, warp_padding_mode, bands) for l in loss.split("+")]
        return LossComposer(losses)

    if loss == "accurate":
        if bands == "only-10m":
            scale = 2
        elif bands == "only-20m":
            scale = 4
        else:
            assert_never(bands)
        return SensorIntegrationLoss(warp_padding_mode=warp_padding_mode, scale=scale)

    if isinstance(loss, str) and loss.startswith("gradl2:"):
        weight = float(loss[len("gradl2:"):])
        return GradL2Loss(weight=weight)

    assert False, f"unsupported loss expression: '{loss}'"


class ILoss(Protocol):
    def __call__(
            self, y: torch.Tensor, l1b: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        First return value is the loss (tensor of shape (1,)).
        Second return value is optionally HR warped to L1B space.
        """
        ...


class SensorIntegrationLoss(torch.nn.Module, ILoss):
    """
    First warp the SR result, staying at the same resolution but with a scene aligned to the target.
    Then apply a box-filter to simulate the sensor integration.
        note that for now fractional integration is not supported
    Finally downsample the image to simulate the sampling operator.
    The last two steps are implemented by torch.nn.functional.avg_pool2d.
    """

    def __init__(
            self,
            warp_padding_mode: str,
            scale: int,
            reduce=True,
    ):
        super(SensorIntegrationLoss, self).__init__()
        self.scale = scale
        self.warp_padding_mode = warp_padding_mode
        self.reduce = reduce

    @override
    def __call__(
            self, y: torch.Tensor, l1b: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c, sr_h, sr_w = y.size()
        b, c, h, w = l1b.size()

        assert grid.shape == (b, c, h, w, 2)

        # upsample the grid to high-res space but aligned to the target
        grid = grid.view(b * c, h, w, 2)
        grid = grid.permute((0, 3, 1, 2))  # move the '2' as channels
        grid = torch.nn.functional.interpolate(
            grid,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False,
        )
        grid = grid.permute((0, 2, 3, 1))  # move the '2' back
        grid = grid.view(b * c, h * self.scale, w * self.scale, 2)

        yy = y.reshape(b * c, 1, sr_h, sr_w)

        # warp the high-res image
        yy = torch.nn.functional.grid_sample(
            yy,
            grid,
            align_corners=True,
            mode="bicubic",
            padding_mode=self.warp_padding_mode,
        )
        yy = yy.view(b, c, h * self.scale, w * self.scale)

        # apply sensor integration and sensor sampling, back to target space
        yy = torch.nn.functional.avg_pool2d(yy, self.scale)
        assert yy.shape == (b, c, h, w)

        trainloss = (yy - l1b)[mask].abs()

        if self.reduce:
            trainloss = trainloss.mean()

        return trainloss, yy.detach()


class LossComposer(torch.nn.Module, ILoss):
    def __init__(self, losses: list[ILoss]):
        super(LossComposer, self).__init__()
        assert len(losses) > 0
        self.losses = losses

    def __call__(
            self, y: torch.Tensor, l1b: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        loss, yy = self.losses[0](y, l1b, grid, mask)
        for l in self.losses[1:]:
            loss = loss + l(y, l1b, grid, mask)[0]
        return loss, yy


class NoLoss(torch.nn.Module, ILoss):
    def __init__(self):
        super(NoLoss, self).__init__()

    def __call__(
            self, y: torch.Tensor, l1b: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        loss = torch.Tensor([0.0]).to(y.device)
        return loss, None


class GradL2Loss(torch.nn.Module, ILoss):
    def __init__(self, weight: float):
        super(GradL2Loss, self).__init__()
        self.weight = weight

    def __call__(
            self, y: torch.Tensor, l1b: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        diff1 = (y[..., 1:, :] - y[..., :-1, :]).square().mean()
        diff2 = (y[..., 1:] - y[..., :-1]).square().mean()
        loss = self.weight * (diff1 + diff2) / 2.0
        return loss, None
