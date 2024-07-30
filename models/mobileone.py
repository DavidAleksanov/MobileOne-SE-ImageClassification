from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from .mobileone_block import MobileOneBlock
from .se_block import SEBlock

# Parameters for different model configurations
PARAMS = {
    's0': {'num_blocks_per_stage': [2, 8, 10, 1], 'width_multipliers': [0.75, 1.0, 1.0, 2.0], 'use_se': False},
    's1': {'num_blocks_per_stage': [2, 8, 10, 1], 'width_multipliers': [1.0, 1.5, 1.5, 2.0], 'use_se': False},
    's2': {'num_blocks_per_stage': [2, 8, 10, 1], 'width_multipliers': [1.0, 2.0, 2.0, 2.0], 'use_se': False},
    's3': {'num_blocks_per_stage': [2, 8, 10, 1], 'width_multipliers': [1.0, 2.5, 2.5, 2.5], 'use_se': False},
    's4': {'num_blocks_per_stage': [2, 8, 10, 1], 'width_multipliers': [1.0, 3.0, 3.0, 3.0], 'use_se': True}
}

class MobileOne(nn.Module):
    def __init__(self, num_blocks_per_stage: List[int], num_classes: int = 1000, width_multipliers: List[float] = None, inference_mode: bool = False, use_se: bool = False, num_conv_branches: int = 1) -> None:
        super(MobileOne, self).__init__()
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2], num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3], num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)
        self.init_params()

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int) -> nn.Sequential:
        strides = [2] + [1] * (num_blocks - 1)
        block_list = nn.Sequential()
        for ix, stride in enumerate(strides):
            use_se = self.use_se and ix >= (num_blocks - num_se_blocks)
            block_list.add_module('block{}'.format(self.cur_layer_idx), MobileOneBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, inference_mode=self.inference_mode, use_se=use_se, num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return block_list

    def init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mobileone(num_classes: int = 1000, inference_mode: bool = False, variant: str = 's0') -> MobileOne:
    variant_params = PARAMS[variant]
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode, **variant_params)
