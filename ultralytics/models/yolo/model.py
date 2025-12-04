from ultralytics.nn.modules.cnn_vit_backbone import CNN_ViT_Backbone
from ultralytics.nn.modules.conv import Conv  # 假设Conv模块已实现
from ultralytics.nn.modules.detect import Detect  # 假设Detect模块已实现
import torch.nn as nn

class YOLOv11Model(nn.Module):
    def __init__(self, nc=80, backbone_args=None, head_args=None):
        super().__init__()
        # 构建backbone
        if backbone_args is None:
            backbone_args = [224, 16, 3, 384, 12, 6]
        self.backbone = CNN_ViT_Backbone(*backbone_args)
        # 构建head
        self.head = nn.Sequential(
            Conv(384, 256, 3, 1),  # 假设ViT输出通道为384
            Detect(256, nc)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x