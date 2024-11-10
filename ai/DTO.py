from dataclasses import dataclass

import torch


@dataclass
class FeatureExtraction:
    layer_0: torch.Tensor
    layer_1: torch.Tensor
    layer_2: torch.Tensor

    def detach_cpu(self):
        self.layer_0 = self.layer_0.cpu().detach()
        self.layer_1 = self.layer_1.cpu().detach()
        self.layer_2 = self.layer_2.cpu().detach()

    def move_to_device(self, device: str):
        self.layer_0 = self.layer_0.to(device)
        self.layer_1 = self.layer_1.to(device)
        self.layer_2 = self.layer_2.to(device)
