from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FeatureExtraction:
    layer_0: torch.Tensor
    layer_1: torch.Tensor
    layer_2: torch.Tensor
    embedded_vectors: Optional[torch.Tensor] = None

    @staticmethod
    def embedding_concat(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def embed_vectors(self):
        """
        embed the vectors into each to form one entire array per pixel in the image
        :return:
        """
        embedding_vectors = self.layer_0
        embedding_vectors = FeatureExtraction.embedding_concat(x=embedding_vectors, y=self.layer_1)
        self.embedded_vectors = FeatureExtraction.embedding_concat(x=embedding_vectors, y=self.layer_2)

    def detach_cpu(self):
        self.layer_0 = self.layer_0.cpu().detach()
        self.layer_1 = self.layer_1.cpu().detach()
        self.layer_2 = self.layer_2.cpu().detach()

    def move_to_device(self, device: str):
        self.layer_0 = self.layer_0.to(device)
        self.layer_1 = self.layer_1.to(device)
        self.layer_2 = self.layer_2.to(device)
