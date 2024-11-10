from dataclasses import dataclass


@dataclass
class PadimADConfig:
    model_name: str
    device: str
    batch_size: int
