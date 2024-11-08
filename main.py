from PadimAD import PadimAnomalyDetector
from config.DTO import PadimADConfig


def main():
    config = PadimADConfig(
        model_name='wide_resnet50_2',
        device='cuda',
    )
    ad_detector = PadimAnomalyDetector(config=config)


if __name__ == '__main__':
    main()
