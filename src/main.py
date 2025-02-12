from data.dataset import PadimDataset
from data.transform import DataTransform
from PadimAD import PadimAnomalyDetector
from config.DTO import PadimADConfig


def main():
    config = PadimADConfig(
        model_name='wide_resnet50_2',
        device='cuda',
        batch_size=8
    )
    path_good_data = r'C:\data\mvtec\bottle\train\good'
    path_anomalous_data = r'C:\data\mvtec\bottle\test\contamination'
    path_anomalous_data = r'C:\data\mvtec\bottle\test'
    ad_detector = PadimAnomalyDetector(config=config)
    good_dataset = PadimDataset(data_path=path_good_data, transform=DataTransform.get_train_transform())
    ad_detector.train_anomaly_detection(dataset=good_dataset)
    ad_dataset = PadimDataset(data_path=path_anomalous_data, transform=DataTransform.get_test_transform())
    ad_detector.detect_anomalies_on_dataset(dataset=ad_dataset, path=r'C:\data\padim\anom')
    ad_detector.detect_anomalies_on_dataset(dataset=good_dataset, path=r'C:\data\padim\good')


if __name__ == '__main__':
    main()
