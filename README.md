## Introduction

This is an unofficial implementation of the paper: https://arxiv.org/abs/2011.08785
The aim of this project is to provide a good entry point of retraining the algorithm on a different
dataset with different needs.

## Get Started

install all the requirements
````commandline
pip install -r requirements.txt
````

For using the padim anomaly detector you should separate your dataset
into 3 chunks:
- Training
- Calibration
- Test

While training and calibration contain no anomalies, so are only "good" data
the testing dataset should contain some anomalies to check whether the algorithm
is properly working.

The training dataset is used to create the features that are used to setup the system. The
calibration dataset is for setting a threshold to compare against real anomalies.

First we need to instantiate an anomaly detector with:
```python
import numpy as np
from PIL import Image

from anomaly_detection_padim.PadimAD import PadimAnomalyDetector
from anomaly_detection_padim.config.DTO import PadimADConfig
from anomaly_detection_padim.data.transform import DataTransform
from anomaly_detection_padim.data.dataset import PadimDataset



path_train_data = r'<insert_train_path>'
path_calibration_data = r'<insert_calibration_path>'
anomaly_im_path = r'<path_to_anomalous_image>'

config = PadimADConfig(
        model_name='wide_resnet50_2',
        device='cuda',
        batch_size=8
    )
padim_ad = PadimAnomalyDetector(config=config)
train_dataset = PadimDataset(data_path=path_train_data, transform=DataTransform.get_train_transform())
padim_ad.train_anomaly_detection(dataset=train_dataset)
calibration_dataset = PadimDataset(data_path=path_calibration_data, transform=DataTransform.get_test_transform())

src_im = Image.open(anomaly_im_path).convert('RGB')
anom_score = padim_ad.detect_anomaly(
    im=src_im, 
    transform=DataTransform.get_test_transform(),
    normalize=True,
)
anomaly_im = Image.fromarray(anom_score.astype(np.uint8))
```

## How to use the GUI
this repository provides

## Roadmap

- [ ] Add Variable Image Sizes to the Padim Anomaly Detector
- [ ] Forward the variable image size to the GUI
- [ ] Add status message for training and calibration part
- [ ] rework home frame
- [ ] create api to serve the anomaly detection results
- [ ] add storing anomaly detectors (serializing to disk)