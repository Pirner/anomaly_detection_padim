import random

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai.DTO import FeatureExtraction
from ai.feature_extraction import FeatureExtractor
from config.DTO import PadimADConfig


class PadimAnomalyDetector:

    train_outputs = None

    """
    central class for managing to create, load, run and use anomaly detection based on PaDim.
    """
    def __init__(self, config: PadimADConfig):
        """
        default constructor with config for a Padim Anomaly Detector
        :param config:
        """
        self.config = config
        self.feat_extractor = FeatureExtractor(
            model_name=config.model_name,
            device=config.device,
        )
        # TODO debug statement to check results for repeatability
        random.seed(1024)
        torch.manual_seed(1024)
        if config.device == 'cuda':
            torch.cuda.manual_seed_all(1024)

    def train_anomaly_detection(self, dataset):
        """
        train an anomaly detection on a given dataset, good data without anomalies
        :param dataset: holds the data on which the AD is being tuned for.
        :return:
        """
        train_dataloader = DataLoader(dataset, batch_size=self.config.batch_size, pin_memory=True)
        print('[INFO] extracting features from training dataset: {}'.format(len(dataset)))
        feature_extractions = []
        for x in tqdm(train_dataloader, total=len(train_dataloader)):
            fe = self.feat_extractor.extract_features(in_sample=x)
            fe.detach_cpu()
            fe.move_to_device('cpu')
            feature_extractions.append(fe)
        fe_summary = FeatureExtraction(
            layer_0=torch.cat([x.layer_0.clone() for x in feature_extractions], dim=0),
            layer_1=torch.cat([x.layer_1.clone() for x in feature_extractions], dim=0),
            layer_2=torch.cat([x.layer_2.clone() for x in feature_extractions], dim=0),
        )
        fe_summary.embed_vectors()

        # randomly select d dimension
        embedding_vectors = torch.index_select(fe_summary.embedded_vectors, 1, self.feat_extractor.idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        print('[INFO] creating covariance matrices for each pixel')
        for i in tqdm(range(H * W), total=H * W):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        self.train_outputs = [mean, cov]
        with open('train_class.pkl', 'wb') as f:
            pickle.dump(self.train_outputs, f)
        print('[INFO] finished adjusting padim anomaly detector.')
