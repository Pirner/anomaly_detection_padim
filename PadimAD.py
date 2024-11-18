import random
import os

import cv2
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from ai.DTO import FeatureExtraction
from ai.feature_extraction import FeatureExtractor
from config.DTO import PadimADConfig
from data.transform import DataTransform


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
        # TODO pickle is broken here!!
        with open('train_class.pkl', 'rb') as f:
            self.train_outputs = pickle.load(f)
        # return

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

    def detect_anomalies_on_dataset(self, dataset, path: str):
        """
        detect anomalies on an entire dataset
        :param dataset: test-dataset to detect anomalies on.
        :return:
        """
        test_dataloader = DataLoader(dataset, batch_size=self.config.batch_size, pin_memory=True)

        test_images = []
        test_fes = []
        for x in tqdm(test_dataloader, '| feature extraction | test |'):
            test_images.extend(x.cpu().detach().numpy())
            with torch.no_grad():
                fe = self.feat_extractor.extract_features(in_sample=x)
                fe.detach_cpu()
                fe.move_to_device('cpu')
                test_fes.append(fe)

        fe_summary = FeatureExtraction(
            layer_0=torch.cat([x.layer_0.clone() for x in test_fes], dim=0),
            layer_1=torch.cat([x.layer_1.clone() for x in test_fes], dim=0),
            layer_2=torch.cat([x.layer_2.clone() for x in test_fes], dim=0),
        )
        fe_summary.embed_vectors()
        # randomly select d dimension
        embedding_vectors = torch.index_select(fe_summary.embedded_vectors, 1, self.feat_extractor.idx)

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in tqdm(range(H * W), total=H * W):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(
            dist_list.unsqueeze(1),
            size=x.size(2),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        score_diff = scores.max() - scores.min()
        threshold = int(0.356 * 255)
        for i in range(len(test_images)):
            origin_im = DataTransform.denormalize_image(test_images[i])
            cv2.imwrite(os.path.join(path, '{:02d}_origin_im.png'.format(i)), origin_im)
            test_score_map = (scores[i] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(path, '{:02d}_scores.png'.format(i)), test_score_map)
            (T, thresh) = cv2.threshold(test_score_map, threshold, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(path, '{:02d}_thresh.png'.format(i)), thresh)

