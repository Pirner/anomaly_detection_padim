import cv2
import numpy as np


class VisionRendering:
    @staticmethod
    def visualize_binary_im_on_rgb(
            bgr_im: np.ndarray,
            bin_im: np.ndarray,
            alpha=0.5,
            gamma=0.0,
    ) -> np.ndarray:
        """
        visualize a binary image on a rgb image for anomaly segmentation.
        Make sure that height and width of the rgb image and the binary image are matching, otherwise an exception
        is thrown
        :param bgr_im: 3 channel image
        :param bin_im: one channel binary image
        :param alpha: weight sharing within the weight
        :param gamma: additional brightness
        :return:
        """
        if bgr_im.shape[:2] != bin_im.shape[:2]:
            raise Exception('shapes not matching of rgb image and binary image: ', bgr_im.shape, bin_im.shape)
        bin_image_red = np.stack([bin_im, bin_im, bin_im], axis=-1)
        bin_image_red[:, :, 1] = 0
        bin_image_red[:, :, 2] = 0
        visualized = cv2.addWeighted(bgr_im, alpha, bin_image_red, 1 - alpha, gamma)
        return visualized
