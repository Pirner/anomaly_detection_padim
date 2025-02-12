import os
import glob

from PIL import Image
from torch.utils.data import Dataset


class PadimDataset(Dataset):
    """
    Class to manage data loading for the Padim Anomaly Detection algorithm.
    """
    def __init__(self, data_path: str, transform, file_extensions='png'):
        """
        default constructor for the PadimDataset class
        :param data_path: path to the data
        :param transform: transformations to run on the data
        :param file_extensions: file extension to be used
        """
        self.data_path = data_path
        self.file_extension = file_extensions
        self.data = glob.glob(os.path.join(data_path, '**/*.{}'.format(self.file_extension)), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        im_path = self.data[item]
        x = Image.open(im_path).convert('RGB')
        x = self.transform(x)
        return x
