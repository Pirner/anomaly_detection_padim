from PIL import Image
from torchvision import transforms as T


class DataTransform:
    @staticmethod
    def get_train_transform(im_size=256, crop_size=224):
        """
        get training transformation for images
        :param im_size: symmetric size to resize the image to
        :param crop_size:  center crop operation size
        :return:
        """
        transform_x = T.Compose([
            T.Resize(im_size, Image.LANCZOS),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform_x
