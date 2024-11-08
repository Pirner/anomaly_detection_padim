from torchvision.models import wide_resnet50_2, resnet18

from constants import constants_ai


class FeatureExtractor:
    def __init__(self, model_name: str, device='cuda'):
        """
        create feature extractor with a model backbone
        :param model_name: name of the model to extract features with
        """
        self.model_name = model_name
        self.model = None
        self.t_d = 0
        self.d = 0
        self.device = device
        self.outputs = []
        self.build_feature_extraction_model(model_name=model_name)

    def build_feature_extraction_model(self, model_name: str):
        """
        build
        :param model_name:
        :return:
        """
        if model_name == 'resnet18':
            self.model = resnet18(pretrained=True, progress=True)
            self.t_d = 448
            self.d = 100
        elif model_name == 'wide_resnet50_2':
            self.model = wide_resnet50_2(pretrained=True, progress=True)
            self.t_d = 1792
            self.d = 550
        else:
            raise ValueError('no valid model name provided: {}, select from {}'.format(
                model_name,
                constants_ai.supported_feature_extraction_models
            ))
        self.model.to(self.device)
        self.model.eval()

        # set model's intermediate outputs
        def hook(module, inputs, output):
            self.outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
