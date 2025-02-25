import json


class ResourcesUtils:
    @staticmethod
    def read_introduction_text(resources_filepath: str):
        """
        reads the introduction text from the resources file
        :param resources_filepath:
        :return: 
        """
        with open(resources_filepath, 'r') as f:
            d = json.load(f)
            return d['introduction_text']

    @staticmethod
    def read_step_training_text(resources_filepath: str):
        """
        reads the introduction text from the resources file
        :param resources_filepath:
        :return:
        """
        with open(resources_filepath, 'r') as f:
            d = json.load(f)
            return d['training_step_text']

    @staticmethod
    def read_step_calibration_text(resources_filepath: str):
        """
        reads the introduction text from the resources file
        :param resources_filepath:
        :return:
        """
        with open(resources_filepath, 'r') as f:
            d = json.load(f)
            return d['calibration_step_text']

    @staticmethod
    def read_step_detecting_text(resources_filepath: str):
        """
        reads the introduction text from the resources file
        :param resources_filepath:
        :return:
        """
        with open(resources_filepath, 'r') as f:
            d = json.load(f)
            return d['detection_step_text']

