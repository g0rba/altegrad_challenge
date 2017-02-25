from abc import ABCMeta, abstractmethod


class AbstractModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, training, training_info):
        pass

    @abstractmethod
    def predict(self, test, test_info):
        pass
