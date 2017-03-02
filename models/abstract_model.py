from abc import ABCMeta, abstractmethod


class AbstractModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, nb_recipients_to_predict=10):
        self.nb_recipients_to_predict = nb_recipients_to_predict

    @abstractmethod
    def fit(self, training, training_info):
        pass

    @abstractmethod
    def predict(self, test, test_info):
        pass
