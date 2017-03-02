from abc import ABCMeta, abstractmethod


class AbstractModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, nb_recipients_to_predict=10, verbose=1):
        self.nb_recipients_to_predict = nb_recipients_to_predict
        self.verbose = verbose

    @abstractmethod
    def fit(self, training, training_info):
        pass

    @abstractmethod
    def predict(self, test, test_info):
        pass
