from .abstract_model import AbstractModel


class HybridModel(AbstractModel):
    """ Class defining hybrid models.

    An hybrid model is a model which use two different models to make its prediction.
    If the length of a mail is below a threshold (length_threshold), then the first model of base_estimators is used.
    If the length is above this threshold then the second model is used
    """

    def __init__(self, base_estimators, length_threshold=50, nb_recipients_to_predict=10, verbose=True):
        """  Constructor of HybridModel

        :param base_estimators: list of two estimators
        :param length_threshold: threshold of length
        """
        AbstractModel.__init__(self, nb_recipients_to_predict, verbose)
        self.length_threshold = length_threshold
        self.base_estimators = base_estimators

        self.__check_base_estimators()

    def __check_base_estimators(self):
        """ Function to check self.base_estimators properties """
        if not isinstance(self.base_estimators, list):
            raise TypeError("In HybridModel, base_estimators should be a list of estimators")

        if len(self.base_estimators) != 2:
            raise ValueError("In HybridModel, base_estimators should contain two estimators")

        for estimator in self.base_estimators:
            if not isinstance(estimator, AbstractModel):
                raise ValueError("In HybridModel, each estimator should be an instance of a subclass os AbstractModel")

    def fit(self, training, training_info):
        # every estimators is trained on the whole dataset
        for estimator in self.base_estimators:
            estimator.fit(training, training_info)

    def predict(self, test, test_info):
        mails_length = test_info["tokens"].apply(lambda tokens: len(tokens.split(" ")))
        mails_length.index = test_info["mid"]

        # for each model, gather the id of messages depending on their length
        first_model_mids = mails_length[mails_length < self.length_threshold].index
        second_model_mids = mails_length[mails_length >= self.length_threshold].index

        # create a copy of test for each model
        first_model_test = test.copy()
        second_model_test = test.copy()

        # for each model, keep only the id of messages we selected
        first_model_test["mids"] = test["mids"].apply(
            lambda mids: " ".join([mid for mid in mids.split(" ") if int(mid) in first_model_mids]))
        second_model_test["mids"] = test["mids"].apply(
            lambda mids: " ".join([mid for mid in mids.split(" ") if int(mid) in second_model_mids]))

        # for each test dataset, delete sender with no messages
        first_model_test = first_model_test[first_model_test["mids"] != '']
        second_model_test = second_model_test[second_model_test["mids"] != '']

        # make the predictions
        first_predictions_per_sender = self.base_estimators[0].predict(first_model_test, test_info)
        second_predictions_per_sender = self.base_estimators[1].predict(second_model_test, test_info)

        # gather the two dictionaries of prediction together
        predictions_per_sender = first_predictions_per_sender.copy()
        for sender, value in second_predictions_per_sender.items():
            # mids and predictions of second model
            second_mids = value[0]
            second_preds = value[1]

            # if sender exists in first_predictions_per_sender then gather the mids and the predictions
            # otherwise add the message ids and the predictions to the final dictionary
            if sender in first_predictions_per_sender.keys():
                # mids and predictions of first model
                mids = first_predictions_per_sender[sender][0]
                preds = first_predictions_per_sender[sender][1]

                # gather them
                mids.extend(second_mids)
                preds.extend(second_preds)

            else:
                mids = second_mids
                preds = second_preds

            # store in predictions_per_sender dict
            predictions_per_sender[sender] = [mids, preds]

        return predictions_per_sender
