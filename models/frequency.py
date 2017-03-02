from .abstract_model import AbstractModel
from tools import create_address_books


class FrequencyModel(AbstractModel):
    def __init__(self, nb_recipients_to_predict=10):
        AbstractModel.__init__(self, nb_recipients_to_predict)
        self.address_books = None

    def fit(self, training, training_info):
        self.address_books = create_address_books(training, training_info)

    def predict(self, test, test_info):

        predictions_per_sender = {}
        for _, row in test.iterrows():
            name_ids = row.tolist()
            sender = name_ids[0]
            # get IDs of the emails for which recipient prediction is needed
            ids_predict = name_ids[1].split(' ')
            ids_predict = [int(my_id) for my_id in ids_predict]
            freq_preds = []
            # select k most frequent recipients for the user
            k_most = [elt[0] for elt in self.address_books[sender][:self.nb_recipients_to_predict]]
            for id_predict in ids_predict:
                # for the frequency baseline, the predictions are always the same
                freq_preds.append(k_most)
            predictions_per_sender[sender] = [ids_predict, freq_preds]

        return predictions_per_sender
