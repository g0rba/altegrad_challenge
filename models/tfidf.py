from .abstract_model import AbstractModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import create_address_books, create_dictionary_mids, cosine_similarity
import numpy as np
import pandas as pd


class TfIdfModel(AbstractModel):
    def __init__(self, nb_recipients_to_predict=10):
        self.nb_recipients_to_predict = nb_recipients_to_predict
        self.tfidf_vectorizer = TfidfVectorizer()

    def fit(self, training, training_info):
        # store training sets
        self.training = training
        self.training_info = training_info

        # compute tdidf for training set
        self.training_matrix = self.tfidf_vectorizer.fit_transform(training_info["tokens"])
        self.address_books = create_address_books(training, training_info)
        self.mids_sender_recipient = create_dictionary_mids(training, training_info)

    def predict(self, test, test_info):
        test_matrix = self.tfidf_vectorizer.transform(test_info["tokens"])
        predictions_per_sender = {}
        for _, row in test.iterrows():
            # retrieve sender attributes
            sender = row[0]
            mids_sender = self.training[self.training["sender"] == sender]["mids"].values[0]
            mids_sender = np.array(mids_sender.split(" "), dtype=int)
            mids_sender_series = pd.Series(mids_sender)
            position_mails_training = self.training_info[self.training_info["mid"].isin(mids_sender)].index.values
            tfidf_mails_sender = self.training_matrix[position_mails_training]

            # get IDs of the emails for which recipient prediction is needed
            mids_predict = np.array(row[1].split(" "), dtype=int)

            # initialize list to store predictions
            tfidf_preds = []

            for mid_predict in mids_predict:
                # get the position of current mail in test_info dataset
                position_mail_predict = test_info[test_info["mid"] == mid_predict].index[0]
                tfidf_mail_predict = test_matrix[position_mail_predict]

                # compute similarities
                similarities = pd.Series(tfidf_mails_sender.dot(tfidf_mail_predict.T).toarray().flatten())

                # loop over sender's address book and for each recipient, find the most similar mail it received
                scores = []
                for recipient, nb_occurrences in self.address_books[sender]:
                    mids_recipient = self.mids_sender_recipient[(sender, recipient)]
                    similarities_recipient = similarities[np.nonzero(mids_sender_series.isin(mids_recipient))[0]]
                    scores.append((recipient, similarities_recipient.max()))

                # sort the scores and get the 10 recipients with higher scores
                prediction = [recipient for recipient, score in
                              sorted(scores, key=lambda elt: elt[1], reverse=True)[:self.nb_recipients_to_predict]]
                tfidf_preds.append(prediction)

            predictions_per_sender[sender] = [mids_predict, tfidf_preds]

        return predictions_per_sender
