from .abstract_model import AbstractModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import create_address_books, create_dictionary_mids, cosine_similarity
import numpy as np
import pandas as pd


class TfIdfModel(AbstractModel):
    def __init__(self, nb_recipients_to_predict=10, use_customed_tokens=True, agg_func='max', ngram_max=2, verbose=1):
        """ Constructor of TfIdfModel

        :param nb_recipients_to_predict: number of recipients to predict (10 by default)
        :param use_customed_tokens: boolean to specify whether customed tokens has to be used. If false, sklearn.TfidfVectorizer tokenization is used. (True by default)
        :param agg_function: function to use when aggregating the tfidf representations of emails ('max' or 'mean')
        :param ngram_max: max size of ngram to consider (2 by default)
        :param verbose: integers to control the verbosity (0 for silent mode, 1 by default)
        """
        AbstractModel.__init__(self, nb_recipients_to_predict)
        self.use_customed_tokens = use_customed_tokens

        if agg_func == 'max':
            self.agg_func = np.max
        elif agg_func == 'mean':
            self.agg_func = np.mean
        elif agg_func == 'sum':
            self.agg_func = np.sum
        else:
            raise ValueError("In constructor of TfIdfModel, agg_func should be either 'max' or 'mean'.")

        self.ngram_max = ngram_max
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, self.ngram_max))

        self.training = None
        self.training_info = None
        self.address_books = None
        self.mids_sender_recipient = None

    def fit(self, training, training_info):
        if self.verbose:
            print("Start fitting model %s..." % self.__class__.__name__)

        # store training sets
        self.training = training
        self.training_info = training_info

        # compute tdidf for training set
        if self.use_customed_tokens:
            self.training_matrix = self.tfidf_vectorizer.fit_transform(training_info["tokens"])
        else:
            self.training_matrix = self.tfidf_vectorizer.fit_transform(training_info["body"])

        self.address_books = create_address_books(training, training_info)
        self.mids_sender_recipient = create_dictionary_mids(training, training_info)

    def predict(self, test, test_info):
        if self.verbose:
            print("Start making predictions in model %s..." % self.__class__.__name__)

        if self.use_customed_tokens:
            test_matrix = self.tfidf_vectorizer.transform(test_info["tokens"])
        else:
            test_matrix = self.tfidf_vectorizer.transform(test_info["body"])

        predictions_per_sender = {}
        nb_senders = len(test)
        for idx, row in test.iterrows():
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

                # compute similarities (tfidf vectors are normed -> cosine_similarity is only a dot product)
                similarities = pd.Series(tfidf_mails_sender.dot(tfidf_mail_predict.T).toarray().flatten())

                # loop over sender's address book and for each recipient, find the most similar mail it received
                scores = []
                for recipient, nb_occurrences in self.address_books[sender]:
                    mids_recipient = self.mids_sender_recipient[(sender, recipient)]
                    similarities_recipient = similarities[np.nonzero(mids_sender_series.isin(mids_recipient))[0]]
                    scores.append((recipient, self.agg_func(similarities_recipient)))

                # sort the scores and get the 10 recipients with higher scores
                prediction = [recipient for recipient, score in
                              sorted(scores, key=lambda elt: elt[1], reverse=True)[:self.nb_recipients_to_predict]]
                tfidf_preds.append(prediction)

            predictions_per_sender[sender] = [mids_predict, tfidf_preds]

            if idx % 9 == 0 and self.verbose:
                print(" -> %d/%d predictions completed" % (idx + 1, nb_senders))

        print("\nPredictions are all completed !")
        return predictions_per_sender
