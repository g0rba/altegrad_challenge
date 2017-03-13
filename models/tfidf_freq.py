from .abstract_model import AbstractModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import create_address_books, create_dictionary_mids
import numpy as np
import pandas as pd
import pickle
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class TfIdfFreqModel(AbstractModel):
    def __init__(self, nb_recipients_to_predict=10, use_customed_tokens=True, agg_func='mean', ngram_max=2, verbose=1,
                 use_pretrained_model=False):
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
        else:
            raise ValueError("In constructor of TfIdfModel, agg_func should be either 'max' or 'mean'.")

        self.ngram_max = ngram_max
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, self.ngram_max))

        self.training = None
        self.training_info = None
        self.address_books = None
        self.mids_sender_recipient = None

        self.use_pretrained_model = use_pretrained_model

    def predict_function(self, data=None, file_name=None):
        if data == None:
            data = pickle.load(open(file_name, "rb"))
        X, Y = data
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        del X[0]
        X.columns = ['sim', 'freq']
        Y.columns = ['dest']

        if (Y['dest'] - Y['dest'].mean()).sum() == 0:
            f_pred = lambda x, y: Y['dest'].mean()

        else:
            ind_y0 = Y[Y['dest'] == 0].index
            ind_y1 = Y[Y['dest'] == 1].index

            if len(ind_y1) < len(ind_y0):
                s_ind = X.loc[ind_y0].sample(len(ind_y1)).index
                n_ind = s_ind.union(ind_y1)

                X = X.loc[n_ind]
                Y = Y.loc[n_ind]

            mu_sim = X['sim'].mean()
            std_sim = X['sim'].std()
            mu_freq = X['freq'].mean()
            std_freq = X['freq'].std()

            if std_sim == 0:
                std_sim = 1
            if std_freq == 0:
                std_freq = 1

            X['freq'] = (X['freq'] - mu_freq) / std_freq
            X['sim'] = (X['sim'] - mu_sim) / std_sim

            clf = QuadraticDiscriminantAnalysis()
            clf.fit(X.values, Y['dest'].values)

            def f(sim, freq):
                sim_normed = (sim - mu_sim) / std_sim
                freq_normed = (freq - mu_freq) / std_freq
                return clf.predict_proba([(sim_normed, freq_normed)])[0][1]

            f_pred = f
        return f_pred

    def fit(self, training, training_info):
        if self.verbose:
            print("Start fitting model %s..." % self.__class__.__name__)

        # store training sets
        self.training = training
        self.training_info = training_info

        if self.use_customed_tokens:
            self.training_matrix = self.tfidf_vectorizer.fit_transform(training_info["tokens"])
        else:
            self.training_matrix = self.tfidf_vectorizer.fit_transform(training_info["body"])

        if self.use_pretrained_model:
            self.address_books = pickle.load(open("temp/address_book.p", "rb"))
        else:
            self.address_books = create_address_books(training, training_info)
            pickle.dump(self.address_books, open("temp/address_book.p", "wb"))

        if self.use_pretrained_model:
            self.mids_sender_recipient = pickle.load(open("temp/mids_sender_recipient.p", "rb"))
        else:
            self.mids_sender_recipient = create_dictionary_mids(training, training_info)
            pickle.dump(self.mids_sender_recipient, open("temp/mids_sender_recipient.p", "wb"))

        # now we create new data: we want to learn the right threshold between tfidf and frequency

        if self.verbose:
            print("We record for each message of each sender tfidf, frequency, receivers")

        if self.use_pretrained_model:
            self.f = pickle.load(open("temp/prediction_function.p", "rb"))
        else:
            nb_senders = len(training)
            dic_f = {}
            for idx, row in training.iterrows():
                # retrieve sender attributes
                sender = row[0]
                mids_sender = self.training[self.training["sender"] == sender]["mids"].values[0]
                mids_sender = np.array(mids_sender.split(" "), dtype=int)
                mids_sender_series = pd.Series(mids_sender)
                position_mails_training = self.training_info[self.training_info["mid"].isin(mids_sender)].index.values

                # get IDs of the emails for which recipient prediction is needed
                mids_predict = np.array(row[1].split(" "), dtype=int)

                # We record the frequency and tfidf in X_sender, and in y_sender if the recipient is one of the true
                x_sender = []
                y_sender = []

                for mid_predict in mids_predict:
                    # get true recipients of the mail
                    true_recipients = training_info[training_info["mid"] == mid_predict]['recipients'].values[0]
                    true_recipients = np.array(true_recipients.split(" "))

                    position_mail_predict = training_info[training_info["mid"] == mid_predict].index[0]
                    # We remove the mail to predict from the similarity matrix                    
                    tfidf_mails_sender = self.training_matrix[
                        np.setdiff1d(position_mails_training, position_mail_predict)]
                    tfidf_mail_predict = self.training_matrix[position_mail_predict]

                    # compute similarities (tfidf vectors are normed -> cosine_similarity is only a dot product)
                    similarities = pd.Series(tfidf_mails_sender.dot(tfidf_mail_predict.T).toarray().flatten())

                    # loop over sender's address book and for each recipient, find the most similar mail it received
                    for recipient, nb_occurrences in self.address_books[sender]:
                        mids_recipient = self.mids_sender_recipient[(sender, recipient)]
                        similarities_recipient = similarities[mids_sender_series.isin(mids_recipient)]

                        if len(similarities_recipient) != 0:
                            x_sender.append((recipient, self.agg_func(similarities_recipient), nb_occurrences))
                            y_sender.append(recipient in true_recipients)

                pickle.dump((x_sender, y_sender), open("temp/training_set_by_sender" + str(idx) + ".p", "wb"))

                dic_f[sender] = self.predict_function(data=(x_sender, y_sender))

                if idx % 9 == 0 and self.verbose:
                    print(" -> %d/%d training completed" % (idx + 1, nb_senders))

            self.f = dic_f
            pickle.dump(dic_f, open("temp/prediction_function.p", "wb"))

        if self.verbose:
            print("Training set built")

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
                    scores.append((recipient, self.f[sender](self.agg_func(similarities_recipient), nb_occurrences)))

                # sort the scores and get the 10 recipients with higher scores
                prediction = [recipient for recipient, score in
                              sorted(scores, key=lambda elt: elt[1], reverse=True)[:self.nb_recipients_to_predict]]

                tfidf_preds.append(prediction)

            predictions_per_sender[sender] = [mids_predict, tfidf_preds]

            if idx % 9 == 0 and self.verbose:
                print(" -> %d/%d predictions completed" % (idx + 1, nb_senders))

        print("\nPredictions are all completed !")
        return predictions_per_sender
