from .abstract_model import AbstractModel
from tools import create_address_books, create_dictionary_mids
import numpy as np
from gensim import models, similarities
from gensim.corpora import dictionary
import os.path
import pandas as pd

class DeepModel(AbstractModel):
    
    def get_normed_emb_w(self, word):
        idx = self.embeddings_index.get(word)
        if idx is None:
            return 0, np.zeros(100)
        else:
            return 1, self.glove_embeddings_normed[idx]
        
    def get_normed_emb(self, doc):
        n_countable_words = 0
        emb = np.zeros(100)
        for word in doc:
            n, emb_word = self.get_normed_emb_w(word)
            emb += emb_word
            n_countable_words += n
        if n_countable_words ==0:
            return emb
        else:
            return emb/n_countable_words 

    def __init__(self, nb_recipients_to_predict=10):
        self.nb_recipients_to_predict = nb_recipients_to_predict

    def fit(self, training, training_info):
        # store training sets
        self.training = training
        self.training_info = training_info
        
        print("Creating training tokens")
        train_tokens = training_info["tokens"].apply(lambda tokens: tokens.split(" ")).values
        
        # The file glove100K.100d.txt is an extract of Glove Vectors, 
        #that were trained on english Wikipedia 2014 + Gigaword 5 (6B tokens).
        #We extracted the 100 000 most frequent words.
        #They have a dimension of 100
        
        embeddings_index = {}
        embeddings_vectors = []
        print("Loading embedding from Gloves")
        f = open('../glove/glove100K.100d.txt', 'rb')
        
        word_idx = 0
        for line in f:
            values = line.decode('utf-8').split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = word_idx
            embeddings_vectors.append(vector)
            word_idx = word_idx + 1
        f.close()        
        
        self.embeddings_index = embeddings_index
        
        # Stack all embeddings in a large numpy array
        glove_embeddings = np.vstack(embeddings_vectors)
        glove_norms = np.linalg.norm(glove_embeddings, axis=-1, keepdims=True)
        self.glove_embeddings_normed = glove_embeddings / glove_norms
        
        print("Creating training normed embedding matrix")
        train_emb = []
        for tokens in train_tokens:
            train_emb.append(self.get_normed_emb(tokens))
            
        self.train_emb = np.array(train_emb)          
        self.address_books = create_address_books(training, training_info)
        self.mids_sender_recipient = create_dictionary_mids(training, training_info)


    def predict(self, test, test_info):
        
        print("Creating Test tokens")
        test_tokens = test_info["tokens"].apply(lambda tokens: tokens.split(" ")).values
        
        print("Creating testing normed embedding matrix")
        test_emb = []
        for tokens in test_tokens:
            test_emb.append(self.get_normed_emb(tokens))
        
        print("prediction per sender")
        predictions_per_sender = {}
        for nb_done, row in test.iterrows():
            print("Progression: %f"%(nb_done/len(test)))
            # retrieve sender attributes
            sender = row[0]
            mids_sender = self.training[self.training["sender"] == sender]["mids"].values[0]
            mids_sender = np.array(mids_sender.split(" "), dtype=int)
            mids_sender_series = pd.Series(mids_sender)
            position_mails_training = self.training_info[self.training_info["mid"].isin(mids_sender)].index.values
            deep_mails_sender = self.train_emb[position_mails_training]
            
            
            
            # get IDs of the emails for which recipient prediction is needed
            mids_predict = np.array(row[1].split(" "), dtype=int)

            # initialize list to store predictions
            deep_preds = []
            for mid_predict in mids_predict:
                # get the position of current mail in test_info dataset
                position_mail_predict = test_info[test_info["mid"] == mid_predict].index.values
                deep_mail_predict = test_emb[position_mail_predict[0]]
                #sims: similarity score ordered by untrained document id
                sims = deep_mails_sender.dot(deep_mail_predict)
                
                scores = []
                for recipient, nb_occurrences in self.address_books[sender]:
                    mids_recipient = self.mids_sender_recipient[(sender, recipient)]
                    ind_sim = np.nonzero(mids_sender_series.isin(mids_recipient))              
                    similarities_recipient = sims[ind_sim]               
                    scores.append((recipient, similarities_recipient.mean()))

                # sort the scores and get the 10 recipients with higher scores
                prediction = [recipient for recipient, score in
                              sorted(scores, key=lambda elt: elt[1], reverse=True)[:self.nb_recipients_to_predict]]
                
                deep_preds.append(prediction)

            predictions_per_sender[sender] = [mids_predict, deep_preds]

        return predictions_per_sender
