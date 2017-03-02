from .abstract_model import AbstractModel
from tools import create_address_books, create_dictionary_mids
import numpy as np
from gensim import models, similarities
from gensim.corpora import dictionary
import os.path

class HdpModel(AbstractModel):
    def __init__(self, nb_recipients_to_predict=10, use_pretrained_model=False):
        self.nb_recipients_to_predict = nb_recipients_to_predict
        self.use_pretrained_model = use_pretrained_model

    def fit(self, training, training_info):
        # store training sets
        self.training = training
        self.training_info = training_info
        
        print("creating train tokens")
        train_tokens = training_info["tokens"].apply(lambda tokens: tokens.split(" ")).values.tolist() 
        print("creating train dict")
        train_my_dict = dictionary.Dictionary(train_tokens)
        print("creating train corpus")
        train_corpus = [train_my_dict.doc2bow(token) for token in train_tokens]
        print("training Hdp model")
        if os.path.isfile('../temp/model.hdp') and self.use_pretrained_model: 
            self.hdp = models.HdpModel.load('../temp/model.hdp')
        else:
            self.hdp = models.HdpModel(train_corpus, id2word=train_my_dict)
            self.hdp.save('../temp/model.hdp')
        print("creating train Hdp matrix")
        self.hdp_train_matrix = np.array([self.hdp[document] for document in train_corpus])
        
        self.address_books = create_address_books(training, training_info)
        self.mids_sender_recipient = create_dictionary_mids(training, training_info)


    def predict(self, test, test_info):
        
        print("creating test tokens")
        test_tokens = test_info["tokens"].apply(lambda tokens: tokens.split(" ")).values.tolist()  
        print("creating test dictionnary")        
        test_my_dict = dictionary.Dictionary(test_tokens)
        print("creating test corpus") 
        test_corpus = [test_my_dict.doc2bow(token) for token in test_tokens]
        print("creating hdp test matrix")
        hdp_test_matrix = np.array([self.hdp[doc] for doc in test_corpus])
        
        print("prediction per sender")
        predictions_per_sender = {}
        for nb_done, row in test.iterrows():
            print("Progression: %f"%(nb_done/len(test)))
            # retrieve sender attributes
            sender = row[0]
            mids_sender = self.training[self.training["sender"] == sender]["mids"].values[0]
            mids_sender = np.array(mids_sender.split(" "), dtype=int)
            position_mails_training = self.training_info[self.training_info["mid"].isin(mids_sender)].index.values
            hdp_mails_sender = self.hdp_train_matrix[position_mails_training]

            index = similarities.MatrixSimilarity(hdp_mails_sender,num_features=500)
            
            # This dictionnary is used to recover the positions in the 
            #training set from the position in the similarity matrix
            dict_Sim_Training = dict(zip(position_mails_training, range(len(position_mails_training)))) 
            
            # get IDs of the emails for which recipient prediction is needed
            mids_predict = np.array(row[1].split(" "), dtype=int)

            # initialize list to store predictions
            hdp_preds = []

            for mid_predict in mids_predict:
                # get the position of current mail in test_info dataset
                position_mail_predict = test_info[test_info["mid"] == mid_predict].index.values
                hdp_mail_predict = hdp_test_matrix[position_mail_predict]
                #sims: similarity score ordered by untrained document id
                sims = index[hdp_mail_predict][0]

                scores = []
                for recipient, nb_occurrences in self.address_books[sender]:
                    mids_recipient = self.mids_sender_recipient[(sender, recipient)]
                    positions_mids_recipient = self.training_info[self.training_info["mid"].isin(mids_recipient)].index.values
                    ind_sim = np.array([dict_Sim_Training[ind] for ind in positions_mids_recipient])                
                    similarities_recipient = sims[ind_sim]
                    scores.append((recipient, similarities_recipient.mean()))

                # sort the scores and get the 10 recipients with higher scores
                prediction = [recipient for recipient, score in
                              sorted(scores, key=lambda elt: elt[1], reverse=True)[:self.nb_recipients_to_predict]]
                
                hdp_preds.append(prediction)

            predictions_per_sender[sender] = [mids_predict, hdp_preds]

        return predictions_per_sender
