import operator
from collections import Counter

from tools import create_submission, load_data, create_address_books

##########################
# load some of the files #
##########################

path_to_data = "../data/"
training, training_info, test, test_info = load_data(path_to_data)

################################
# create some handy structures #                    
################################

address_books = create_address_books(training, training_info)

#############
# baselines #                           
#############

# will contain email ids, predictions for random baseline, and predictions for frequency baseline
predictions_per_sender = {}

# number of recipients to predict
k = 10

for _, row in test.iterrows():
    name_ids = row.tolist()
    sender = name_ids[0]
    # get IDs of the emails for which recipient prediction is needed
    ids_predict = name_ids[1].split(' ')
    ids_predict = [int(my_id) for my_id in ids_predict]
    freq_preds = []
    # select k most frequent recipients for the user
    k_most = [elt[0] for elt in address_books[sender][:k]]
    for id_predict in ids_predict:
        # for the frequency baseline, the predictions are always the same
        freq_preds.append(k_most)
    predictions_per_sender[sender] = [ids_predict, freq_preds]

#################################################
# write predictions in proper format for Kaggle #                           
#################################################

folder = "../results/"
create_submission(predictions_per_sender, folder=folder)
