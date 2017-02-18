import random
import operator
import pandas as pd
from collections import Counter

path_to_data = "../data/"

##########################
# load some of the files #                           
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

################################
# create some handy structures #                    
################################

# convert training set to dictionary
emails_ids_per_sender = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1:][0].split(' ')
    emails_ids_per_sender[sender] = ids

# save all unique sender names
all_senders = emails_ids_per_sender.keys()

# create address book with frequency information for each user
address_books = {}
for sender, email_ids_sender in emails_ids_per_sender.items():
    # cast email ids into int
    list_ids_sender = [int(id_) for id_ in email_ids_sender]
    # get subset of messages sent by this user
    messages_sender = training_info[training_info["mid"].isin(list_ids_sender)]
    # retrieve recipients of those messages
    recipients_sender = [recipient for recipients in messages_sender["recipients"].values.tolist() for recipient in recipients.split(" ") if "@" in recipient]
    # compute occurence of recipients
    rec_occ = dict(Counter(recipients_sender))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
    # save into dictionnary
    address_books[sender] = sorted_rec_occ

# save all unique recipient names    
all_recs = list(set([elt[0] for sublist in address_books.values() for elt in sublist]))

# save all unique user names 
all_users = []
all_users.extend(all_senders)
all_users.extend(all_recs)
all_users = list(set(all_users))

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
    random_preds = []
    freq_preds = []
    # select k most frequent recipients for the user
    k_most = [elt[0] for elt in address_books[sender][:k]]
    for id_predict in ids_predict:
        # select k users at random
        random_preds.append(random.sample(all_users, k))
        # for the frequency baseline, the predictions are always the same
        freq_preds.append(k_most)
    predictions_per_sender[sender] = [ids_predict, random_preds, freq_preds]

#################################################
# write predictions in proper format for Kaggle #                           
#################################################

path_to_results = "../results/"

with open(path_to_results + 'predictions_random.txt', 'wb') as my_file:
    my_file.write(b'mid,recipients\n')
    for sender, preds in predictions_per_sender.items():
        ids = preds[0]
        random_preds = preds[1]
        for index, my_preds in enumerate(random_preds):
            s_to_write = str(ids[index]) + ',' + ' '.join(my_preds) + '\n'
            my_file.write(s_to_write.encode())

with open(path_to_results + 'predictions_frequency.txt', 'wb') as my_file:
    my_file.write(b'mid,recipients\n')
    for sender, preds in predictions_per_sender.items():
        ids = preds[0]
        freq_preds = preds[2]
        for index, my_preds in enumerate(freq_preds):
            s_to_write = str(ids[index]) + ',' + ' '.join(my_preds) + '\n'
            my_file.write(s_to_write.encode())