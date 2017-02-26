import time
import pandas as pd
import numpy as np
from collections import Counter
import operator


def load_data(path_to_data="../data/"):
    training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
    training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
    test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
    test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

    return training, training_info, test, test_info


def create_submission(predictions_per_sender, folder="../results", file_name=None):
    """ Create the submission file from a dictionary of predictions

    predictions_per_sender: dictionary of predictions {<email of sender>: <list of 10 email recipients>}
    """

    if file_name is None:
        file_name = "%s/submission_%s.txt" % (folder, time.strftime("%Y%m%d_%H%M%S"))

    with open(file_name, 'wb') as my_file:
        my_file.write(b'mid,recipients\n')
        for sender, preds in predictions_per_sender.items():
            ids = preds[0]
            predictions = preds[1]
            for index, my_preds in enumerate(predictions):
                s_to_write = str(ids[index]) + ',' + ' '.join(my_preds) + '\n'
                my_file.write(s_to_write.encode())


def create_address_books(training, training_info):
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
        recipients_sender = [recipient for recipients in messages_sender["recipients"] for recipient in
                             recipients.split(" ") if "@" in recipient]
        # compute occurrence of recipients
        rec_occ = dict(Counter(recipients_sender))
        # order by frequency
        sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
        # save into dictionary
        address_books[sender] = sorted_rec_occ

    return address_books


def cosine_similarity(u, v, eps=1e-10):
    # flat arrays
    u_flatten = u.flatten()
    v_flatten = v.flatten()

    # compute norms
    u_norm = np.linalg.norm(u, ord=2)
    v_norm = np.linalg.norm(v, ord=2)

    # compute similarity (small epsilon to avoid dividing by 0)
    similarity = np.dot(u_flatten, v_flatten) / (u_norm * v_norm + eps)

    return similarity
