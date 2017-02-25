import time
import pandas as pd


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
