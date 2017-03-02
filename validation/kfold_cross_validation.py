import pandas as pd
import numpy as np


class KFoldCrossValidation:
    def __init__(self, n_split=5):
        self.n_split = n_split

    def split(self, training, training_info):
        """ Generate training and test datasets in a k-fold cross validation way """

        # create training_fold and test_fold, training and test datasets for current fold
        training_fold = training.copy()
        test_fold = training.copy()

        # create a linear sample of [0, 1] os step 1 / n_split
        percentages = np.linspace(0, 1, self.n_split + 1)

        # retrieve list of message ids
        total_mids = training["mids"].apply(lambda mids: mids.split(" "))

        for fold_nb in range(self.n_split):
            min_percentage = percentages[fold_nb]
            max_percentage = percentages[fold_nb + 1]

            # retrieve mids for training and test datasets
            training_mids = total_mids.apply(
                lambda mids: mids[np.floor(min_percentage * len(mids)).astype(int): np.floor(
                    max_percentage * len(mids)).astype(int)])
            test_mids = total_mids.apply(
                lambda mids: np.concatenate([mids[:np.floor(min_percentage * len(mids)).astype(int)],
                                             mids[np.floor(max_percentage * len(mids)).astype(int):]]).tolist())

            # store mids in corresponding datasets (mids are joined in an unique string to mimicate training and test format)
            training_fold["mids"] = training_mids.apply(lambda mids: " ".join(mids))
            test_fold["mids"] = test_mids.apply(lambda mids: " ".join(mids))

            # create a list containing all training mids, and one for all test mids
            flatten_training_mids = [int(mid) for mids in training_mids for mid in mids]
            flatten_test_mids = [int(mid) for mids in test_mids for mid in mids]

            # create training_info_fold and test_info_fold, by slicing original training_info
            training_info_fold = training_info[training_info["mid"].isin(flatten_training_mids)]
            test_info_fold = training_info[training_info["mid"].isin(flatten_test_mids)]

            # remove recipients from test_info_fold
            test_info_fold = test_info_fold[[column for column in test_info_fold if column != "recipients"]]

            # yield training and test datasets for current fold
            yield training_fold, training_info_fold, test_fold, test_info_fold
