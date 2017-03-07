import numpy as np


def MAPn(predictions_per_sender, y_true, n=10):
    """ Compute mean average precision at n (cf https://www.kaggle.com/wiki/MeanAveragePrecision) from dictionary of predictions and the true recipients """
    # initialize list to store average precisions
    average_precisions = []
    # iterate over the dictionary's items to compute the average precisions
    for sender, (mids, predictions_per_mid) in predictions_per_sender.items():
        for mid, predictions in zip(mids, predictions_per_mid):
            # retrieve true recipients of this email and minimum between n and the number of real recipients
            true_recipients = y_true[mid]
            min_n_m = np.min([len(true_recipients), n])

            # store the temporary precision (before normalisation by min(n,m))
            temp_precision = 0

            # iterate over predictions and add the temporary precisions together
            nb_good_predictions = 0
            for idx_prediction, prediction in enumerate(predictions):
                if prediction in true_recipients:
                    nb_good_predictions += 1
                    temp_precision += nb_good_predictions / (idx_prediction + 1)

            # compute average precision for current email and store it in the list
            average_precisions.append(temp_precision / min_n_m)

    # compute the mean of average precision and return it
    return np.mean(average_precisions)
