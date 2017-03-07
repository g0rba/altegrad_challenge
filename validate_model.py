import numpy as np
from tools import load_data
from models import FrequencyModel, TfIdfModel
from validation import KFoldCrossValidation, MAPn

# load data
path_to_data = "../data/"
training, training_info, test, test_info = load_data(path_to_data)

# create model
nb_recipients_to_predict = 10
model = FrequencyModel(nb_recipients_to_predict=nb_recipients_to_predict)

# validation
n_split = 5
cross_validation = KFoldCrossValidation(n_split)

scores = []
for fold_nb, (training_fold, training_info_fold, test_fold, test_info_fold, y_test_fold) in enumerate(
        cross_validation.split(training, training_info)):
    print("\nFold #%d..." % fold_nb)

    # fit model
    model.fit(training_fold, training_info_fold)

    # predict
    predictions_per_sender = model.predict(test_fold, test_info_fold)

    # compute MAP@n
    score = MAPn(predictions_per_sender, y_test_fold, nb_recipients_to_predict)
    scores.append(score)
    print("\nMAP@n for this fold -> %0.5f" % score)

# average scores over folds
final_score = np.mean(scores)
print("\n Final Score -> %0.5f" % final_score)
