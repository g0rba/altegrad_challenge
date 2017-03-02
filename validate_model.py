from tools import create_submission, load_data
from models import FrequencyModel
from validation import KFoldCrossValidation

# load data
path_to_data = "../data/"
training, training_info, test, test_info = load_data(path_to_data)

# create model
frequency_model = FrequencyModel(nb_recipients_to_predict=10)

# validation
n_split = 5
cross_validation = KFoldCrossValidation(n_split)

for fold_nb, (training_fold, training_info_fold, test_fold, test_info_fold, y_test_fold) in enumerate(
        cross_validation.split(training, training_info)):
    print("\nFold #%d..." % fold_nb)

    # fit model
    frequency_model.fit(training_fold, training_info_fold)

    # predict
    frequency_model.predict(test_fold, test_info_fold)
