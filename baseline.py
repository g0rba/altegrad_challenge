import operator
from collections import Counter

from tools import create_submission, load_data, create_address_books
from models import FrequencyModel

# load data
path_to_data = "../data/"
training, training_info, test, test_info = load_data(path_to_data)

# create model
frequency_model = FrequencyModel(nb_recipients_to_predict=10)

# fit
frequency_model.fit(training, training_info)

# predict
predictions_per_sender = frequency_model.predict(test, test_info)

# create submission
folder = "../results/"
create_submission(predictions_per_sender, folder=folder)
