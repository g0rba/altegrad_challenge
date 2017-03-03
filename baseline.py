from tools import create_submission, load_data
from models import FrequencyModel, TfIdfModel

# load data
path_to_data = "../data/"
training, training_info, test, test_info = load_data(path_to_data)

# create model
# frequency_model = FrequencyModel(nb_recipients_to_predict=10)
model = TfIdfModel(use_customed_tokens=False, agg_func='mean', ngram_max=2, nb_recipients_to_predict=10)

# fit
model.fit(training, training_info)

# predict
predictions_per_sender = model.predict(test, test_info)

# create submission
folder = "../results/"
create_submission(predictions_per_sender, folder=folder)
