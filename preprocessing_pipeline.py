from tools import load_data
from preprocessing import create_simple_tokens

# load data
path_to_data = "data/"
training, training_info, test, test_info = load_data(path_to_data)

# tokenize emails' bodies (tokens are combined into a single string in order to write in csv files)
print("Tokenize training set...")
training_info["tokens"] = training_info["body"].apply(lambda body: " ".join(create_simple_tokens(body)))
print("Tokenize test set...")
test_info["tokens"] = test_info["body"].apply(lambda body: " ".join(create_simple_tokens(body)))

# write into files
print("Write into csv files...")
training_info.to_csv("data/training_info.csv", sep=',', index=False)
test_info.to_csv("data/test_info.csv", sep=',', index=False)
print("Preprocessing done.")