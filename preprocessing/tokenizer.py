import string
import re
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def create_simple_tokens(text, remove_stopwords=True, pos_filtering=False, stemming=True):
    punct = string.punctuation.replace('-', '')

    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)
    # strip extra white space
    text = re.sub(' +', ' ', text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')

    # if text is empty, return an empty list
    if tokens == [""]:
        return []

    if pos_filtering == True:
        # apply POS-tagging
        tagged_tokens = pos_tag(tokens)
        # retain only nouns and adjectives
        tokens = [token for token, tag in tagged_tokens if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']]

    if remove_stopwords:
        stpwds = stopwords.words('english')
        # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]

    if stemming:
        stemmer = PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    return (tokens)
