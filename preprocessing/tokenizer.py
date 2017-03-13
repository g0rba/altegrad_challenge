import string
import re
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def clean_text_regex(text):
    """ Tool used to clean a text by applying standard regex """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def create_simple_tokens(text, remove_stopwords=True, pos_filtering=False, stemming=True, use_regex=True):
    if use_regex:
        text = clean_text_regex(text)

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
