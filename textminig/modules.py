

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import nltk

import matplotlib.pyplot as plt

import string
import warnings
import re
import pandas as pd
import random
from sklearn.model_selection import GridSearchCV

#Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups

warnings.filterwarnings("ignore")
try :
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try :
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
