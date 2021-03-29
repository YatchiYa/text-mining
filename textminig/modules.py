

from sklearn.naive_bayes import MultinomialNB
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

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from termcolor import colored
import time
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

data_compare = []
data_duration_compare = []
docs_new = ['God is love', 'OpenGL on the GPU is fast', 'I am God loving if not God fearing', 'I like sports', 'GeForce card']

svm_file = "result/SVM_classifier.txt"
naive_bays_file = "result/NB_classifier.txt"
knn_file = "result/KNN_classifier.txt"
lr_file = "result/logistic_regression_classifier.txt"

svm_file_img = "result/img_SVM_classifier.png"
naive_bays_file_img = "result/img_NB_classifier.png"
knn_file_img = "result/img_KNN_classifier.png"
lr_file_img = "result/img_logistic_regression_classifier.png"