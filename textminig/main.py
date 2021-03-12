# import the modules we need
from modules import *
from data_config import *
from display_figures import *
from data_parse import *
from NaiveBays import *
from SVM import *

# load dataset training
dataset_20_train = loading_data('train')
# get lenght and names of the data
len_data, names = size_and_names_dataset(dataset_20_train)
# get the frequency of each data
targets, frequency = finding_frequency_each_data(dataset_20_train)
# print : 
targets_str = np.array(dataset_20_train.target_names)
# print(list(zip(targets_str, frequency)))
# disply graph of the data
# diplay_figure_1(dataset_20_train)


# load dataset test
dataset_20_test = loading_data('test')
len_data_test, names_test = size_and_names_dataset(dataset_20_test)
targets_test, frequency_test = finding_frequency_each_data(dataset_20_test)
targets_test_str = np.array(dataset_20_test.target_names)
# print(list(zip(targets_test_str, frequency_test)))
# disply graph of the data
# diplay_figure_1(dataset_20_test)

# create a frame with our data
mydata_train_df = data_frame(dataset_20_train)
# filter the data with lowering all the data and delete the ponctuation
mydata_train_df['data'] = mydata_train_df.data.map(alphanumeric).map(punc_lower)
#print (mydata_train_df.head())


# create a frame with our data
mydata_test_df = data_frame(dataset_20_test)
# filter the data with lowering all the data and delete the ponctuation
mydata_test_df['data'] = mydata_test_df.data.map(alphanumeric).map(punc_lower)
#print (mydata_test_df.head())

# tokenizing and filtering of stopwords	
X_train_cv, X_test_cv, X_train_cv_df, X_test_cv_df, count_vect = tokenize_text(mydata_train_df, mydata_test_df)
#print (X_train_cv_df.head())

# Creating a document-term matrix using TF-IDF
X_train_tfidfV, X_test_tfidfV, X_train_tfidfV_df = TfidfVectorizer_text(mydata_train_df, mydata_test_df)
#print (X_train_tfidfV_df.head())

#tf-id transform
xtrain_tfidf, tfidf_transformer = Tfid_transform(X_train_cv)
#print (xtrain_tfidf.shape)


# using naïve Bayes classifier test data
print ("\n\n Naive Bays \n ----------------------- \n ")
clf = NaivBaysTestingData(xtrain_tfidf, dataset_20_train, mydata_test_df, X_test_cv)

# using pipeline  test data
NaivBaysPipline(mydata_train_df, mydata_test_df, dataset_20_test)

#test customized : 	
NaivBaysTest(count_vect, tfidf_transformer, clf, dataset_20_train)

# ---------------------------------------------------------------

# Evaluation of the performance on the test set using SVM Classifier
print ("\n\n SVM \n ----------------------- \n ")
clf2 = SVMTestingData(xtrain_tfidf, dataset_20_train, mydata_test_df, X_test_cv)

# using pipeline  test data
SVMTestingPipeline(mydata_train_df, mydata_test_df, dataset_20_test)

#test customized : 	
SVMTest(count_vect, tfidf_transformer, clf2, dataset_20_train)


# ---------------------------------------------------------------

# Evaluation of the performance on the test set using SVM Classifier
print ("\n\n SVM \n ----------------------- \n ")
clf2 = SVMTestingData(xtrain_tfidf, dataset_20_train, mydata_test_df, X_test_cv)

# using pipeline  test data
SVMTestingPipeline(mydata_train_df, mydata_test_df, dataset_20_test)

#test customized : 	
SVMTest(count_vect, tfidf_transformer, clf2, dataset_20_train)
