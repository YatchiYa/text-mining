from modules import *

# # Text preprocessing steps - remove numbers, captial letters and punctuation
#define the alphanumeric function
alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)

# define the ponctuation and to lower the function
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

def data_frame(data):
	return (pd.DataFrame({'data': data.data, 'target': data.target}))


def tokenize_text(mydata_train_df, mydata_test_df):
	count_vect = CountVectorizer(stop_words='english')
	X_train_cv = count_vect.fit_transform(mydata_train_df.data)  # fit_transform learns the vocab and one-hot encodes
	X_test_cv = count_vect.transform(mydata_test_df.data) # transform uses the same vocab and one-hot encodes

	X_train_cv_df = pd.DataFrame(X_train_cv.todense())
	X_train_cv_df.columns = sorted(count_vect.vocabulary_)

	X_test_cv_df = pd.DataFrame(X_test_cv.todense())
	X_test_cv_df.columns = sorted(count_vect.vocabulary_)
	return (X_train_cv, X_test_cv, X_train_cv_df, X_test_cv_df, count_vect)

def TfidfVectorizer_text(mydata_train_df, mydata_test_df):
	tfidfV = TfidfVectorizer(stop_words='english') 
	# tfidfV = TfidfVectorizer(ngram_range=(1, 2), binary =True, stop_words='english') 

	X_train_tfidfV = tfidfV.fit_transform(mydata_train_df.data) # fit_transform learns the vocab and one-hot encodes 
	X_test_tfidfV = tfidfV.transform(mydata_test_df.data) # transform uses the same vocab and one-hot encodes 

	X_train_tfidfV_df = pd.DataFrame(X_train_tfidfV.todense())
	X_train_tfidfV_df.columns = sorted(tfidfV.vocabulary_)

	return (X_train_tfidfV, X_test_tfidfV, X_train_tfidfV_df)


def Tfid_transform(xtrain_counts):	
	tfidf_transformer = TfidfTransformer()
	xtrain_tfidf = tfidf_transformer.fit_transform(xtrain_counts)
	return (xtrain_tfidf, tfidf_transformer)