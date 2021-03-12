from modules import *

# Naive Bays declaration
def SVM(xtrain_tfidf, data):
	mnb = SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None) 
	mnb.fit(xtrain_tfidf, data.target)
	return (mnb)

# 20 Newest data test 
def SVMTestingData(xtrain_tfidf, dataset_20_train, mydata_test_df, X_test_cv):
	print ("SVM classifier") 	
	clf = SVM(xtrain_tfidf, dataset_20_train)
	y_pred_cv_mnb = clf.predict(X_test_cv) 
	print ("The output is all of the predictions with test data")
	print (y_pred_cv_mnb)
	y_test = mydata_test_df.target
	print("accuracy_score : ", accuracy_score(y_test, y_pred_cv_mnb))
	print("classification_report : \n", classification_report(y_test, y_pred_cv_mnb))
	conf_mat = confusion_matrix(y_test, y_pred_cv_mnb)
	#display_matrix(conf_mat, dataset_20_train)
	return (clf)

# using pipeline
def SVMTestingPipeline(mydata_train_df, mydata_test_df, dataset_20_test):
	start = time.time()
	svm_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
	])
	svm_clf.fit(mydata_train_df.data, mydata_train_df.target)  

	preds = svm_clf.predict(mydata_test_df.data)
	end = time.time()
	acc = np.mean(preds == mydata_test_df.target)
	print('\n ----->>>>>>   Accuracy = ', acc, "time duration : ", end-start, "\n")

	creport = classification_report(mydata_test_df.target, preds, target_names=dataset_20_test.target_names)
	print(creport)
	mx = confusion_matrix(mydata_test_df.target, preds)
	print (mx)

# customized test data
def SVMTest(count_vect, tfidf_transformer, clf, data):
	print ("To try to predict the outcome on a new document we need to extract the features using almost the same feature extracting chain as before.")
	print ("list = ['God is love', 'OpenGL on the GPU is fast', 'I am God loving if not God fearing', 'I like sports', 'GeForce card']")
	docs_new = ['God is love', 'OpenGL on the GPU is fast', 'I am God loving if not God fearing', 'I like sports', 'GeForce card']
	xnew_counts = count_vect.transform(docs_new)
	xnew_tfidf = tfidf_transformer.transform(xnew_counts)
	preds = clf.predict(xnew_tfidf)
	print(preds)
	for doc,pred in zip(docs_new, preds):
	    cat = data.target_names[pred] # Converting numeric category to string
	    print('%r => %s' % (doc, cat))