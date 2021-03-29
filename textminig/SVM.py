from modules import *
from display_figures import *

# Naive Bays declaration
def SVM(xtrain_tfidf, data):
	mnb = SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None) 
	mnb.fit(xtrain_tfidf, data.target)
	return (mnb)

# 20 Newest data test 
def SVMTestingData(xtrain_tfidf, dataset_20_train, mydata_test_df, X_test_cv):
	print ("SVM classifier testing...") 	
	clf = SVM(xtrain_tfidf, dataset_20_train)
	y_pred_cv_mnb = clf.predict(X_test_cv) 
	y_test = mydata_test_df.target
	f = open(svm_file, "w")
	f.write("\n ----- \n SVM classifier\n")
	f.write("The output is all of the predictions with test data\n")
	f.write(str(y_pred_cv_mnb))
	f.write("\naccuracy_score : ")
	f.write(str(accuracy_score(y_test, y_pred_cv_mnb)))
	f.write("\nclassification_report : \n")
	f.write(str(classification_report(y_test, y_pred_cv_mnb)))
	conf_mat = confusion_matrix(y_test, y_pred_cv_mnb)
	display_matrix(conf_mat, dataset_20_train, svm_file_img)
	f.close()
	data_compare.append({
		'name':"[ SVM manual test ]",
		'accuracy_score': accuracy_score(y_test, y_pred_cv_mnb),
		'classification_report': classification_report(y_test, y_pred_cv_mnb)
	})
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
	f = open(svm_file, "a")
	f.write("\n\n ------- \nSVM classifier with pipeline \n")
	f.write("\n ----->>>>>>   Accuracy = ")
	f.write(str(acc))
	f.write("\ntime duration : ")
	f.write(str(end - start))
	#print('', , , t, "\n")

	creport = classification_report(mydata_test_df.target, preds, target_names=dataset_20_test.target_names)
	f.write("\n")
	f.write(str(creport))
	#print(creport)
	mx = confusion_matrix(mydata_test_df.target, preds)
	#print (mx)
	f.write("\n \n ---------- \n confusion matrix of test data with preds value")
	f.write(str(mx))
	f.write("\n")
	data_duration_compare.append({
		'name':"[ SVM Pipeline ]",
		'accuracy_score': acc,
		'duration': end-start
	})
	f.close()

# customized test data
def SVMTest(count_vect, tfidf_transformer, clf, data):
	f = open(svm_file, "a")
	f.write("\n \n--------------------- \n SVM test on data :\n")
	f.write("To try to predict the outcome on a new document we need to extract the features using almost the same feature extracting chain as before.")
	f.write(str(docs_new))
	xnew_counts = count_vect.transform(docs_new)
	xnew_tfidf = tfidf_transformer.transform(xnew_counts)
	preds = clf.predict(xnew_tfidf)
	f.write("\n")
	f.write(str(preds))
	f.write("\n")
	#print(preds)
	for doc,pred in zip(docs_new, preds):
		cat = data.target_names[pred] # Converting numeric category to string
		f.write(str(doc))
		f.write(" ===> [ ")
		f.write(str(cat))
		f.write(" ] \n")
	f.close()