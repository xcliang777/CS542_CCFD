<<<<<<< HEAD
=======

>>>>>>> dcc5d71e6aeb21446f24b5587adedeb966330edd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
<<<<<<< HEAD
import itertools

=======

import itertools
>>>>>>> dcc5d71e6aeb21446f24b5587adedeb966330edd

#train with undersampling data
def LR_under(X_train_undersample, Y_train_undersample, X_test, Y_test):
	lr = LogisticRegression(C=0.01, solver='liblinear', penalty='l2')
	lr.fit(X_train_undersample, Y_train_undersample.values.ravel())

	#predict
	Y_pred_training = lr.predict(X_train_undersample)
	Y_pred = lr.predict(X_test.values)

	# Compare confusion matrix
	train_matrix = confusion_matrix(Y_train_undersample, Y_pred_training)
	test_matrix = confusion_matrix(Y_test, Y_pred)
	np.set_printoptions(precision=2)

	print("Recall metric in the testing dataset(training undersampling data): ", test_matrix[1, 1]/(test_matrix[1, 0]+test_matrix[1, 1]))

	# Plot non-normalized confusion matrix
	class_names = [0, 1]

	plt.figure()
	plot_confusion_matrix(train_matrix,
<<<<<<< HEAD
						  classes=class_names,
						  title='Logistic Regression: Training Result for undersampled dataset')
	plt.show()

	plt.figure()
	plot_confusion_matrix(test_matrix,
						  classes=class_names,
=======
						  classes=class_names,
						  title='Logistic Regression: Training Result for undersampled dataset')
	plt.show()

	plt.figure()
	plot_confusion_matrix(test_matrix,
						  classes=class_names,
>>>>>>> dcc5d71e6aeb21446f24b5587adedeb966330edd
						  title='Logistic Regression: Testing Result for undersampled dataset')
	plt.show()

# print("------------------------------------------------------------------------")

#train with oversampling data
def LR_over(X_train_over, y_train_over, X_test, Y_test):


	lr = LogisticRegression(C=0.01, solver='liblinear', penalty='l2')
	lr.fit(X_train_over, y_train_over.ravel())

	#predict
	Y_pred_training = lr.predict(X_train_over)
	Y_pred = lr.predict(X_test.values)

	# Compare confusion matrix
	train_matrix = confusion_matrix(y_train_over, Y_pred_training)
	test_matrix = confusion_matrix(Y_test, Y_pred)
	np.set_printoptions(precision=2)

	print("Recall metric in the testing dataset(training oversampling data): ", test_matrix[1, 1]/(test_matrix[1, 0]+test_matrix[1, 1]))

	# Plot non-normalized confusion matrix
	class_names = [0, 1]

	plt.figure()
	plot_confusion_matrix(train_matrix,
						  classes=class_names,
						  title='Logistic Regression: Training Result for oversampled dataset')
	plt.show()

	plt.figure()
	plot_confusion_matrix(test_matrix,
						  classes=class_names,
						  title='Logistic Regression: Testing Result for oversampled dataset')
	plt.show()



# print("------------------------------------------------------------------------")

#train with all data
def LR_all(X_train, Y_train, X_test, Y_test):
	lr = LogisticRegression(C=10, solver='liblinear', penalty='l2')
	lr.fit(X_train, Y_train.values.ravel())

	Y_pred_training = lr.predict(X_train)
	Y_pred_alldata = lr.predict(X_test.values)


	# Compute confusion matrix
	train_matrix = confusion_matrix(Y_train, Y_pred_training)
	cnf_matrix = confusion_matrix(Y_test, Y_pred_alldata)
	np.set_printoptions(precision=2)

	print("Recall metric in the testing dataset(training all data): ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

	# Plot non-normalized confusion matrix
	class_names = [0, 1]

	plt.figure()
	plot_confusion_matrix(train_matrix,
						  classes=class_names,
						  title='Logistic Regression: Training Result for all dataset')

	plt.show()

	plt.figure()
	plot_confusion_matrix(cnf_matrix,
							classes=class_names,
							title='Logistic Regression: Testing Result for all dataset')

	plt.show()

# print("--------------------------------------------------------------------")
<<<<<<< HEAD
=======

>>>>>>> dcc5d71e6aeb21446f24b5587adedeb966330edd
# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment='center',
				 color='white' if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel("True Label")
	plt.xlabel("Predicted Label")