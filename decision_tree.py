import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.tree import DecisionTreeClassifier


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	:param cm:
	:param classes:
	:param title:
	:param cmap:
	:return:
	"""
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

def dt_under(undersample_X, undersample_Y, test_X, test_Y):
	train_under = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
	train_under = train_under.fit(undersample_X, undersample_Y)

	y_train = train_under.predict(undersample_X)

	cm = confusion_matrix(undersample_Y, y_train)
	# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(5, 5))
	# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
	class_names = [0, 1]
	plot_confusion_matrix(cm, class_names, title='Decision Tree: Training Result for undersampled dataset')
	plt.show()

	y_test = train_under.predict(test_X)

	cm = confusion_matrix(test_Y, y_test)
	# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(5, 5))
	# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
	class_names = [0, 1]
	plot_confusion_matrix(cm, class_names, title='Decision Tree: Testing Result for undersampled dataset')
	plt.show()
#######################################################################


def dt_over(oversample_X, oversample_Y, test_X, test_Y):
	train_under = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
	train_under = train_under.fit(oversample_X, oversample_Y)

	y_train = train_under.predict(oversample_X)


	cm = confusion_matrix(oversample_Y, y_train)
	# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(5, 5))
	# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
	class_names = [0, 1]
	plot_confusion_matrix(cm, class_names, title='Decision Tree: Training Result for oversampled dataset')
	plt.show()

	y_test = train_under.predict(test_X)


	cm = confusion_matrix(test_Y, y_test)
	# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(5, 5))
	# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
	class_names = [0, 1]
	plot_confusion_matrix(cm, class_names, title='Decision Tree: Testing Result for oversampled dataset')
	plt.show()
###################################################################

def dt_raw(raw_X, raw_Y, test_X, test_Y):
	train_under = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
	train_under = train_under.fit(raw_X, raw_Y)

	y_train = train_under.predict(raw_X)


	cm = confusion_matrix(raw_Y, y_train)
	# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(5, 5))
	# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
	class_names = [0, 1]
	plot_confusion_matrix(cm, class_names, title='Decision Tree: Training Result for raw dataset')
	plt.show()

	y_test = train_under.predict(test_X)

	cm = confusion_matrix(test_Y, y_test)
	# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(5, 5))
	# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
	class_names = [0, 1]
	plot_confusion_matrix(cm, class_names, title='Decision Tree: Testing Result for raw dataset')
	plt.show()



