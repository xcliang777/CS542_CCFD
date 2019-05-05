
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import itertools


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


# import CSV data
data = pd.read_csv("creditcard.csv")


# Normalize the 'Amount' feature and remove unwanted 'Time' features
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)


# Solving sample imbalance problem by using downsampling strategy
X = data.ix[:, data.columns != 'Class']
Y = data.ix[:, data.columns == 'Class']


# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)


# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index


# Out of the indices we picked, randomly select "x" number(number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)


# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])


# Under sample dataset
under_sample_data = data.iloc[under_sample_indices, :]


X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
Y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# Split data into trainging set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


print("")
print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total member of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
print("")
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(X_undersample, Y_undersample, test_size=0.3, random_state=0)
print("Number transactions undersample train dataset: ", len(X_train_undersample))
print("Number transactions undersample test dataset: ", len(X_test_undersample))
print("Total member of undersample transactions: ", len(X_train_undersample)+len(X_test_undersample))
print("")


#clf = RandomForestClassifLier(random_state=42)
clf = LogisticRegression()
clf = clf.fit(X_train_undersample,Y_train_undersample)

y_test_hat = clf.predict(X_train_undersample)

print(metrics.classification_report(Y_train_undersample, y_test_hat))
print(metrics.confusion_matrix(Y_train_undersample, y_test_hat))

cm = confusion_matrix(Y_train_undersample, y_test_hat)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(5,5))
#plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
class_names = [0, 1]
plot_confusion_matrix(cm, class_names, title='Normalized confusion matrix')
plt.show()
######################################################################
y_test_hata = clf.predict(X_test_undersample)

print(metrics.classification_report(Y_test_undersample, y_test_hata))
print(metrics.confusion_matrix(Y_test_undersample, y_test_hata))

cm = confusion_matrix(Y_test_undersample, y_test_hata)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cm, class_names, title='Normalized confusion matrix')
#plot_confusion_matrix(cm, title='Normalized confusion matrix')
plt.show()


from imblearn.over_sampling import SMOTE
oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_sample(X_train, Y_train)
print(np.shape(os_features))
clf = clf.fit(os_features,os_labels)
y_test_hato = clf.predict(os_features)
cm = confusion_matrix(os_labels, y_test_hato)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(5,5))
#plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plot_confusion_matrix(cm, class_names, title='Normalized confusion matrix')
plt.show()

y_test_hate = clf.predict(X_test_undersample)

print(metrics.classification_report(Y_test_undersample, y_test_hate))
print(metrics.confusion_matrix(Y_test_undersample, y_test_hate))

cm = confusion_matrix(Y_test_undersample, y_test_hate)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cm_normalized, class_names, title='Normalized confusion matrix')
#plot_confusion_matrix(cm, title='Normalized confusion matrix')
plt.show()









