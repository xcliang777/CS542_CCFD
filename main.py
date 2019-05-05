import random_forest
import logistic_regression as logr


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

print("first enter")
'''
Part 1:  Dataset: Overview
'''

# import CSV data
data = pd.read_csv("creditcard.csv")

print("Number of features: ", len(data.columns)-1)
print('Number of No Fraud transactions: ', data['Class'].value_counts()[0])
print('Number of Fraud transactions: ', data['Class'].value_counts()[1])
print('Percentage of No Fraud transactions: ', round(data['Class'].value_counts()[0]/len(data) * 100,2), '%')
print('Percentage of Fraud transactions: ', round(data['Class'].value_counts()[1]/len(data) * 100,2), '%')

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=data, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()

'''
Part 2:  Dataset: Preprocessing
'''
# 1. Normalize the 'Amount' feature and remove unwanted 'Time' features
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
X = data.ix[:, data.columns != 'Class']
Y = data.ix[:, data.columns == 'Class']

# 2. Split dataset into train test part
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 3. Preparing Undersampling training dataset using random undersampling algorithm
# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

undersample_X = under_sample_data.ix[:, under_sample_data.columns != 'Class']
undersample_Y = under_sample_data.ix[:, under_sample_data.columns == 'Class']

X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(undersample_X, undersample_Y, test_size=0.3, random_state=0)
# Showing ratio
print("--------------------------------------------------------------------------------------------------------------")
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

print("--------------------------------------------------------------------------------------------------------------")
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=undersample_Y, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()


# 4. Preparing Oversampling training dataset using SMOTE algorithm
oversampler = SMOTE(random_state=0)
oversample_X, oversample_Y = oversampler.fit_sample(X_train, Y_train)
print("--------------------------------------------------------------------------------------------------------------")
print("Percentage of normal transactions: ", len([x for x in oversample_Y if x == 0])/len(oversample_Y))
print("Percentage of normal transactions: ", len([x for x in oversample_Y if x == 1])/len(oversample_Y))
print("Total number of oversampled transactions: ", len(oversample_Y))


'''
Part 3:  Method: Logical Regression
'''
logr.LR_under(undersample_X, undersample_Y, X_test_undersample, Y_test_undersample)
logr.LR_over(oversample_X, oversample_Y, X_test_undersample, Y_test_undersample)
logr.LR_all(X_train, Y_train, X_test_undersample, Y_test_undersample)

'''
Part 4:  Method: Random Forest
'''
random_forest.rf_under(undersample_X, undersample_Y, X_test_undersample, Y_test_undersample)
random_forest.rf_over(oversample_X, oversample_Y, X_test_undersample, Y_test_undersample)
random_forest.rf_raw(X_train, Y_train, X_test_undersample, Y_test_undersample)




