
import os

%load_ext autoreload
%autoreload 2

import numpy as np
from matplotlib import pyplot
import os
from copy import deepcopy

from time import time

from math import ceil
from scipy.stats import spearmanr, gamma, poisson
from scipy.sparse import csc_matrix

from anndata import AnnData, read_h5ad
import scanpy as sc
from scanpy import read
import pandas as pd
import random
#import XGBoost


#from torch.utils.data import DataLoader, TensorDataset
#from torch import tensor
#from torch.cuda import is_available
import umap
from sklearn.svm import SVC
from sklearn.utils import all_estimators

import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets


def get_index_positions_2(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list

  
def IntersecOfSets(arr1, arr2, arr3):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)
      
    # Calculates intersection of 
    # sets on s1 and s2
    set1 = s1.intersection(s2)         #[80, 20, 100]
      
    # Calculates intersection of sets
    # on set1 and s3
    result_set = set1.intersection(s3)
      
    # Converts resulting set to list
    final_list = list(result_set)
    return final_list
  
#Import Train Data File and Train Predictions:

# original gene matrix
adata_covid19_train_red = read_h5ad('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Training/adata_covid19_haniffa_red_final_overlap_4209_RELABELED.h5ad')


# proportion estimates for each cell-type

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Output')
cellType_probs_validation_data = read_h5ad('CellO_basic_finalOuput_tunedHyper_validationData.h5ad')
cellType_probs_training_data = read_h5ad('CellO_basic_finalOuput_tunedHyper_trainingData.h5ad')

#Import Test Data File and Test Predictions:

# original gene matrix
adata_covid19_test_data = read_h5ad('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Test/adata_covid19_test_combined_overlap_4209.h5ad')

# proportion estimates for each cell-type
os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Output')
cellType_probs_test_data = read_h5ad('basic1_Cello_prob_test_basic_final.h5ad')


#Split gene matrix into train and test datasets:

random.seed(10)

pat_names = adata_covid19_train_red.obs.patient_id.unique()
n_pat = len(pat_names)
train_pat_index = random.sample(range(0,n_pat), k = round(n_pat*.7))

train_pat = list(pd.DataFrame(pat_names).iloc[train_pat_index][0])
print("Number of training patients:", len(train_pat))

val_pat = list(pat_names)
for index in sorted(train_pat_index, reverse=True):
    del val_pat[index]

val_pat = list(val_pat)
print("Number of validating patients:", len(val_pat))

L = list(adata_covid19_train_red.obs.patient_id)
index_allTrain = list()
index_allVal = list()

for i in range(0,len(train_pat)):
    N = list(np.where(np.isin(L,train_pat[i]))[0])
    index_allTrain = index_allTrain + N

for i in range(0,len(val_pat)):
    N = list(np.where(np.isin(L,val_pat[i]))[0])
    index_allVal = index_allVal + N
    
cellType_probs_validation_data.obs['Patient'] = training_data.obs['patient_id']
cellType_probs_training_data.obs['Patient'] = validation_data.obs['patient_id']

train_cellTypes = list(cellType_probs_training_data.obs['Most specific cell type'].unique())

validation_cellTypes = list(cellType_probs_validation_data.obs['Most specific cell type'].unique())

test_cellTypes = list(cellType_probs_test_data.obs['Most specific cell type'].unique())

cell_types_inter = IntersecOfSets(train_cellTypes, validation_cellTypes, test_cellTypes)

cell_types_inter_prob = ['plasmablast (probability)',
 'platelet (probability)',
 'naive B cell (probability)',
 'common myeloid progenitor (probability)',
 'CD14-positive, CD16-negative classical monocyte (probability)',
 'myeloid dendritic cell, human (probability)',
 'central memory CD8-positive, alpha-beta T cell (probability)',
 'mononuclear cell (probability)',
 'T follicular helper cell (probability)',
 'natural killer cell (probability)']

#Compute average probabilities:
cellType_probs_training_final = cellType_probs_training_data.obs[cell_types_inter_prob]
cellType_probs_training_final['Patient'] = training_data.obs.patient_id

cellType_probs_validation_final = cellType_probs_validation_data.obs[cell_types_inter_prob]
cellType_probs_validation_final['Patient'] = validation_data.obs.patient_id

cellType_probs_test_final = cellType_probs_test_data.obs[cell_types_inter_prob]
cellType_probs_test_final['Patient'] = adata_covid19_test_data.obs.donor

cellType_probs_training_means = cellType_probs_training_final.groupby('Patient').mean()
cellType_probs_validation_means = cellType_probs_validation_final.groupby('Patient').mean()
cellType_probs_test_means = cellType_probs_test_final.groupby('Patient').mean()

# Patient names
train_patients = training_data.obs['patient_id'].unique()
val_patients = validation_data.obs['patient_id'].unique()
test_patients = adata_covid19_test_data.obs['donor'].unique()
# Training cell counts
train_cellType_counts = pd.DataFrame()
for i in range(0, len(train_patients)):
    index = get_index_positions_2(list(cellType_probs_training_final['Patient']), train_patients[i])
    celltype_estimates_train = cellType_probs_training_data.obs['Most specific cell type']
    celltype_estimates_train.index = cellType_probs_training_final['Patient']
    counts = pd.DataFrame(celltype_estimates_train[index].value_counts()).T
    train_cellType_counts = train_cellType_counts.append(counts)

train_cellType_counts.index = train_patients
train_cellType_counts = train_cellType_counts[cell_types_inter] 
##############################################################################

# Validation cell counts
validation_cellType_counts = pd.DataFrame()
for i in range(0, len(val_patients)):
    index = get_index_positions_2(list(cellType_probs_validation_final['Patient']), val_patients[i])
    celltype_estimates_val = cellType_probs_validation_data.obs['Most specific cell type']
    celltype_estimates_val.index = cellType_probs_validation_final['Patient']
    counts = pd.DataFrame(celltype_estimates_val[index].value_counts()).T
    validation_cellType_counts = validation_cellType_counts.append(counts)

validation_cellType_counts.index = val_patients
validation_cellType_counts = validation_cellType_counts[cell_types_inter] 

##############################################################################

# Test cell counts
test_cellType_counts = pd.DataFrame()
for i in range(0, len(test_patients)):
    index = get_index_positions_2(list(cellType_probs_test_final['Patient']), test_patients[i])
    celltype_estimates_test = cellType_probs_test_data.obs['Most specific cell type']
    celltype_estimates_test.index = cellType_probs_test_final['Patient']
    counts = pd.DataFrame(celltype_estimates_test[index].value_counts()).T
    test_cellType_counts = test_cellType_counts.append(counts)

test_cellType_counts.index = test_patients
test_cellType_counts = test_cellType_counts[cell_types_inter]

##############################################################################

# Y
y_train = training_data.obs.Status
y_train.index = training_data.obs.patient_id
index = y_train.index
is_duplicate = index.duplicated(keep="first")
not_duplicate = ~is_duplicate
y_train = y_train[not_duplicate]


y_val = validation_data.obs.Status
y_val.index = validation_data.obs.patient_id
index = y_val.index
is_duplicate = index.duplicated(keep="first")
not_duplicate = ~is_duplicate
y_val = y_val[not_duplicate]


y_test = adata_covid19_test_data.obs.disease
y_test = adata_covid19_test_data.obs.disease
y_test.index = adata_covid19_test_data.obs.donor
index = y_test.index
is_duplicate = index.duplicated(keep="first")
not_duplicate = ~is_duplicate
y_test = y_test[not_duplicate]
y_test = y_test.replace('COVID-19', 'Covid')
y_test = y_test.replace('normal', 'Healthy')

abc_final = AdaBoostClassifier(n_estimators=final_n_estimators,
                         learning_rate=final_learning_rate)
# Train Adaboost Classifer
model_final = abc_final.fit(X_train, y_train)

#Final model: training performance
#Predict the response for validation dataset
y_pred = model_final.predict(X_train)
#importing accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_train, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_train, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_train, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_train, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_train, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_train, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_train, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_train, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_train, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_train, y_pred, average='weighted')))


confusion_train = confusion_matrix(y_pred, y_train)
print('Confusion Matrix\n')
pd.DataFrame(confusion_train)

#Final model: validation performance
#Predict the response for validation dataset
y_pred = model_final.predict(X_val)

#importing accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_val, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_val, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_val, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_val, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_val, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_val, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_val, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_val, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_val, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_val, y_pred, average='weighted')))

confusion_val = confusion_matrix(y_pred, y_val)
print('Confusion Matrix\n')
pd.DataFrame(confusion_val)

#Run the final model on the test dataset: test performance

#Predict the response for test dataset

y_pred = model_final.predict(X_test)

#importing accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

confusion_test = confusion_matrix(y_pred, y_test)
print('Confusion Matrix\n')
pd.DataFrame(confusion_test)

equal = [None] * len(y_pred)
for i in range(0,len(y_pred)):
   equal[i] = y_pred[i] == y_test[i]
    
output = pd.DataFrame()
output['True Covid Status'] = y_test
output['Predicted Covid Status'] = y_pred
output['Correct?'] = equal
print(output)
