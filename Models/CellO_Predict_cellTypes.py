
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
import anndata as ad

from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
from torch.cuda import is_available

import umap

adata_covid19_train_red = read_h5ad('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Training/adata_covid19_haniffa_red_final_5000.h5ad')

adata_covid19_test_adaptive = sc.read_h5ad("/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Test/adata_covid19_test_adaptive_RELABELED.h5ad")

adata_covid19_test_innate = sc.read_h5ad("/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Test/adata_covid19_test_innate_RELABELED_CORRECT.h5ad")

# combine test data sets
import anndata as ad
combined_test = ad.concat([adata_covid19_test_adaptive, adata_covid19_test_innate], join="inner")

########
#Checking overlapping genes
########

test_geneset = list(adata_covid19_test_adaptive.var['feature_name'])
train_geneset = list(adata_covid19_train_red.var_names)
overlapping_genes = set.intersection(set(test_geneset), set(train_geneset))

combined_test.var_names = list(adata_covid19_test_adaptive.var['feature_name'])
all_genes = adata_covid19_test_adaptive.var['feature_name']
index_overlapping_genes = all_genes.isin(overlapping_genes)
test_overlapHVGs = combined_test[:,index_overlapping_genes]

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Test')
test_overlapHVGs.write("adata_covid19_test_combined_overlap_4209.h5ad")

"""
Running CellO: (1) installing packages
"""

import os
import pandas as pd
import scanpy as sc
from anndata import AnnData
import cello
import umap

"""
Running CellO: (2) Load the expression matrix
"""

df = pd.DataFrame(test_overlapHVGs.X.todense() )  

df.columns = test_overlapHVGs.var_names
df.index = test_overlapHVGs.obs_names


# only keep over lapping genes between the train HVGs and test genes
df = df[overlapping_genes] 

print("Number of overlapping genes: ", len(overlapping_genes))
print("Number of cells with 0 expression: ", sum(df.sum(axis=1)==0))

# remove any cells that have sum of 0 expression across these genes
cells_to_keep = df.sum(axis=1)!=0
df = df[cells_to_keep] 

print("Dataframe shape: ", df.shape)

adata = AnnData(df)

weight = 10 
n_comps = 55 
n_neighbors = 20
mod = 3 

"""
Running CellO: (3) Normalize the data by estimating log transcripts per million (TPM)
"""

sc.pp.normalize_total(adata, target_sum=1e6)
upweight_geneList2 = ['CD4', 'CD8A', 'CD8B', 'ITGAM', 'ITGAX', 'ITGAE', 'CD14', 'FCGR3A']
adata[:,upweight_geneList2].X = adata[:,upweight_geneList2].X * weight
sc.pp.log1p(adata)

#Running CellO: (4) Perform principal components analysis (PCA)

sc.pp.pca(adata, n_comps=n_comps)

#Running CellO: (5) Compute nearest-neighbors graph, Cluster the cells with Leiden
sc.pp.neighbors(adata, n_neighbors=n_neighbors)
sc.tl.leiden(adata, resolution=2.0)

#Running CellO: (6) Specify CellO’s resource location
cello_resource_loc = "."

#Running CellO: (7)
### Future importing of the model, (do not have to retrain above)
model_prefix = "CellO_train_hyper" + str(mod)
os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Models/CellO1')
cello.scanpy_cello(
    adata, 
    'leiden',
    cello_resource_loc, 
    model_file=f'{model_prefix}.model.dill'
)

#Running CellO: (8) Run UMAP
sc.tl.umap(adata)

#Running CellO: (9) Create UMAP plot with cells colored by cluster
fig = sc.pl.umap(adata, color='leiden', return_fig=True)

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Output')

out_file1 = 'unnammed_clusters_cellO_test_basic_final.pdf' # <-- Name of the output file
fig.savefig(out_file1, bbox_inches='tight', format='pdf')
fig = sc.pl.umap(adata, color='Most specific cell type', return_fig=True)

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Output')

out_file2 = 'named_clusters_cellO_test_basic_final.pdf' # <-- Name of the output file
fig.savefig(out_file2, bbox_inches='tight', format='pdf')

#Running CellO: (10) Write CellO's output to a TSV file

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Output')
adata.write("basic1_Cello_prob_test_basic_final.h5ad")

#Evalute performance of CellO on training data:

# output from CellO for training data

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Output')
test_output = read_h5ad('basic1_Cello_prob_test_basic_final.h5ad')

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Test')
test_overlapHVGs = read_h5ad("adata_covid19_test_combined_overlap_4209.h5ad")

y_test = list(test_overlapHVGs.obs['new_cell_label']) 
y_pred = list(test_output.obs["Most specific cell type"])

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_pred, y_test)

print('Confusion Matrix\n')
pd.DataFrame(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


cellO_output_testData = pd.DataFrame(y_test)
cellO_output_testData['Predicted Cell Types'] = y_pred
cellO_output_testData.columns = ['Cell Type', 'Predicted Cell Type']

os.chdir('/Users/ameliaschroeder/Box/Penn Classes/CIS 519 Applied ML/CIS519_Project_SharedFolder/Data/Output')
cellO_output_testData.to_csv('CellO_basic_finalOuput_tunedHyper_testData.csv', index = True, header=True)

def get_index_positions_2(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list

y_test = list(test_overlapHVGs.obs['new_cell_label']) 
y_pred = list(test_output.obs["Most specific cell type"])

equal = [None] * len(y_pred)
for i in range(0,len(y_pred)):
   equal[i] = y_pred[i] == y_test[i]

test_incorrect = pd.DataFrame()
test_incorrect['Cell Type'] = y_test
test_incorrect['Predicted Cell Type'] = y_pred
test_incorrect['correct?'] = equal
index = get_index_positions_2(list(test_incorrect['correct?']), False)
test_incorrect = test_incorrect.iloc[index]

# Follicular helper T are a subset of Helper T cells who function very similarly to standard "helper T"
# altering accuracy prediction here to accept either TH or TFH as "correct"

TFH = test_incorrect[test_incorrect['Predicted Cell Type'] == 'T follicular helper cell']

TFH_wrong = TFH[TFH['Cell Type'] != 'helper T cell']

CD14CD16_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'CD14-positive, CD16-positive monocyte']

CD4MEM_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'CD4-positive, alpha-beta memory T cell']
CD4MEM_wrong = CD4MEM_wrong[CD4MEM_wrong['Cell Type'] != 'helper T cell']


# CellO only predicted one type of CD8+ T cell, and training dataset clustered all CD8s together

CD8_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'central memory CD8-positive, alpha-beta T cell']
CD8_wrong = CD8_wrong[CD8_wrong['Cell Type'] != 'central memory CD8-positive, alpha-beta T cell']

NK_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'natural killer cell']
NK_wrong = NK_wrong[NK_wrong['Cell Type'] != 'natural killer cell']

CD14_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'CD14-positive, CD16-negative classical monocyte']
CD14_wrong = CD14_wrong[CD14_wrong['Cell Type'] != 'CD14-positive, CD16-negative classical monocyte']

MemB = test_incorrect[test_incorrect['Predicted Cell Type'] == 'class switched memory B cell']
MemB_wrong1 = MemB[MemB['Cell Type'] != 'class switched memory B cell']
#MemB_wrong2 = MemB_wrong1[MemB_wrong1['Cell Type'] != 'naive B cell']
MemB_wrong = MemB_wrong1

NaiveB = test_incorrect[test_incorrect['Predicted Cell Type'] == 'naive B cell']
NaiveB_wrong = NaiveB[NaiveB['Cell Type'] != 'naive B cell']

# Considering anything that is an alpha-beta T cell as correctly classified
AB_T = test_incorrect[test_incorrect['Predicted Cell Type'] == 'alpha-beta T cell']
AB_T_1 = AB_T[AB_T['Cell Type'] != 'helper T cell']
AB_T_2 = AB_T_1[AB_T_1['Cell Type'] != 'central memory CD8-positive, alpha-beta T cell']
AB_T_3 = AB_T_2[AB_T_2['Cell Type'] != 'UNKNOWN CELL TYPE']
AB_T_wrong = AB_T_3

IE_T = test_incorrect[test_incorrect['Predicted Cell Type'] == 'innate effector T cell']
IE_T_wrong = IE_T[IE_T['Cell Type'] != 'UNKNOWN CELL TYPE']

NaiveT_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'naive T cell']

Treg = test_incorrect[test_incorrect['Predicted Cell Type'] == 'CD4-positive, CD25-positive, alpha-beta regulatory T cell']
Treg_wrong = Treg[Treg['Cell Type'] != 'helper T cell']

DC_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'myeloid dendritic cell, human']
DC_wrong = DC_wrong[DC_wrong['Cell Type'] != 'myeloid dendritic cell, human']

platelet_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'platelet']
platelet_wrong = platelet_wrong[platelet_wrong['Cell Type'] != 'platelet']

PB_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'plasmablast']
PB_wrong = PB_wrong[PB_wrong['Cell Type'] != 'plasmablast']

Mono_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'mononuclear cell']
CMP_wrong = test_incorrect[test_incorrect['Predicted Cell Type'] == 'common myeloid progenitor']


WRONG_TOTAL = TFH_wrong.append(CD14CD16_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(CD4MEM_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(CD8_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(NK_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(CD14_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(MemB_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(NaiveB_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(AB_T_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(IE_T_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(NaiveT_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(Treg_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(DC_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(platelet_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(PB_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(Mono_wrong)
WRONG_TOTAL = WRONG_TOTAL.append(CMP_wrong)

test_accuracy = 1 - (WRONG_TOTAL.shape[0] / len(y_pred))
