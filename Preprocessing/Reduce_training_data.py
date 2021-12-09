import os
os.chdir("/home/amesch/CITER/Data/")

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

from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
from torch.cuda import is_available

from sciPEN_ME.sciPEN_API import sciPEN_API


adata_covid19_haniffa = read_h5ad('/home/amesch/CITER/Data/Covid_Combined_SCE_raw.h5ad')


status = adata_covid19_haniffa.obs['Status']

def condition(x): return x == "Covid" 
index_keep_covid = [idx for idx, element in enumerate(status) if condition(element)]
def condition(x): return x == "Healthy" 
index_keep_healthy = [idx for idx, element in enumerate(status) if condition(element)]

index_keep = index_keep_covid + index_keep_healthy


adata_covid19_haniffa_red = adata_covid19_haniffa[index_keep]
adata_covid19_haniffa = None

adata_covid19_haniffa_red_log = adata_covid19_haniffa_red.copy()
sc.pp.log1p(adata_covid19_haniffa_red_log)
sc.pp.highly_variable_genes(adata_covid19_haniffa_red_log, subset = False, batch_key = 'Site', n_top_genes = 1000)

hvgs = adata_covid19_haniffa_red_log.var.index[adata_covid19_haniffa_red_log.var['highly_variable']].copy()
gene_set = list(set(hvgs))
adata_covid19_haniffa_red_final = adata_covid19_haniffa_red[:, gene_set].copy()

os.chdir("/home/amesch/CITER/CIS519_Project/Data")
adata_covid19_haniffa_red_final.write("adata_covid19_haniffa_red_final.h5ad")
