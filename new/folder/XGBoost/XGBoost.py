import zipfile
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import scanpy
import numpy as np
import os
import pandas as pd
import gc
from anndata import read_h5ad


with zipfile.ZipFile("adata_covid19_haniffa_red_final_overlap_4209_RELABELED.h5ad.zip", 'r') as zip_ref:
    zip_ref.extractall("train")
    
df = read_h5ad("train/adata_covid19_haniffa_red_final_overlap_4209_RELABELED.h5ad")
scanpy.pp.normalize_total(df)
scanpy.pp.log1p(df)
scanpy.pp.pca(df, n_comps=200)
X_train = df.obsm["X_pca"]
y_train = df.obs["new_cell_label"]

model = XGBClassifier()
model.fit(X_train, y_train)

with zipfile.ZipFile("adata_covid19_test_combined_overlap_4209.h5ad.zip", 'r') as zip_ref:
    zip_ref.extractall("test")
    
df = read_h5ad("test/adata_covid19_test_combined_overlap_4209.h5ad")
scanpy.pp.normalize_total(df)
scanpy.pp.log1p(df)
scanpy.pp.pca(df, n_comps=200)
X_test = df.obsm["X_pca"]
y_test = df.obs["new_cell_label"]


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
