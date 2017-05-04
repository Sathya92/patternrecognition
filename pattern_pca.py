import sys
import numpy as np
import scipy.stats as ss
import pandas as pd

import random
from sklearn import cluster as Kcluster, metrics as SK_Metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix




def find_pca(data_frame):
    pca = PCA(n_components=2)
    return pd.DataFrame(pca.fit_transform(data_frame),columns=['col1','col2'])

def getEigenVals(inputDF):
    X = inputDF.ix[:,0:10].values
    y = inputDF.ix[:,10].values
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    np.savetxt("eig_val.csv", eig_vals, delimiter=",")

def get_squared_loadings(dataframe, intrinsic):
    std_input = StandardScaler().fit_transform(dataframe)
    pca = PCA(n_components=intrinsic)
    pca.fit_transform(std_input)

    loadings = pca.components_
    squared_loadings = []
    a = np.array(loadings)
    a = a.transpose()
    for i in range(len(a)):
        squared_loadings.append(np.sum(np.square(a[i])))

    df_attributes = pd.DataFrame(pd.DataFrame(dataframe).columns)
    df_attributes.columns = ["attributes"]
    df_sqL = pd.DataFrame(squared_loadings)
    df_sqL.columns = ["squared_loadings"]
    sample = df_attributes.join([df_sqL])
    sample = sample.sort_values(["squared_loadings"], ascending=[False])
    sample.to_csv("squared_loadings.csv",sep=',')
    top3_loadings = sample.head(n=3)
    return top3_loadings['attributes'].values.tolist()


def main():
    global df 
    df = pd.read_csv("randomData.csv")
    K_Optimum = 4
    intrinsic = 4
    pca_data = find_pca(df)
    df1 = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])
    scatter_matrix(df1,alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()
    getEigenVals(df)
    squared_loadings = get_squared_loadings(df,intrinsic)

    df.ix[:, squared_loadings].to_csv("scatterplot_matrix.csv", sep=',')
main()