#import all the necessary libraries

import pandas as pd
import numpy as np
import pandas as pd
import os

# To Scale our data
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score

# To perform KMeans clustering 
from sklearn.cluster import KMeans

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree




def main():
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, 'C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/dataset.csv')
    filename = os.path.abspath(os.path.realpath(filename))
    dat = pd.read_csv(filename)    
    #datm=dat.drop(['full_path', 'file_name', 'dir_path', 'repo'],axis=1)
    fun_pca(dat)

def fun_pca(df):    
    datm=df.drop(['full_path', 'file_name', 'dir_path', 'repo'],axis=1)
    standard_scaler = StandardScaler()
    dat2 = standard_scaler.fit_transform(datm)
    pca = PCA(svd_solver='randomized', random_state=42)  
    pca.fit(dat2)
    colnames = list(datm.columns)
    pcs_df = pd.DataFrame({ 'Feature':colnames,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2]})    
    pca_final = IncrementalPCA(n_components=4)
    df_train_pca = pca_final.fit_transform(dat2)
    pc = np.transpose(df_train_pca)
    rownames = list(df['full_path'])
    pcs_df2 = pd.DataFrame({'full_path':rownames,'PC1':pc[0],'PC2':pc[1],'PC3':pc[2]})
    dat3 = pcs_df2
    dat3_1 = standard_scaler.fit_transform(dat3.drop(['full_path'],axis=1))
    
    sse_ = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k).fit(dat3_1)
        sse_.append([k, silhouette_score(dat3_1, kmeans.labels_)])

    ssd = []
    for num_clusters in list(range(1,10)):
        model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
        model_clus.fit(dat3_1)
        ssd.append(model_clus.inertia_)

    model_clus5 = KMeans(n_clusters = 5, max_iter=50)
    model_clus5.fit(dat3_1)

    dat4=dat3
    dat4.index = pd.RangeIndex(len(dat4.index))
    dat_km = pd.concat([dat4, pd.Series(model_clus5.labels_)], axis=1)
    dat_km.columns = ['full_path', 'PC1', 'PC2','PC3','ClusterID']

    dat5=pd.merge(df,dat_km,on='full_path')
    dat6=dat5[['loginflow','tenantflow','authenflow','authflow','usercreation','tenantcreation','tenantmodifcation','ClusterID']]

    clu_loginflow = pd.DataFrame(dat6.groupby(["ClusterID"]).loginflow.mean())
    clu_tenantflow = pd.DataFrame(dat6.groupby(["ClusterID"]).tenantflow.mean())
    clu_authenflow = pd.DataFrame(dat6.groupby(["ClusterID"]).authenflow.mean())
    clu_authflow = pd.DataFrame(dat6.groupby(["ClusterID"]).authflow.mean())
    clu_usercreation = pd.DataFrame(dat6.groupby(["ClusterID"]).usercreation.mean())
    clu_tenantcreation = pd.DataFrame(dat6.groupby(["ClusterID"]).tenantcreation.mean())         
    clu_tenantmodifcation = pd.DataFrame(dat6.groupby(["ClusterID"]).tenantmodifcation.mean())

    df = pd.concat([pd.Series([0,1,2,3]),clu_loginflow,clu_tenantflow,clu_authenflow,clu_authflow,clu_usercreation,clu_tenantcreation,clu_tenantmodifcation], axis=1)
    df.columns = ["ClusterID", "loginflow", "tenantflow", "authenflow","authflow","usercreation","tenantcreation","tenantmodifcation"]
    dat5.to_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/result.csv')

pca = PCA(svd_solver='randomized', random_state=42)
if __name__ == "__main__":
    main()