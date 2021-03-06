
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import altair as alt
import networkx as nx
import matplotlib  as plt
from anytree import Node, RenderTree


st.write("""Intelligent Automation""")

def main():    
    df_repo = pd.read_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/repo.csv')
    df_commit = pd.read_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/commit.csv') 
    data = pd.read_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/dataset.csv')  

    repo_name = st.selectbox("Select Repo", list(df_repo['repo']))
    df_commit1 = df_commit[df_commit['Repo']==repo_name]
    commit_name = st.selectbox("Select Commit", list(df_commit1['PR']))


    main_page(data, commit_name)

def main_page(data, commit_name):    
    file_name = st.selectbox("Select File", list(data['file_name']))
    get_cluster(file_name)

def get_cluster(file_name):
    result = pd.read_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/result.csv')      
    df = result[result['file_name']==file_name]
    st.write(df['label'])    
    network_graph(df)

def network_graph(df):
    result = pd.read_csv('C:/Harshad.Ambekar/personal/github/hackathon-intel-auto/dataset/labels.csv')
    G=nx.bull_graph()
    pos=nx.spring_layout(G) # positions for all nodes
    
    # nodes
    nx.draw_networkx_nodes(G,pos,
                        #nodelist=['loginflow','tenantflow','authenflow','authflow'],                        
                        nodelist=[0,1,2,3,4],
                        node_color='b',
                        node_size=250,
                        alpha=0.8)
    

    # edges
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_edges(G,pos,
                        edgelist=[(0,1),(0,2),(0,3),(1,2),(2,4)],
                        #edgelist=[('loginflow','tenantflow'),('loginflow','authenflow'),('loginflow','authflow')],
                        width=4,
                        alpha=0.5,
                        edge_color='r')
    # some math labels
    labels={}
    labels[0]=r'$loginflow$'
    labels[1]=r'$tenantflow$'
    labels[2]=r'$authenflow$'
    labels[3]=r'$authflow$' 
    labels[4]=r'$usercreation$' 
    nx.draw_networkx_labels(G,pos,labels,font_size=8)

    #plt.axis('off')
    #plt.savefig("labels_and_colors.png") # save as png
    #plt.figure(figsize=(4, 3), dpi=70)

    st.pyplot()
    
if __name__ == "__main__":
    main()