#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[23]:


# Membaca data
datacustomer = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Customer.csv", delimiter=';')
datastore = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Store.csv", delimiter=';')
dataproduct = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Product.csv", delimiter=';')
datatransaction = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Transaction.csv", delimiter=';')


# In[24]:


# Data Cleansing

# Menghapus duplikat dan mengubah tipe data
# datacustomer
datacustomer = datacustomer.drop_duplicates()
datacustomer = datacustomer.dropna(axis=0)

#datastore
datastore = datastore.drop_duplicates()
datastore = datastore.dropna(axis=0)


#dataproduct
dataproduct = dataproduct.drop_duplicates()
dataproduct = dataproduct.dropna(axis=0)

#datatransaction
datatransation = datatransaction.drop_duplicates()
datatransaction = datatransaction.dropna(axis=0)

#Mengubah tipe data yang belum sesuai
datatransaction['Date'] = pd.to_datetime(datatransaction['Date'])


# In[25]:


# Cek data apakah ada miss data
dataproduct.isnull().sum()


# In[26]:


datastore.isnull().sum()


# In[27]:


datatransation.isnull().sum()


# In[28]:


datacustomer.isnull().sum()


# In[31]:


#Menggabungkan data pertama
merged_data = pd.merge(datatransaction, datacustomer, on='CustomerID')


# In[32]:


dataclustering = pd.merge(merged_data, dataproduct, on='ProductID')


# In[33]:


dataclustering


# In[34]:


#Melakukan data cleansing dengan menyesuaikan tipe data dengan benar
dataclustering['CustomerID'] = dataclustering['CustomerID'].astype(str)
dataclustering['TransactionID'] = dataclustering['TransactionID'].astype(str)
dataclustering['Qty'] = dataclustering['Qty'].astype(float)
dataclustering['TotalAmount'] = dataclustering['TotalAmount'].astype(float)


# In[35]:


# Membuat Data Baru untuk Clustering
# Mengelompokkan berdasarkan 'customerID' dan mengagregasi 'transactionID' dengan fungsi count(), 'qty' dengan fungsi sum(), dan 'total_amount' dengan fungsi sum()
df_cluster = dataclustering.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()


# In[13]:


# Memilih Variabel untuk Clustering
# Pilih variabel yang akan digunakan untuk clustering
X = df_cluster[['TransactionID', 'Qty', 'TotalAmount']]


# In[14]:


# Normalisasi Data
# Jika variabel memiliki skala yang berbeda, normalisasikan data untuk memastikan variabel memiliki pengaruh yang seimbang dalam proses clustering
X_normalized = (X - X.mean()) / X.std()


# In[15]:


# Menggunakan K-means Clustering
# Menentukan jumlah cluster yang diinginkan
n_clusters = 4  # Ubah sesuai dengan kebutuhan Anda
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_normalized)


# In[36]:


# Menambahkan Label Cluster ke DataFrame
df_cluster['cluster_label'] = kmeans.labels_


# In[37]:


# Menampilkan Hasil Clustering
print(df_cluster[['CustomerID', 'cluster_label']])


# In[18]:


# Evaluasi Hasil Clustering
# Melakukan evaluasi hasil clustering menggunakan metrik evaluasi clustering seperti SSE (Sum of Squared Errors)
sse = kmeans.inertia_
print("SSE (Sum of Squared Errors):", sse)


# In[38]:


df_cluster


# In[39]:


# Visualisasi Hasil Clustering
plt.scatter(X_normalized['Qty'], X_normalized['TotalAmount'], c=df_cluster['cluster_label'], cmap='viridis')
plt.xlabel('Qty')
plt.ylabel('Total Amount')
plt.title('Clustering Result')
plt.colorbar(label='Cluster')
plt.show()


# In[40]:


# Menghitung jumlah pelanggan per klaster
cluster_counts = df_cluster['cluster_label'].value_counts().reset_index()
cluster_counts.columns = ['cluster_label', 'customer_count']

# Menampilkan hasil jumlah pelanggan per klaster
print(cluster_counts)


# In[ ]:




