#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta


# In[18]:


# Membaca data
datacustomer = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Customer.csv", delimiter=';')
datastore = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Store.csv", delimiter=';')
dataproduct = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Product.csv", delimiter=';')
datatransaction = pd.read_csv("/Users/dwiyulianto/Downloads/Case Study - Transaction.csv", delimiter=';')


# In[19]:


# Data Customer Section
# Handle duplicates if necessary
datacustomer_cleaned = datacustomer.drop_duplicates()
datacustomer_cleaned = datacustomer.dropna(axis=0)

# Validate the results
print(datacustomer_cleaned.head())
print(datacustomer_cleaned.shape)

datacustomer_cleaned.isnull().sum()


# In[20]:


# Data Product Section
# Handle duplicates if necessary
dataproduct_cleaned = dataproduct.drop_duplicates()

# Validate the results
print(dataproduct_cleaned.head())
print(dataproduct_cleaned.shape)

dataproduct_cleaned.isnull().sum()


# In[21]:


# Data Store Section
# Identify and handle missing values
datastore_cleaned = datastore.dropna()

# Handle duplicates if necessary
datastore_cleaned = datastore_cleaned.drop_duplicates()

# Validate the results
print(datastore_cleaned.head())
print(datastore_cleaned.shape)

datastore_cleaned.isnull().sum()


# In[22]:


# Data Transaction Section

# Identify and handle missing values
datatransaction_cleaned = datatransaction.dropna()

# Handle duplicates if necessary
datatransaction_cleaned = datatransaction_cleaned.drop_duplicates()

#Mengubah tipe data supaya sesuai
datatransaction_cleaned['Date'] = pd.to_datetime(datatransaction_cleaned['Date'])

# Validate the results
print(datatransaction_cleaned.head())
print(datatransaction_cleaned.shape)

datatransaction_cleaned.isnull().sum()


# In[23]:


#Mengubah tipe data yang belum sesuai
datatransaction_cleaned['Date'] = pd.to_datetime(datatransaction_cleaned['Date'])


# In[8]:


datacustomer_cleaned.info()
datatransaction_cleaned.info()
datastore_cleaned.info()
dataproduct_cleaned.info()


# In[24]:


#Menggabungkan data pertama
merged_data = pd.merge(datatransaction_cleaned, datacustomer_cleaned, on='CustomerID')


# In[25]:


merged_data_final = pd.merge(merged_data, dataproduct_cleaned, on='ProductID')


# In[26]:


merged_data_final


# In[27]:


transaksiharian = merged_data_final.groupby('Date')['Qty'].sum().reset_index()


# In[28]:


transaksiharian


# In[29]:


# Build and train the ARIMA model
model = ARIMA(transaksiharian['Qty'], order=(2, 1, 2))
model_fit = model.fit()


# In[30]:


# Generate predictions
predictions = model_fit.predict(start=transaksiharian.index[0], end=transaksiharian.index[-1])


# In[31]:


# Plot the actual vs predicted values
plt.plot(transaksiharian['Date'], transaksiharian['Qty'], label='Actual')
plt.plot(transaksiharian['Date'], predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.show()


# In[32]:


predictions


# In[ ]:




