#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd


# In[18]:


# Load data


# In[19]:


df = pd.read_csv(r"C:\Users\rasha\OneDrive\Desktop\Job 2026\Phase 2\Online retail KPI analysis\Online Retail KPI analysis.csv", encoding="latin1")


# In[20]:


df.head() # First 5 rows of dataset


# In[21]:


df.isnull().sum() # Check number of nulls / blanks in features


# In[22]:


df = df.dropna() # Drop rows with blanks


# In[23]:


df.shape # Rows left after dropping blanks


# # Total_revenue

# In[24]:


df['revenue'] = df['Quantity']*df['UnitPrice']
total_revenue = df['revenue'].sum()
print(total_revenue)


# # High value customers

# In[25]:


df.head()


# In[26]:


df["CustomerID"] = df["CustomerID"].astype("Int64").astype(str)
df['CustomerID'] = df['CustomerID'].astype(str)


# In[27]:


user_stats = df.groupby('CustomerID')['revenue'].sum().reset_index()


# In[28]:


top_10_customers  = user_stats.sort_values(by = 'revenue', ascending  = False).head(10)
top_10_customers


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


sns.barplot(x= 'CustomerID', data = top_10_customers , y = 'revenue')


# # Identifying customers who spend on more expensive items

# In[31]:


df['total_revenue'] = df['UnitPrice'] * df['Quantity']
customer_spend = df.groupby('CustomerID')['total_revenue'].sum().reset_index()

customer_avg_price = df.groupby('CustomerID')['UnitPrice'].mean().reset_index()
customer_avg_price.rename(columns={'UnitPrice': 'AvgPrice'}, inplace=True)

customer_summary = pd.merge(customer_spend, customer_avg_price, on='CustomerID')
print(customer_summary)


# In[34]:


high_value_customers = customer_summary[customer_summary['AvgPrice'] > 200].sort_values(by = 'AvgPrice', ascending = False)
print(high_value_customers)


# In[37]:


unique_invoice = df['InvoiceNo'].nunique()
unique_invoice


# # Average order value

# In[38]:


average_order_value = df['revenue'].sum()/ unique_invoice
average_order_value


# # Most sold item

# In[39]:


df.groupby('StockCode')['Quantity'].sum().sort_values(ascending = False).head(10)


# # Which items are sold together commonly

# In[244]:


basket = df.groupby("InvoiceNo")["StockCode"].apply(list)
print(basket)


# In[245]:


from itertools import combinations
from collections import Counter

pair_counter = Counter()

for items in basket:
    # unique pairs from invoice
    pairs = combinations(sorted(set(items)), 2)
    pair_counter.update(pairs)

# most common pairs
print(pair_counter.most_common(5))


# In[246]:


pair_list = pair_counter.most_common(5)   # or .most_common() for all

# Make DataFrame
df_pairs = pd.DataFrame(pair_list, columns=["Item_Pair", "Count"])

# If you want ItemA, ItemB as separate columns:
df_pairs[["ItemA", "ItemB"]] = pd.DataFrame(df_pairs["Item_Pair"].tolist(), index=df_pairs.index)
df_pairs = df_pairs[["ItemA", "ItemB", "Count"]]

# Save to CSV
df_pairs.to_csv("item_pairs.csv", index=False)
print("Saved to item_pairs.csv")


# In[247]:


# Itempairs = pd.read_csv(r"C:\Users\rasha\OneDrive\Desktop\Job 2026\Phase 2\Online retail KPI analysis\item combinations.csv", encoding="latin1")


# # Which items are more popular in festive seasons

# In[203]:


import datetime


# In[204]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['month'] = df['InvoiceDate'].dt.month


# In[205]:


df.head()


# In[206]:


Dec_jan_feb = df[(df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)]


# In[241]:


Dec_jan_feb.groupby('StockCode')['Quantity'].sum().sort_values(ascending = False)


# In[208]:


# 22197 - SMALL POPCORN HOLDER', is most sold itmes in festive months too, 
# closely followed by 84077 - 'WORLD WAR 2 GLIDERS ASSTD DESIGNS'


# # Which item is most popular in different Countries

# In[242]:


# total quantity per item per country
item_sales = df.groupby(["Country", "StockCode"])["Quantity"].sum().reset_index()

# find the item with max quantity for each country
most_popular = item_sales.loc[item_sales.groupby("Country")["Quantity"].idxmax()]
print(most_popular)


# # Return Rate by items

# In[228]:


return_rate = df.loc[df['Quantity']<0, 'Quantity'].abs().sum() / df['Quantity'].abs().sum()* 100
print('% of items returned out of total items sold ', return_rate)


# # Conversion rate

# In[ ]:


# Number of invoices with net positive quantity / total invoices


# In[229]:


Total_invoices = df['InvoiceNo'].unique()


# In[232]:


Total_invoice_no = len(Total_invoices)


# In[236]:


Total_invoice_no


# In[237]:


Positive_invoice = (df.groupby('InvoiceNo')['Quantity'].sum() > 0).sum()


# In[238]:


Positive_invoice


# In[239]:


Conversion_rate =  (Positive_invoice/Total_invoice_no)* 100


# # CHURN ANALYSIS

# In[2]:


import pandas as pd

churnrate = pd.read_csv(r"C:\Users\rasha\OneDrive\Desktop\Job 2026\Phase 2\Online retail KPI analysis\Forchurnrate.csv",encoding="latin1")

# Count active users per cohort per month
cohort_counts = churnrate.groupby(['Cohort_Index','Cohort_Month'])['CustomerID'].nunique().reset_index()
cohort_counts.rename(columns={'CustomerID':'ActiveUsers'}, inplace=True)

print(cohort_counts)


# In[12]:


churnrate['InvoiceDate'] = pd.to_datetime(churnrate['InvoiceDate'])

# Compute Cohort_Index (first month each customer appeared)
churnrate['Cohort_Index'] = churnrate.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')

churnrate['Cohort_Index'] = churnrate['Cohort_Index'].dt.to_timestamp()

# Compute Cohort_Month = months since cohort start
churnrate['Cohort_Month'] = (churnrate['InvoiceDate'].dt.year - churnrate['Cohort_Index'].dt.year) * 12 + \
                             (churnrate['InvoiceDate'].dt.month - churnrate['Cohort_Index'].dt.month)

# Count distinct active users per cohort per month
cohort_counts = churnrate.groupby(['Cohort_Index','Cohort_Month'])['CustomerID'].nunique().reset_index()
cohort_counts.rename(columns={'CustomerID':'ActiveUsers'}, inplace=True)

# Pivot to create retention matrix
retention = cohort_counts.pivot(index='Cohort_Index', columns='Cohort_Month', values='ActiveUsers')

# Cohort size (Month 0 users)
cohort_size = retention[0]

# Calculate Retention %
retention_pct = retention.divide(cohort_size, axis=0) * 100

# Calculate Cumulative Churn %
cumulative_churn = 100 - retention_pct

# Calculate Periodic Churn %
periodic_churn = retention_pct.shift(axis=1, fill_value=100) - retention_pct

# Display results
print("Retention %:\n", retention_pct)
print("\nCumulative Churn %:\n", cumulative_churn)
print("\nPeriodic Churn %:\n", periodic_churn)


# In[15]:


# retention_pct is your retention % matrix with NaNs
overall_churns = []

for cohort in retention_pct.index:
    # Take the last available month (not NaN) for this cohort
    last_retention = retention_pct.loc[cohort].dropna().iloc[-1]
    overall_churns.append(100 - last_retention)

# Weighted or simple average overall churn
overall_churn = sum(overall_churns) / len(overall_churns)
print(f"Overall churn % (corrected for NaNs): {overall_churn:.2f}%")

