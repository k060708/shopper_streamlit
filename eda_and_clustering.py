import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# setting plot font and size 
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


print("Loading dataset...")
# load the clean dataset 
df = pd.read_csv("cleaned_online_retail.csv")


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


# Transaction Volume by Country
print("\n--- 1. Top Countries by Transaction Volume ---")
country_sales = df.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=country_sales.values, y=country_sales.index, palette="viridis")
plt.title("Top 10 Countries by Number of Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# Top-Selling Products

print("\n--- 2. Top-Selling Products ---")
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette="mako")
plt.title("Top 10 Selling Products by Quantity Sold")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Product Description")
plt.tight_layout()
plt.show()

# Purchase Trends Over Time

print("\n--- 3. Purchase Trends Over Time ---")
# Set index to InvoiceDate and resample monthly
df_time = df.set_index('InvoiceDate')
monthly_sales = df_time['TotalPrice'].resample('ME').sum()

plt.figure(figsize=(12, 5))
monthly_sales.plot(marker='o', color='b', linewidth=2)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue ($)")
plt.tight_layout()
plt.show()

# Monetary Distribution per Transaction & Customer

print("\n--- 4. Monetary Distribution ---")
transaction_spend = df.groupby('InvoiceNo')['TotalPrice'].sum()
customer_spend = df.groupby('CustomerID')['TotalPrice'].sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot transaction spend (log scale because data is skewed)
sns.histplot(transaction_spend[transaction_spend < 2000], bins=50, ax=axes[0], color="purple")
axes[0].set_title("Distribution of Spend per Transaction (< $2000)")
axes[0].set_xlabel("Transaction Spend ($)")

# Plot customer spend
sns.histplot(customer_spend[customer_spend < 5000], bins=50, ax=axes[1], color="teal")
axes[1].set_title("Distribution of Spend per Customer (< $5000)")
axes[1].set_xlabel("Total Spend ($)")

plt.tight_layout()
plt.show()

# 8. Product Recommendation Heatmap / Similarity Matrix
print("\n--- 8. Generating Item-Item Similarity Matrix ---")
# To prevent computer freeze, take the top 20 most popular products
top_20_items = df['Description'].value_counts().head(20).index
subset_df = df[df['Description'].isin(top_20_items)]

# Create Customer-Item Matrix
item_matrix = subset_df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)
# Convert to binary (Bought = 1, Not Bought = 0)
item_matrix_binary = (item_matrix > 0).astype(int)

# Compute Cosine Similarity between Items
item_similarity = cosine_similarity(item_matrix_binary.T)
item_sim_df = pd.DataFrame(item_similarity, index=item_matrix.columns, columns=item_matrix.columns)

# Plot Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(item_sim_df, annot=False, cmap="YlGnBu", xticklabels=True, yticklabels=True)
plt.title("Product Recommendation Similarity Heatmap (Top 20 Items)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nEDA and Analysis Completed Successfully!")