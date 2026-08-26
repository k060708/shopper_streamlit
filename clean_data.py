import pandas as pd

# reading the file online_retail.csv
df = pd.read_csv("online_retail.csv")

# analysing the dataset 

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Checking if duplicate rows exist 

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# Identifying unusual records 

#rows with missing customer ids
print("\nRows with missing CustomerID:")
print(df[df["CustomerID"].isnull()])

#rows with cancelled invoices
print("\nCancelled invoices:")
print(df[df["InvoiceNo"].astype(str).str.startswith("C")])

#rows with negative or zero quantity
print("\nRows with negative or zero Quantity:")
print(df[df["Quantity"] <= 0])

#rows with negative or zero unitprice 
print("\nRows with negative or zero UnitPrice:")
print(df[df["UnitPrice"] <= 0])

#finding shape of the dataset
print("\nOriginal dataset shape:")
print(df.shape)

# Remove rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove rows with cancelled invoices where InvoiceNo starts with C
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove rows with negative or zero quantities
df = df[df["Quantity"] > 0]

# Remove rows negative or zero prices
df = df[df["UnitPrice"] > 0]

# Remove duplicate rows
df = df.drop_duplicates()

print("\nCleaned dataset shape:")
print(df.shape)

# Checking the dataset after cleaning to ensure nothing is missed 

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nCancelled invoices after cleaning:")
print(df[df["InvoiceNo"].astype(str).str.startswith("C")])

print("\nInvalid Quantity after cleaning:")
print(df[df["Quantity"] <= 0])

print("\nInvalid UnitPrice after cleaning:")
print(df[df["UnitPrice"] <= 0])

# save the cleaned dataset 

df.to_csv("cleaned_online_retail.csv", index=False)
print("\nCleaned dataset saved as cleaned_online_retail.csv")