# --------------------------------------
# Customer Segmentation using RFM Analysis
# --------------------------------------

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------
# Load Dataset
# --------------------------------------

df = pd.read_csv("Online Retail.csv")

print("First 5 Rows of Dataset")
print(df.head())

print("\nDataset Information")
print(df.info())

# --------------------------------------
# Data Cleaning
# --------------------------------------

# Remove rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove negative or zero quantities
df = df[df['Quantity'] > 0]

# --------------------------------------
# Create Total Price Column
# --------------------------------------

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# --------------------------------------
# Create Reference Date
# --------------------------------------

reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# --------------------------------------
# Calculate RFM Values
# --------------------------------------

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("\nRFM Table")
print(rfm.head())

# --------------------------------------
# Assign RFM Scores
# --------------------------------------

rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

# Combine scores
rfm['RFM_Score'] = (
    rfm['R_score'].astype(str) +
    rfm['F_score'].astype(str) +
    rfm['M_score'].astype(str)
)

print("\nRFM Scores")
print(rfm.head())

# --------------------------------------
# Customer Segmentation
# --------------------------------------

def segment_customer(row):
    
    if row['R_score'] == 4 and row['F_score'] == 4:
        return "Loyal Customers"
    
    elif row['R_score'] >= 3 and row['F_score'] >= 3:
        return "Potential Loyalists"
    
    elif row['R_score'] == 4:
        return "New Customers"
    
    else:
        return "At Risk Customers"

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print("\nCustomer Segments")
print(rfm.head())

# --------------------------------------
# Visualization
# --------------------------------------

sns.set_style("whitegrid")

# Segment Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Segment', data=rfm)
plt.title("Customer Segments Distribution")
plt.xticks(rotation=30)
plt.show()

# --------------------------------------
# Heatmap for RFM values
# --------------------------------------

plt.figure(figsize=(6,5))
corr = rfm[['Recency','Frequency','Monetary']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("RFM Correlation Heatmap")
plt.show()

# --------------------------------------
# Final Observation
# --------------------------------------

print("\nRFM Analysis Completed Successfully")

print("\nInsights:")
print("1. Loyal customers purchase frequently and spend more money.")
print("2. New customers recently made purchases.")
print("3. At-risk customers have not purchased recently.")
print("4. Businesses can target each group with different marketing strategies.")