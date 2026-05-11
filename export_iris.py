import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column (the species)
df['species'] = [iris.target_names[i] for i in iris.target]

# Save to CSV
filename = "iris_dataset.csv"
df.to_csv(filename, index=False)

print(f"Successfully exported Iris dataset to {filename}")
print("\nFirst 5 rows:")
print(df.head())
