import openml
import pandas as pd

# Fetch the dataset with ID 61 (Iris dataset)
[5, 12, 32]
for i in range(1, 60):
    print("-------------")
    print(i)
    dataset = openml.datasets.get_dataset(i, download_data=True, download_qualities=True, download_features_meta_data=True)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    print(y.dtype)
    print(X.shape)


# Convert to DataFrame
df = pd.DataFrame(X)
df['target'] = y

print(df.head())


