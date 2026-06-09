# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize Hugging Face API client using the token stored as an environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the path to the dataset registered on the Hugging Face dataset space
DATASET_PATH = "hf://datasets/sheetalkarande/tourism-package-model/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Initial shape:", df.shape)

# ----------------------- Data Cleaning -----------------------

# Drop unnecessary columns:
#   'Unnamed: 0' -> leftover pandas index column, not useful for modeling
#   'CustomerID' -> unique identifier, not useful for modeling
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

# Fix inconsistent category labels found in the raw data
# 'Fe Male' is a typo for 'Female' in the Gender column
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
# 'Unmarried' is treated the same as 'Single' in MaritalStatus
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# Handle missing values:
#   - numeric columns are filled with the median
#   - categorical columns are filled with the mode (most frequent value)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ----------------------- Encoding -----------------------

# Encode all categorical (object) columns into numerical representations
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
print("Encoded categorical columns:", categorical_cols)

# Define the target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------- Train-Test Split -----------------------

# Perform a stratified train-test split to preserve the class balance of the target
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", Xtrain.shape, "| Test shape:", Xtest.shape)

# Save the resulting splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# ----------------------- Upload to Hugging Face -----------------------

# Upload the train and test datasets back to the Hugging Face dataset space
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sheetalkarande/tourism-package-model",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face dataset space.")
