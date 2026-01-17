import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import warnings
import joblib # Library to save the model

# Suppress RDKit warnings and user warnings for cleaner output
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Step 1: Data Preparation using Predefined Interaction Types ---
print("Loading and preparing data...")
# Load the primary data files and the new type definition files
try:
    ddi_data = pd.read_csv("DDI_data.csv")
    smiles_data = pd.read_csv("drug_smiles.csv")
    ddi_types_map_df = pd.read_csv("DDI_types.csv")
    ddi_types_merged_df = pd.read_csv("DDI_types_merged.csv")
except FileNotFoundError as e:
    print(f"Error: Could not find a required data file: {e.filename}")
    exit()

# --- Clean and Prepare the Predefined Type Mappings ---
ddi_types_map_df.columns = [c.strip() for c in ddi_types_map_df.columns]
ddi_types_map_df["Origin DDI's type"] = ddi_types_map_df["Origin DDI's type"].str.strip().str.strip("',")

ddi_types_merged_df.columns = [c.strip() for c in ddi_types_merged_df.columns]
ddi_types_merged_df["type name"] = ddi_types_merged_df["type name"].str.strip()

int_to_type = pd.Series(
    ddi_types_merged_df['type name'].values,
    index=ddi_types_merged_df['merged DDI type index'] - 1
).to_dict()
num_classes = len(int_to_type)
target_names = [int_to_type[i] for i in range(num_classes)]

print(f"Loaded {num_classes} unique predefined interaction types.")

# --- Merge main data with predefined labels ---
ddi_data['interaction_type'] = ddi_data['interaction_type'].str.strip()
ddi_data = pd.merge(ddi_data, ddi_types_map_df, left_on='interaction_type', right_on="Origin DDI's type", how='left')
ddi_data.dropna(subset=['merged DDI type index'], inplace=True)
ddi_data['interaction_label'] = ddi_data['merged DDI type index'].astype(int) - 1

# --- Step 2: Feature Engineering ---
print("Creating drug name to SMILES mapping...")
drug1_info = ddi_data[['drug1_name', 'drug1_id']].rename(columns={'drug1_name': 'drug_name', 'drug1_id': 'drug_id'})
drug2_info = ddi_data[['drug2_name', 'drug2_id']].rename(columns={'drug2_name': 'drug_name', 'drug2_id': 'drug_id'})
all_drug_info = pd.concat([drug1_info, drug2_info]).drop_duplicates()
merged_smiles_info = pd.merge(all_drug_info, smiles_data, on='drug_id')
drug_smiles_mapping = pd.Series(merged_smiles_info.smiles.values, index=merged_smiles_info.drug_name).to_dict()

def smiles_to_fingerprint(smiles, n_bits=1024):
    """Converts a SMILES string to a Morgan fingerprint."""
    if pd.isna(smiles): return np.zeros(n_bits, dtype=int)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return np.zeros(n_bits, dtype=int)
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = mg.GetFingerprintAsNumPy(mol)
    return fp.astype(np.int8)

print("\nGenerating molecular fingerprints...")
ddi_data['drug1_smiles'] = ddi_data['drug1_name'].map(drug_smiles_mapping)
ddi_data['drug2_smiles'] = ddi_data['drug2_name'].map(drug_smiles_mapping)
ddi_data.dropna(subset=['drug1_smiles', 'drug2_smiles'], inplace=True)

if ddi_data.empty:
    print("\nError: Could not map SMILES for any drug pairs.")
    exit()

fp1 = np.array([smiles_to_fingerprint(s) for s in ddi_data['drug1_smiles']])
fp2 = np.array([smiles_to_fingerprint(s) for s in ddi_data['drug2_smiles']])

X = np.concatenate([fp1, fp2], axis=1)
y = ddi_data['interaction_label'].values

print(f"Successfully generated features for {X.shape[0]} drug pairs.")

# --- Step 3: Train-Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 4: Handle Class Imbalance with SMOTE ---
print("Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --- Step 5: Train XGBoost Multiclass Classifier ---
print("Training XGBoost multiclass model...")
model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False
)
model.fit(X_train_resampled, y_train_resampled)

# --- Step 6: Evaluate Model ---
print("Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# --- Step 7: Save the trained model and mappings to files ---
print("\nSaving trained model and necessary data...")
joblib.dump(model, 'drug_interaction_model.joblib')
joblib.dump(drug_smiles_mapping, 'drug_smiles_mapping.joblib')
joblib.dump(int_to_type, 'int_to_type.joblib')

print("Model and data saved successfully. You can now use predict_from_model.py for instant predictions.")
