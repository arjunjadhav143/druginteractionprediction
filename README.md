#druginteractionprediction
🧪 Drug–Drug Interaction Prediction System
A machine learning system that predicts predefined drug–drug interaction (DDI) types using molecular structure data. The model leverages RDKit molecular fingerprints, SMOTE for class imbalance, and an XGBoost multiclass classifier to accurately classify interaction categories.

🚀 Overview
This project processes drug pairs using their SMILES representations, converts them into molecular fingerprints, and predicts the type of interaction between two drugs. It is designed for research, healthcare analytics, and ML-based bioinformatics projects.

✨ Features
🔬 Molecular fingerprint generation using RDKit
🧠 Multiclass classification with XGBoost
⚖️ Handles imbalanced datasets using SMOTE
🧾 Uses predefined & merged DDI interaction types
💾 Saves trained models and mappings for reuse
📊 Detailed evaluation with accuracy & classification report

🛠️ Tech Stack
Language: Python
Cheminformatics: RDKit
ML: XGBoost, Scikit-learn
Imbalance Handling: imbalanced-learn (SMOTE)
Data: Pandas, NumPy
Model Persistence: Joblib

📂 Project Structure
├── train_model.py
├── DDI_data.csv
├── drug_smiles.csv
├── DDI_types.csv
├── DDI_types_merged.csv
├── drug_interaction_model.joblib
├── drug_smiles_mapping.joblib
├── int_to_type.joblib
└── README.md

▶️ How to Run
pip install -r requirements.txt
python train_model.py
After training, the model and mappings are saved and can be reused for fast predictions.

📊 Model Pipeline
Load and clean DDI & SMILES datasets
Map drugs to molecular structures
Generate Morgan fingerprints
Balance classes using SMOTE
Train XGBoost multiclass model
Evaluate accuracy & save artifacts

📈 Output
Accuracy score
Per-class precision, recall & F1-score
Saved model for deployment or inference scripts

🎯 Use Cases
Drug safety analysis
Pharmacological research
Bioinformatics ML projects
Academic & final-year projects

📌 License
This project is licensed under the MIT License.

👤 Author
Arjun Jadhav
Machine Learning & Data Science Project
Drug Interaction Prediction System
