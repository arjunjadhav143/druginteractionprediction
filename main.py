import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import MolsToGridImage
import warnings
import joblib
from PIL import Image
import io

# Suppress RDKit warnings
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Dictionary for Interaction Type Explanations ---
EXPLANATION_DICT = {
    1: "**Meaning:** One or both drugs can affect the heart's electrical rhythm. Taking them together increases the risk of a dangerous type of irregular heartbeat called Torsades de Pointes.",
    2: "**Meaning:** One drug can change how the other is absorbed or processed, affecting its concentration in the blood. This could make one drug less effective or increase its risk of side effects.",
    3: "**Meaning:** This interaction impacts the nervous system. It could relate to neuromuscular blocking effects, which affect how nerves communicate with muscles.",
    4: "**Meaning:** These drugs have stimulatory effects that might be amplified or counteracted when taken together, potentially affecting the heart, blood pressure, or nervous system.",
    5: "**Meaning:** This interaction can affect blood sugar levels (hyperglycemic) or sodium levels (hyponatremic). This is especially important for patients with diabetes or kidney issues.",
    6: "**Meaning:** Taking these drugs together could increase the risk of developing an irregular heartbeat (arrhythmia), which can range from mild to life-threatening.",
    7: "**Meaning:** The combination may increase the chance, severity, or range of side effects (adverse events) for one or both drugs. This is a general category for negative outcomes.",
    8: "**Meaning:** One drug may affect how the body breaks down (metabolizes) the other. This can lead to the second drug being either less effective or building up to potentially harmful levels.",
    9: "**Meaning:** This interaction affects the airways. One drug might constrict the bronchi (bronchoconstriction) while the other dilates them (bronchilation), potentially interfering with asthma or COPD treatment.",
    10: "**Meaning:** The combination could increase the risk of toxicity, meaning one or both drugs could become poisonous and potentially damage organs like the kidneys, liver, or ears.",
    11: "**Meaning:** Taking these drugs together can enhance their sedative or depressant effects on the central nervous system, leading to increased drowsiness, dizziness, or impaired coordination.",
    12: "**Meaning:** One drug can affect how quickly the other is removed from the body (excreted), usually through the kidneys. This can alter the drug's concentration and effectiveness.",
    13: "**Meaning:** This interaction increases the risk of muscle problems (myopathy), which can range from muscle pain and weakness to a severe condition called rhabdomyolysis.",
    14: "**Meaning:** The combination could decrease the intended therapeutic effect of one or both drugs, making a treatment less effective than it should be.",
    15: "**Meaning:** These drugs can suppress the immune system. When taken together, this effect can be magnified, increasing the risk of infections.",
    16: "**Meaning:** This interaction relates to blood pressure. It could cause a significant drop (hypotension) or, in some cases, an increase (hypertension).",
    17: "**Meaning:** The combination can disrupt the balance of electrolytes in the body, such as potassium (hyperkalemia) or calcium (hypercalcemia), which is critical for nerve and muscle function.",
    18: "**Meaning:** The combination may increase the sedative or hypnotic (sleep-inducing) effects, leading to excessive drowsiness.",
    19: "**Meaning:** These drugs can affect blood clotting. Taking them together might increase the risk of bleeding or, conversely, the risk of forming a blood clot (thrombosis).",
    20: "**Meaning:** This is a miscellaneous category for other known interactions that don't fit into the more specific groups."
}

# --- Step 1: Load the Saved Model and Data Mappings (with Caching) ---
@st.cache_resource
def load_model_and_data():
    """Loads the saved model, data mappings, color, and distribution information."""
    try:
        model = joblib.load('drug_interaction_model.joblib')
        drug_smiles_mapping = joblib.load('drug_smiles_mapping.joblib')
        int_to_type = joblib.load('int_to_type.joblib')
        drug_names = sorted(list(drug_smiles_mapping.keys()))
        
        ddi_types_merged_df = pd.read_csv("DDI_types_merged.csv")
        
        int_to_color = pd.Series(
            ddi_types_merged_df['color'].str.strip('""').values,
            index=ddi_types_merged_df['merged DDI type index'] - 1
        ).to_dict()

        distribution_data = ddi_types_merged_df[['type name ', 'count']].copy()
        distribution_data.rename(columns={'type name ': 'Interaction Type', 'count': 'Count'}, inplace=True)
        
        total_count = distribution_data['Count'].sum()
        distribution_data['Frequency (%)'] = ((distribution_data['Count'] / total_count) * 100).round(2)
        
        distribution_data.set_index('Interaction Type', inplace=True)

        return model, drug_smiles_mapping, int_to_type, drug_names, int_to_color, distribution_data
    except FileNotFoundError:
        st.error("Error: Could not find the required model or data files.")
        st.error("Please make sure all '.joblib' files and 'DDI_types_merged.csv' are in the same directory.")
        return None, None, None, None, None, None

model, drug_smiles_mapping, int_to_type, drug_names, int_to_color, distribution_data = load_model_and_data()

# --- Step 2: Define Necessary Functions ---
def smiles_to_fingerprint(smiles, n_bits=1024):
    """Converts a SMILES string to a Morgan fingerprint."""
    if pd.isna(smiles): return np.zeros(n_bits, dtype=int)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return np.zeros(n_bits, dtype=int)
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = mg.GetFingerprintAsNumPy(mol)
    return fp.astype(np.int8)

def get_text_color(bg_color_str):
    """Determines if text should be black or white based on background brightness."""
    r, g, b = map(int, bg_color_str.split(','))
    if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
        return 'black'
    else:
        return 'white'

def predict_interaction_type(drug1_name, drug2_name):
    """Predicts the interaction and returns the result string, the prediction index, and an error flag."""
    drug1_smiles = drug_smiles_mapping.get(drug1_name)
    drug2_smiles = drug_smiles_mapping.get(drug2_name)

    if not drug1_smiles or not drug2_smiles:
        missing_drugs = []
        if not drug1_smiles: missing_drugs.append(f"'{drug1_name}'")
        if not drug2_smiles: missing_drugs.append(f"'{drug2_name}'")
        error_message = f"Prediction failed: Could not find SMILES for {', '.join(missing_drugs)} in the provided data."
        return error_message, None, True 

    fp1 = smiles_to_fingerprint(drug1_smiles)
    fp2 = smiles_to_fingerprint(drug2_smiles)
    feature_vector = np.concatenate([fp1, fp2]).reshape(1, -1)
    
    prediction_int = model.predict(feature_vector)[0]
    interaction_type = int_to_type[prediction_int]
    
    result = f"{drug1_name} and {drug2_name} is interacted and its type is {interaction_type}"
    return result, prediction_int, False

# --- Step 3: Create the Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("Drug-Drug Interaction Predictor")

st.metric(label="Overall Model Accuracy", value="79.42 %")
st.info("This accuracy represents the model's performance on the test set.")

st.write("---") 

st.header("Predict an Interaction")
st.write("Select two drug names to predict their interaction type.")

if drug_names:
    col1, col2 = st.columns(2)
    with col1:
        drug1 = st.selectbox("Select Drug 1 Name:", options=drug_names, index=None, placeholder="Choose a drug...")
        if drug1:
            try:
                drug1_smiles = drug_smiles_mapping.get(drug1)
                mol1 = Chem.MolFromSmiles(drug1_smiles)
                if mol1:
                    img = MolsToGridImage([mol1], legends=[drug1], subImgSize=(300, 300))
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    st.image(byte_im)
            except Exception:
                st.warning(f"Could not display structure for {drug1}.")

    with col2:
        drug2 = st.selectbox("Select Drug 2 Name:", options=drug_names, index=None, placeholder="Choose a drug...")
        if drug2:
            try:
                drug2_smiles = drug_smiles_mapping.get(drug2)
                mol2 = Chem.MolFromSmiles(drug2_smiles)
                if mol2:
                    img = MolsToGridImage([mol2], legends=[drug2], subImgSize=(300, 300))
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    st.image(byte_im)
            except Exception:
                st.warning(f"Could not display structure for {drug2}.")

    st.write("") 
    if st.button("Predict Interaction"):
        if model and all((drug_smiles_mapping, int_to_type, int_to_color, distribution_data is not None)):
            if drug1 and drug2:
                result, prediction_int, is_error = predict_interaction_type(drug1, drug2)
                if is_error:
                    st.error(result)
                else:
                    interaction_type_name = int_to_type[prediction_int]
                    stats = distribution_data.loc[interaction_type_name.strip()]
                    count = int(stats['Count'])
                    frequency = stats['Frequency (%)']

                    explanation_key = prediction_int + 1
                    explanation = EXPLANATION_DICT.get(explanation_key, "No explanation available.")
                    bg_color = int_to_color.get(prediction_int, "128, 128, 128")
                    text_color = get_text_color(bg_color)

                    st.markdown(
                        f"""
                        <div style="background-color:rgb({bg_color}); padding:15px; border-radius:10px; color:{text_color};">
                            <h4 style="color:{text_color};">Prediction Result</h4>
                            <p style="color:{text_color};"><strong>Interaction:</strong> {result}</p>
                            <p style="color:{text_color};">{explanation}</p>
                            <hr style="border-color:{text_color}; opacity:0.5;">
                            <p style="color:{text_color};"><strong>Dataset Info for this interaction type:</strong><br>
                            - <strong>Count:</strong> {count:,}<br>
                            - <strong>Frequency:</strong> {frequency}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.write("---")
                    
                    # --- Graphs Section ---
                    st.header("Visualizations")

                    # Model Comparison Section
                    st.subheader("Model Performance Comparison")
                    comparison_data = {
                        'Accuracy %': [79, 72]
                    }
                    comparison_df = pd.DataFrame(comparison_data, index=['XGBoost', 'Logistic Regression'])
                    st.bar_chart(comparison_df)

                    # Top 5 Most Frequent Interactions Section
                    if distribution_data is not None:
                        st.subheader("Top 5 Most Common Interaction Types in the Dataset")
                        top_5_data = distribution_data.sort_values(by='Frequency (%)', ascending=False).head(5)
                        st.bar_chart(top_5_data['Frequency (%)'])

            else:
                st.warning("Please select both drug names.")

