import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import qrcode
from PIL import Image
import io
import base64

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('Model_Save\modelBoosting.joblib')

# App title and description
st.set_page_config(
    page_title="Protein Gel Hardness Predictor",
    page_icon="🔬",
    layout="wide"
)

st.title("Prédicteur de dureté des gels protéiques")
st.markdown("""
Cette application prédit la dureté (solidité/fermeté) des gels protéiques en fonction de différentes propriétés.
Ajustez simplement les paramètres et cliquez sur 'Prédire' pour obtenir la prédiction.
""")

# Create a sidebar for inputs
st.sidebar.header("Paramètres d'entrée")

# Définition des colonnes pour les caractéristiques
feature_info = {
    "Code protéine": {"type": "numeric", "min": 10000.0, "max": 90000.0, "default": 30000.0, "step": 1000.0, "help": "Code identifiant le type de protéine"},
    "Concentration en protéine (%)": {"type": "numeric", "min": 1.0, "max": 20.0, "default": 10.0, "step": 0.5, "help": "Concentration de la protéine en pourcentage"},
    "Code traitement": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 30000.0, "step": 1000.0, "help": "Code identifiant le type de traitement"},
    "Code condition traitement": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 30000.0, "step": 1000.0, "help": "Code pour la condition de traitement"},
    "Valeur condition traitement": {"type": "numeric", "min": 0.0, "max": 500.0, "default": 100.0, "step": 10.0, "help": "Valeur pour la condition de traitement"},
    "Température traitement (°C)": {"type": "numeric", "min": 0.0, "max": 120.0, "default": 80.0, "step": 5.0, "help": "Température du traitement en °C"},
    "Temps traitement (min)": {"type": "numeric", "min": 0.0, "max": 500.0, "default": 30.0, "step": 5.0, "help": "Durée du traitement en minutes"},
    "Additifs": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 0.0, "step": 1000.0, "help": "Code identifiant les additifs utilisés"},
    "Concentration additifs (%)": {"type": "numeric", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Concentration des additifs en pourcentage"},
    "pH": {"type": "numeric", "min": 1.0, "max": 14.0, "default": 7.0, "step": 0.1, "help": "Valeur du pH de la solution"},
    "Type de sel": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 60000.0, "step": 1000.0, "help": "Code identifiant le type de sel utilisé"},
    "Force ionique (M)": {"type": "numeric", "min": 0.0, "max": 1.0, "default": 0.1, "step": 0.01, "help": "Force ionique en Molarité"},
    "Température de chauffage (°C) pour préparation du gel": {"type": "numeric", "min": 50.0, "max": 120.0, "default": 90.0, "step": 5.0, "help": "Température de chauffage pour la préparation du gel en °C"},
    "Temps de chauffage/maintien (min)": {"type": "numeric", "min": 0.0, "max": 120.0, "default": 30.0, "step": 5.0, "help": "Durée de chauffage/maintien en minutes"},
    "Température de stockage des échantillons (°C)": {"type": "numeric", "min": 0.0, "max": 30.0, "default": 4.0, "step": 1.0, "help": "Température de stockage en °C"},
    "Temps de stockage (h)": {"type": "numeric", "min": 0.0, "max": 72.0, "default": 12.0, "step": 1.0, "help": "Durée de stockage en heures"},
    "Si un gel peut être formé (0-1)": {"type": "numeric", "min": 0.0, "max": 1.0, "default": 1.0, "step": 1.0, "help": "Valeur binaire indiquant si un gel peut être formé (1) ou non (0)"}
}

# Create columns for cleaner layout in the sidebar
def create_feature_inputs():
    inputs = {}
    
    for feature, config in feature_info.items():
        if config["type"] == "numeric":
            inputs[feature] = st.sidebar.slider(
                feature, 
                min_value=float(config["min"]), 
                max_value=float(config["max"]), 
                value=float(config["default"]), 
                step=float(config["step"]),
                help=config["help"]
            )
    
    return inputs

# File upload function
uploaded_file = st.sidebar.file_uploader("Télécharger un fichier CSV", type=["csv"])

# Handle file upload
if uploaded_file is not None:
    # Lire le fichier CSV
    data = pd.read_csv(uploaded_file)
    st.write("Données téléchargées :")
    st.dataframe(data.head())
    
    # Utiliser ces données pour faire des prédictions
    model = load_model()
    
    predictions = model.predict(data)
    
    st.header("Prédictions pour les données téléchargées")
    data['Prédictions de dureté (g)'] = predictions
    st.dataframe(data)

# Create a prediction button
if st.sidebar.button("Prédire la dureté du gel"):
    # Convert inputs to DataFrame for prediction
    input_df = pd.DataFrame([user_inputs])
    
    # Load the model
    model = load_model()
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display the prediction in main area
    st.header("Résultats de la prédiction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Dureté prédite du gel (g)", f"{prediction:.2f} g")
    
    with col2:
        # Create a gauge or progress bar for visualization
        fig, ax = plt.subplots(figsize=(3, 1))
        cmap = plt.cm.RdYlGn
        
        # Normalize the prediction for visualization (assuming max around 2000g based on the data)
        max_value = 2000
        normalized_value = min(prediction / max_value, 1.0)  
        
        ax.barh(0, normalized_value, color=cmap(normalized_value), height=0.3)
        ax.barh(0, 1, color='lightgrey', height=0.3, alpha=0.3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '500', '1000', '1500', '2000'])
        ax.set_xlabel('Dureté (g)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        st.pyplot(fig)
    
    # Feature importance visualization
    st.subheader("Importance des caractéristiques")
    
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Caractéristique': input_df.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Caractéristique', data=feature_importance_df, palette='viridis')
    ax.set_title('Importance des caractéristiques pour la prédiction')
    ax.set_xlabel('Score d\'importance')
    ax.set_ylabel('Caractéristique')
    st.pyplot(fig)
    
    # Show input parameters used
    st.subheader("Paramètres d'entrée utilisés")
    st.dataframe(input_df)
