import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

st.title("Protein Gel Hardness Predictor")
st.markdown("""
This application predicts the hardness/firmness/strength of protein gels based on various properties.
Simply adjust the parameters and click 'Predict' to get the prediction.
""")

# Create a sidebar for inputs
st.sidebar.header("Input Parameters")

# Define columns based on the model
feature_info = {
    "Protein codes": {"type": "numeric", "min": 10000.0, "max": 90000.0, "default": 30000.0, "step": 1000.0, "help": "Code identifying the protein type"},
    "Protein Concentration (%)": {"type": "numeric", "min": 1.0, "max": 20.0, "default": 10.0, "step": 0.5, "help": "Concentration of protein in percentage"},
    "Treatment code": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 30000.0, "step": 1000.0, "help": "Code identifying the treatment type"},
    "Treatment condition code": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 30000.0, "step": 1000.0, "help": "Code for treatment condition"},
    "Treatment condition value": {"type": "numeric", "min": 0.0, "max": 500.0, "default": 100.0, "step": 10.0, "help": "Value for treatment condition"},
    "Treatment temperature ( °C)": {"type": "numeric", "min": 0.0, "max": 120.0, "default": 80.0, "step": 5.0, "help": "Temperature of treatment in °C"},
    "Treatment time (min)": {"type": "numeric", "min": 0.0, "max": 500.0, "default": 30.0, "step": 5.0, "help": "Duration of treatment in minutes"},
    "Additives": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 0.0, "step": 1000.0, "help": "Code identifying additives used"},
    "Additives Concentration (%)": {"type": "numeric", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Concentration of additives in percentage"},
    "pH": {"type": "numeric", "min": 1.0, "max": 14.0, "default": 7.0, "step": 0.1, "help": "pH value of the solution"},
    "Type of salt": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 60000.0, "step": 1000.0, "help": "Code identifying the type of salt used"},
    "ionic strength (M)": {"type": "numeric", "min": 0.0, "max": 1.0, "default": 0.1, "step": 0.01, "help": "Ionic strength in Molarity"},
    "Heating temperature (°C) for gel preparation": {"type": "numeric", "min": 50.0, "max": 120.0, "default": 90.0, "step": 5.0, "help": "Temperature for gel preparation in °C"},
    "Heating/hold time (min)": {"type": "numeric", "min": 0.0, "max": 120.0, "default": 30.0, "step": 5.0, "help": "Duration of heating/hold in minutes"},
    "Samples stored (°C)": {"type": "numeric", "min": 0.0, "max": 30.0, "default": 4.0, "step": 1.0, "help": "Storage temperature in °C"},
    "Storage time (h)": {"type": "numeric", "min": 0.0, "max": 72.0, "default": 12.0, "step": 1.0, "help": "Duration of storage in hours"},
    "If a gel can be formed (0-1)": {"type": "numeric", "min": 0.0, "max": 1.0, "default": 1.0, "step": 1.0, "help": "Binary value indicating if gel can be formed (1) or not (0)"}
}

# Create columns for cleaner layout in the sidebar
def create_feature_inputs():
    inputs = {}
    
    for feature, config in feature_info.items():
        if config["type"] == "numeric":
            # Ensure all slider values are of consistent float type
            inputs[feature] = st.sidebar.slider(
                feature, 
                min_value=float(config["min"]), 
                max_value=float(config["max"]), 
                value=float(config["default"]), 
                step=float(config["step"]),
                help=config["help"]
            )
    
    return inputs

# Create reset button
if st.sidebar.button("Reset to Defaults"):
    for feature in feature_info:
        st.session_state[feature] = feature_info[feature]["default"]

# Get inputs
user_inputs = create_feature_inputs()

# Create a prediction button
if st.sidebar.button("Predict Gel Hardness"):
    # Convert inputs to DataFrame for prediction
    input_df = pd.DataFrame([user_inputs])
    
    # Load the model
    model = load_model()
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display the prediction in main area
    st.header("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Gel Hardness (g)", f"{prediction:.2f} g")
    
    with col2:
        # Create a gauge or progress bar for visualization
        fig, ax = plt.subplots(figsize=(3, 1))
        # Define a color map for visualization
        cmap = plt.cm.RdYlGn
        
        # Normalize the prediction for visualization (assuming max around 2000g based on the data)
        max_value = 2000
        normalized_value = min(prediction / max_value, 1.0)  
        
        # Create a simple horizontal gauge
        ax.barh(0, normalized_value, color=cmap(normalized_value), height=0.3)
        ax.barh(0, 1, color='lightgrey', height=0.3, alpha=0.3)
        
        # Customize the appearance
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '500', '1000', '1500', '2000'])
        ax.set_xlabel('Hardness (g)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        st.pyplot(fig)
    
    # Feature importance visualization
    st.subheader("Feature Importance")
    
    # Get feature importance from the model (assuming XGBoost)
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': input_df.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    ax.set_title('Feature Importance for Prediction')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    st.pyplot(fig)
    
    # Show input parameters used
    st.subheader("Input Parameters Used")
    st.dataframe(input_df)
    
    # Add an explanation section
    st.subheader("How to Interpret the Results")
    st.markdown("""
    - **Gel Hardness**: The predicted hardness/firmness/strength of the protein gel in grams.
    - **Feature Importance**: Shows which parameters have the most impact on the prediction.
    - **Tips for Optimization**:
        - Focus on adjusting parameters with higher importance scores.
        - The ionic strength, pH, and protein concentration typically have significant effects on gel hardness.
        - Storage conditions can also affect the final gel properties.
    """)

# Add a section explaining the model
st.markdown("---")
st.header("About the Model")
with st.expander("Model Information"):
    st.markdown("""
    This prediction tool uses an XGBoost regression model trained on protein gel hardness data. The model was trained with the following metrics:
    
    - **R² Score**: 95.46%
    - **Mean Absolute Error**: 74.47g
    
    The model takes into account various protein properties, treatment conditions, and storage parameters to predict the gel hardness in grams.
    """)

with st.expander("How to Use This Tool"):
    st.markdown("""
    1. Adjust the parameters in the sidebar according to your protein gel formulation.
    2. Click the "Predict" button to get the predicted gel hardness.
    3. Examine the feature importance to understand which parameters have the most impact.
    4. Iterate by changing parameters to optimize your gel hardness.
    
    This tool can be used for:
    - Research and development of new protein gels
    - Quality control of existing formulations
    - Educational purposes to understand the relationship between parameters and gel properties
    """)

with st.expander("Data Information"):
    st.markdown("""
    The model was trained on a dataset containing 1,073 samples with various protein properties. The dataset includes information about:
    
    - Protein types and concentrations
    - Treatment conditions and methods
    - Additive types and concentrations
    - pH and ionic strength
    - Heating and storage conditions
    - Resulting gel hardness measurements
    
    The dataset was cleaned and preprocessed to handle missing values and outliers before model training.
    """)

# QR Code Generator Section
st.markdown("---")
st.header("Share This Application")
st.markdown("""
If you want to share this application with colleagues, you can generate a QR code linking to it.
Once the app is deployed online, this QR code will allow others to easily access it.
""")

# QR code generator functionality
if st.button("Generate QR Code for this App"):
    import qrcode
    from PIL import Image
    import io
    import base64
    
    # URL of the deployed app (replace with your actual URL when deployed)
    # For local development, this will be a placeholder
    app_url = "https://protein-gel-hardness-predictor.streamlit.app"  # Replace with your actual URL when deployed
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(app_url)
    qr.make(fit=True)
    
    # Create an image from the QR Code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Display the image
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Display the QR code
    st.image(f"data:image/png;base64,{img_str}", caption=f"QR Code for {app_url}")
    
    # Add download button for the QR code
    st.download_button(
        label="Download QR Code",
        data=buffered.getvalue(),
        file_name="protein_gel_predictor_qr.png",
        mime="image/png"
    )
    
    st.info("Note: This QR code links to the URL shown in the caption. Make sure to update the URL in the code once your app is deployed.")

# Footer
st.markdown("---")
st.markdown("Developed for protein gel research and analysis | ©2025")