import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "modelBoosting.joblib")
    return joblib.load(model_path)

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
# Define columns based on the model
feature_info = {
        "Protein Name": {
        "type": "select",
        "options": sorted([
            "Whey Protein Isolate", "Beta-lactoglobulin", "Alpha-lactalbumin", "Casein",
            "Soy Protein Isolate", "Pea Protein", "Egg Albumin", "Wheat Gluten",
            "Myofibrillar Protein", "Collagen", "Rice Protein", "Chia Protein",
            "Quinoa Protein", "Lentil Protein", "Gelatin", "Hemp Protein",
            "Potato Protein", "Zein Protein", "Lupin Protein", "Flaxseed Protein"
        ]),
        "default": "Alpha-lactalbumin",
        "help": "Name of the protein used (corresponds to 'Protein' column in dataset)"
    },

    "Protein Concentration (%)": {"type": "numeric", "min": 1.0, "max": 30.0, "default": 10.0, "step": 0.25, "help": "Concentration of protein in percentage"},
    "pH": {"type": "numeric", "min": 1.0, "max": 14.0, "default": 7.0, "step": 0.1, "help": "pH value of the solution"},
    "Type of salt": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 60000.0, "step": 1, "help": "Code identifying the type of salt used"},
    "ionic strength (M)": {"type": "numeric", "min": 0.0, "max": 1.0, "default": 0.1, "step": 0.001, "help": "Ionic strength in Molarity"},
    "Treatment code": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 30000.0, "step": 1, "help": "Code identifying the treatment type"},
    "Treatment condition code": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 30000.0, "step": 1, "help": "Code for treatment condition"},
    "Treatment condition value": {"type": "numeric", "min": 0.0, "max": 500.0, "default": 100.0, "step": 10.0, "help": "Value for treatment condition"},
    "Treatment temperature ( °C)": {"type": "numeric", "min": 0.0, "max": 120.0, "default": 80.0, "step": 5.0, "help": "Temperature of treatment in °C"},
    "Treatment time (min)": {"type": "numeric", "min": 0.0, "max": 500.0, "default": 30.0, "step": 5.0, "help": "Duration of treatment in minutes"},
    "Additives": {"type": "numeric", "min": 0.0, "max": 90000.0, "default": 0.0, "step": 0.01, "help": "Code identifying additives used"},
    "Additives Concentration (%)": {"type": "numeric", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Concentration of additives in percentage"},
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
        elif config["type"] == "select":
            inputs[feature] = st.sidebar.selectbox(
                feature,
                options=config["options"],
                index=config["options"].index(config["default"]),
                help=config["help"]
            )
    
    return inputs

# Create reset button
if st.sidebar.button("Reset to Defaults"):
    for feature in feature_info:
        st.session_state[feature] = feature_info[feature]["default"]

# Get inputs
user_inputs = create_feature_inputs()

# Function to categorize gel hardness
def categorize_gel_hardness(hardness):
    if hardness < 1000:
        return "Soft", "0 to 1000 grams (up to 10 Newtons)", "#3498db"  # Light blue
    elif hardness < 5000:
        return "Firm", "1000 to 5000 grams (up to 50 Newtons)", "#f39c12"  # Orange
    else:
        return "Rigid", "5000 to 1,000,000 grams (up to 200 Newtons)", "#e74c3c"  # Red

# Create a prediction button
if st.sidebar.button("Predict Gel Hardness"):
    try:
        # Convert inputs to DataFrame for prediction
        input_df = pd.DataFrame([user_inputs])
        
        # Store the protein name for display and rename it to "Protein codes" for the model
        if "Protein Name" in input_df.columns:
            protein_name = input_df["Protein Name"].iloc[0]
            # Au lieu de supprimer la colonne, la renommer en "Protein codes" et lui attribuer une valeur par défaut
            input_df = input_df.rename(columns={"Protein Name": "Protein codes"})
            # Convertir la valeur en un code numérique basé sur votre dataset
            protein_codes = {
                "Whey Protein Isolate": 10301,  # Code le plus fréquent dans votre dataset
                "Beta-lactoglobulin": 10107,
                "Alpha-lactalbumin": 10501,
                "Casein": 10401,
                "Soy Protein Isolate": 11101,
                "Pea Protein": 30105,  # Très fréquent dans votre dataset
                "Egg Albumin": 30107,
                "Wheat Gluten": 30108,
                "Myofibrillar Protein": 30302,
                "Collagen": 30303,
                "Rice Protein": 11803,
                "Chia Protein": 11804,
                "Quinoa Protein": 12001,
                "Lentil Protein": 12002,
                "Gelatin": 30304,
                "Hemp Protein": 30401,
                "Potato Protein": 30402,
                "Zein Protein": 30403,
                "Lupin Protein": 30404,
                "Flaxseed Protein": 10305
            }
            # Assigner le code numérique correspondant ou une valeur par défaut
            input_df["Protein codes"] = protein_codes.get(protein_name, 10301)  # 10301 comme valeur par défaut
        
        # Load the model
        model = load_model()

        # Réorganiser les colonnes avant prédiction
        input_df = input_df[model.get_booster().feature_names]

        # Make prediction
        prediction = model.predict(input_df)[0]

        
        # Categorize the gel hardness
        category, range_text, category_color = categorize_gel_hardness(prediction)
        
        # Display the prediction in main area
        st.header("Prediction Results")
        
        # Display protein name
        st.subheader(f"Protein: {protein_name}")
        
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            st.metric("Predicted Gel Hardness (g)", f"{prediction:.2f} g")
        
        with col2:
            # Display gel category
            st.markdown(f"""
            <div style="border-radius:10px; padding:10px; background-color:{category_color}; color:white; text-align:center;">
                <h2 style="margin:0;">{category} Gel</h2>
                <p style="margin:0;">{range_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Create a gauge or progress bar for visualization
            fig, ax = plt.subplots(figsize=(4, 1))
            
            # Define threshold lines for categories
            max_value = 10000  # Extend to show all categories
            normalized_value = min(prediction / max_value, 1.0)
            soft_threshold = 1000 / max_value
            firm_threshold = 5000 / max_value
            
            # Create background with category zones
            ax.barh(0, soft_threshold, color='#3498db', height=0.3, alpha=0.3)  # Soft zone
            ax.barh(0, firm_threshold - soft_threshold, left=soft_threshold, color='#f39c12', height=0.3, alpha=0.3)  # Firm zone
            ax.barh(0, 1 - firm_threshold, left=firm_threshold, color='#e74c3c', height=0.3, alpha=0.3)  # Rigid zone
            
            # Create the gauge value
            ax.barh(0, normalized_value, color='darkblue', height=0.3)
            
            # Add threshold markers
            ax.axvline(x=soft_threshold, color='black', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=firm_threshold, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add category labels
            ax.text(soft_threshold/2, -0.2, "Soft", ha='center', va='center', fontsize=8)
            ax.text(soft_threshold + (firm_threshold-soft_threshold)/2, -0.2, "Firm", ha='center', va='center', fontsize=8)
            ax.text(firm_threshold + (1-firm_threshold)/2, -0.2, "Rigid", ha='center', va='center', fontsize=8)
            
            # Customize the appearance
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([0, 0.1, 0.5, 1])
            ax.set_xticklabels(['0', '1000', '5000', '10000+'])
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
        display_df = pd.DataFrame([user_inputs])
        st.dataframe(display_df)
        
        # Add an explanation section
        st.subheader("How to Interpret the Results")
        st.markdown(f"""
        - **Protein**: {protein_name}
        - **Gel Hardness**: The predicted hardness/firmness/strength of the protein gel is {prediction:.2f} grams.
        - **Gel Category**: The gel is classified as a **{category} Gel** ({range_text}).
        - **Feature Importance**: Shows which parameters have the most impact on the prediction.
        - **Tips for Optimization**:
            - Focus on adjusting parameters with higher importance scores.
            - The ionic strength, pH, and protein concentration typically have significant effects on gel hardness.
            - Storage conditions can also affect the final gel properties.
            - To achieve a different gel category, adjust the parameters that have the highest importance.
        """)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Traceback:")
        import traceback
        st.code(traceback.format_exc())

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

with st.expander("Gel Hardness Categories"):
    st.markdown("""
    The predicted gel hardness is categorized into three groups:
    
    - **Soft Gel**: 0 to 1000 grams (up to 10 Newtons)
      - Characteristics: Easily deformable, low resistance to pressure, spreadable
      - Applications: Soft spreads, fillings, some desserts
      
    - **Firm Gel**: 1000 to 5000 grams (up to 50 Newtons)
      - Characteristics: Moderate stability, cohesive structure, sliceable
      - Applications: Most food gels, yogurt, puddings, jellies
      
    - **Rigid/Stable Gel**: 5000 to 1,000,000 grams (up to 200 Newtons)
      - Characteristics: High structural integrity, brittle fracture, very high resistance to pressure
      - Applications: Industrial applications, specialized food products, technical gels
      
    These categories help in targeting specific applications and functional properties for your protein gel formulations.
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