import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load(r"Model_Save\modelBoosting.joblib")

# Protein dictionary
proteins_dict = {
    'Tilapia (Orechromis niloticus)': 20113, 'conglycinin (7S)': 10303, 'Amaranth protein isolate': 12601,
    'Black bean protein isolate (BBPI)': 11804, 'Bovine plasma protein': 30403, 'Casein Protein': 30104,
    'Chicken plasma protein': 30401, 'Chicken plasma protein ': 30401, 'Chickpea Protein Concentration ': 10417,
    'Chickpea Protein Isolate': 10412, 'Cowpea Protein': 11701, 'Cowpea protein isolate': 11701,
    'Dried egg white protein': 30303, 'Duck Egg White Protein': 30305, 'Duck egg white ': 30305, 'Egg White Protein': 30302,
    'Egg ovalbumin Protein': 30304, 'Egg white ': 30302, 'European Anguilla anguilla (eel) protein isolate (EPI)': 20119,
    'Ginkgo Seed Protein Isolate (GSPI)': 10501, 'Glycinin (11S)': 10302, 'Lentil Protein Isolate (LPI)': 10107,
    'Mung Bean Protein Isolate (MBPI)': 10801, 'Native soy protein isolate (SPIn)': 10305,
    'Oyster (Crassostrea gigas) protein': 20115, 'Pea Protein Concentrate (PPC)': 10404, 'Pea Protein Isolate (PPI)': 10401,
    'Pea Protein Isolate (PPI) ': 10401, 'Peanut Protein Isolate (PPI)': 11101, 'Porcine Plasma protein': 30402,
    'Porcine plasma protein': 30402, 'Potato Protein': 12002, 'Potato Protein Isolate ': 12001, 'Potato protein isolate (PPI)': 12001,
    'Rapeseed protein isolates': 12301, 'Rice Glutelin (RG)': 11301, 'Sheep plasma protein ': 30404,
    'Silver carp Myofibrillar Protein': 20103, 'Soy Protein Isolate (SPI)': 10301, 'Soy protein Isolate': 10301,
    'Soy protein isolate': 10301, 'Walnut Protein Isolate (WNPI)': 11501, 'Wheat gluten': 11203,
    'Whey Protein Isolate (WPI)': 30105, 'Whey protein': 30108, 'Whey protein concentrate': 30107, 'Whey protein isolate': 30105,
    'oat protein isolate (OPI)': 11002, 'pinto bean protein isolate': 11803, 'whey protein concentrate': 30107,
    'whey protein isolate': 30105, 'whey protein isolate (WPI)': 30105
}

# Function to display the two options: Upload file or adjust sliders
def display_mode_selection():
    option = st.radio("Select the mode that suits you", ("Upload a data file", "Adjust sliders"))
    
    if option == "Upload a data file":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            # Read the uploaded CSV file
            data = pd.read_csv(uploaded_file)
            st.write(data)
            return data
    elif option == "Adjust sliders":
        # Show the sliders
        return None

# Display the mode selection
selected_data = display_mode_selection()

if selected_data is None:
    # Sliders section to adjust values
    st.sidebar.header("Choose the variable values")

    # Dropdown menu for proteins
    selected_protein = st.sidebar.selectbox("Choose the protein", list(proteins_dict.keys()))
    protein_code = proteins_dict[selected_protein]
    
    protein_codes = st.sidebar.slider("Protein codes", min_value=1, max_value=100, value=protein_code)
    protein_concentration = st.sidebar.slider("Protein Concentration (%)", min_value=0.0, max_value=100.0, value=50.0)
    treatment_code = st.sidebar.slider("Treatment code", min_value=1, max_value=10, value=5)
    treatment_condition_code = st.sidebar.slider("Treatment condition code", min_value=1, max_value=10, value=5)
    treatment_condition_value = st.sidebar.slider("Treatment condition value", min_value=0.0, max_value=100.0, value=50.0)
    treatment_temperature = st.sidebar.slider("Treatment temperature (°C)", min_value=-20, max_value=100, value=25)
    treatment_time = st.sidebar.slider("Treatment time (min)", min_value=0.1, max_value=100.0, value=5.0)
    additive_concentration = st.sidebar.slider("Additives Concentration (%)", min_value=0.0, max_value=100.0, value=50.0)
    pH = st.sidebar.slider("pH", min_value=1.0, max_value=14.0, value=7.0)
    ionic_strength = st.sidebar.slider("Ionic strength (M)", min_value=0.0, max_value=1.0, value=0.5)
    heating_temperature = st.sidebar.slider("Heating temperature (°C) for gel preparation", min_value=0, max_value=200, value=100)
    heating_hold_time = st.sidebar.slider("Heating/hold time (min)", min_value=0.1, max_value=300.0, value=60.0)
    samples_stored = st.sidebar.slider("Samples stored (°C)", min_value=-80, max_value=25, value=4)
    storage_time = st.sidebar.slider("Storage time (h)", min_value=0, max_value=1000, value=500)
    gel_formation = st.sidebar.slider("If a gel can be formed (0-1)", min_value=0, max_value=1, value=0)

    # Categorical variables
    additives = st.sidebar.selectbox("Additives", ["None", "Additive A", "Additive B", "Additive C"])
    type_of_salt = st.sidebar.selectbox("Type of salt", ["NaCl", "KCl", "MgCl2"])

    # Encoding categorical variables
    additives_dict = {"None": 0, "Additive A": 1, "Additive B": 2, "Additive C": 3}
    type_of_salt_dict = {"NaCl": 0, "KCl": 1, "MgCl2": 2}

    additives_encoded = additives_dict[additives]
    type_of_salt_encoded = type_of_salt_dict[type_of_salt]

    # Create the feature array
    X_input = np.array([[ 
        protein_codes, protein_concentration, treatment_code, treatment_condition_code,
        treatment_condition_value, treatment_temperature, treatment_time,
        additives_encoded, additive_concentration, pH, type_of_salt_encoded,
        ionic_strength, heating_temperature, heating_hold_time, samples_stored,
        storage_time, gel_formation
    ]])

    # Create a mini dataset for client selection
    selection_client = pd.DataFrame(X_input, columns=[ 
        'Protein codes', 'Protein Concentration', 'Treatment code', 'Treatment condition code',
        'Treatment condition value', 'Treatment temperature', 'Treatment time', 'Additives',
        'Additive Concentration', 'pH', 'Type of Salt', 'Ionic strength', 'Heating temperature',
        'Heating/hold time', 'Samples stored', 'Storage time', 'Gel formation'
    ])
    
    st.write("### Client Selection")
    st.write(selection_client)

    # Prediction button (just above the grey line)
    predict_button = st.button("Predict")
    if predict_button:
        prediction = model.predict(X_input)  
        st.write("### Prediction result:")
        st.write(f"Estimated gel hardness: {prediction[0]}")

    # Add a grey blurred line to separate the prediction area from the information area
    st.markdown(
        """
        <hr style="border: none; border-top: 1px solid gray; margin: 50px 0;"/>
        <div style="filter: blur(5px);"></div>
        """, 
        unsafe_allow_html=True
    )

    # "About the model" section
    st.markdown("### About the model")
    st.write(
        """
        We used XGBoost as the model to predict gel hardness, with an accuracy score of 95%. This model is an ensemble algorithm that combines several decision trees to provide robust and efficient predictions. It is particularly suited for regression and classification problems with large datasets and complex relationships between variables.
        """
    )

    # Display the image after the "About the model" section
    st.image(r"C:\Model_to_predict_GelHardness\app\img\PredictGel.png", caption="Gel Hardness Prediction", use_column_width=True)

    # Information section
    st.markdown("## Information")

    # New title for the variables
    st.markdown("### Model Variables")

    # Initial variables (first three variables)
    first_three_vars = """
    - **Protein codes** : Identifier of the chosen protein.
    - **Protein Concentration** : Protein concentration as a percentage.
    - **Treatment code** : Code representing the type of treatment applied.
    """
    
    # Button to show more or less
if 'show_more' not in st.session_state:
    st.session_state.show_more = False

def toggle_show_more():
    st.session_state.show_more = not st.session_state.show_more

# Display all variables based on 'show_more' state
if st.session_state.show_more:
    # Display all variables
    st.markdown(first_three_vars + """
    - **Treatment condition code** : Code representing the treatment conditions applied.
    - **Treatment condition value** : Value of the treatment applied.
    - **Treatment temperature** : Temperature applied during the treatment (°C).
    - **Treatment time** : Duration of the treatment (minutes).
    - **Additives** : Additive used (e.g., gelling agent).
    - **Additive Concentration** : Additive concentration (percentage).
    - **pH** : pH level of the sample.
    - **Ionic strength** : Ionic strength of the medium.
    - **Heating temperature** : Temperature for gel preparation (°C).
    - **Heating/hold time** : Heating duration.
    - **Samples stored** : Sample storage temperature (°C).
    - **Storage time** : Time the sample was stored.
    - **Gel formation** : Whether a gel can be formed (binary).
    """)
else:
    st.markdown(first_three_vars)
    
# Display toggle button
st.button("Show more", on_click=toggle_show_more)