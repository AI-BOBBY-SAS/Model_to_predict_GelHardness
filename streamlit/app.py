import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Gemini Validation Dashboard - Protein Gel Data",
    page_icon="📊",
    layout="wide"
)

# Titre principal
st.title("Gemini Validation Dashboard: Protein Gel Data Extraction")
st.markdown("Continuous validation workflow for scientific article data extraction")

# Données pour les statistiques de validation
summary_stats = {
    "totalSamples": 87,
    "samplesWithErrors": 20,
    "samplesWithoutErrors": 67,
    "averageAccuracy": 95.59,
    "samplesAboveThreshold": 67,
    "samplesBelowThreshold": 20
}

# Données pour la fréquence des erreurs
error_frequency_data = [
    {"field": "Additive Concentration", "count": 5},
    {"field": "Heating Time", "count": 4},
    {"field": "pH", "count": 4},
    {"field": "Heating Temperature", "count": 4},
    {"field": "Hardness", "count": 3},
    {"field": "Protein Concentration", "count": 3}
]
error_frequency_df = pd.DataFrame(error_frequency_data)

# Données pour la dureté par type de protéine
hardness_by_protein_data = [
    {"protein": "Egg ovalbumin Protein", "avgHardness": 14415.16, "count": 12},
    {"protein": "Chickpea Protein Isolate", "avgHardness": 1718.20, "count": 2},
    {"protein": "Pea Protein Isolate (PPI)", "avgHardness": 1371.50, "count": 2},
    {"protein": "Chickpea Protein Concentration", "avgHardness": 1356.20, "count": 1},
    {"protein": "Whey Protein Isolate (WPI)", "avgHardness": 1075.35, "count": 4},
    {"protein": "Potato Protein", "avgHardness": 766.60, "count": 24},
    {"protein": "Soy Protein Isolate (SPI)", "avgHardness": 663.79, "count": 4},
    {"protein": "Lentil Protein Isolate (LPI)", "avgHardness": 534.13, "count": 2},
    {"protein": "Mung Bean Protein Isolate", "avgHardness": 179.35, "count": 7},
    {"protein": "Potato Protein Isolate", "avgHardness": 151.39, "count": 13},
    {"protein": "Ginkgo Seed Protein Isolate", "avgHardness": 103.91, "count": 12}
]
hardness_by_protein_df = pd.DataFrame(hardness_by_protein_data)

# Données pour la dureté par pH
hardness_by_ph_data = [
    {"ph": "3", "avgHardness": 426.13, "count": 3},
    {"ph": "5", "avgHardness": 241.17, "count": 5},
    {"ph": "6", "avgHardness": 10838.32, "count": 16},
    {"ph": "7", "avgHardness": 737.69, "count": 45},
    {"ph": "9", "avgHardness": 761.00, "count": 2}
]
hardness_by_ph_df = pd.DataFrame(hardness_by_ph_data)

# Données pour la dureté par concentration de protéine
hardness_by_concentration_data = [
    {"concentration": "10", "avgHardness": 420.03, "count": 17},
    {"concentration": "12", "avgHardness": 103.91, "count": 12},
    {"concentration": "14", "avgHardness": 674.11, "count": 9},
    {"concentration": "15", "avgHardness": 1017.71, "count": 2},
    {"concentration": "16", "avgHardness": 95.44, "count": 1},
    {"concentration": "18", "avgHardness": 1105.15, "count": 11},
    {"concentration": "20", "avgHardness": 1270.26, "count": 7},
    {"concentration": "25", "avgHardness": 6998.00, "count": 25}
]
hardness_by_concentration_df = pd.DataFrame(hardness_by_concentration_data)

# Données pour les résultats de validation
validation_results_data = [
    {"id": 1, "citation": "Zhang et al. (2024)", "protein": "Ginkgo Seed Protein Isolate (GSPI)", "accuracy": 100, "totalFields": 6, "correctFields": 6, "status": "Pass", "incorrectFields": []},
    {"id": 2, "citation": "Zhang et al. (2024)", "protein": "Ginkgo Seed Protein Isolate (GSPI)", "accuracy": 83.33, "totalFields": 6, "correctFields": 5, "status": "Review Required", "incorrectFields": ["Heating Time"]},
    {"id": 3, "citation": "Zhang et al. (2024)", "protein": "Ginkgo Seed Protein Isolate (GSPI)", "accuracy": 66.67, "totalFields": 6, "correctFields": 4, "status": "Review Required", "incorrectFields": ["pH", "Hardness"]},
    {"id": 4, "citation": "Tang et al. (2024)", "protein": "Lentil Protein Isolate (LPI)", "accuracy": 100, "totalFields": 6, "correctFields": 6, "status": "Pass", "incorrectFields": []},
    {"id": 5, "citation": "Tang et al. (2024)", "protein": "Whey Protein Isolate (WPI)", "accuracy": 100, "totalFields": 6, "correctFields": 6, "status": "Pass", "incorrectFields": []},
    {"id": 6, "citation": "Yaputri et al. (2024)", "protein": "Pea Protein Isolate (PPI)", "accuracy": 83.33, "totalFields": 6, "correctFields": 5, "status": "Review Required", "incorrectFields": ["Protein Concentration"]},
    {"id": 7, "citation": "Li et al. (2024)", "protein": "Potato Protein", "accuracy": 100, "totalFields": 6, "correctFields": 6, "status": "Pass", "incorrectFields": []}
]
validation_results_df = pd.DataFrame(validation_results_data)

# Données pour la tendance hebdomadaire
weekly_trend_data = [
    {"week": "Week 1", "accuracy": 81.7},
    {"week": "Week 2", "accuracy": 84.9},
    {"week": "Week 3", "accuracy": 87.2},
    {"week": "Week 4", "accuracy": 89.5},
    {"week": "Week 5", "accuracy": 91.8},
    {"week": "Week 6", "accuracy": 93.4}
]
weekly_trend_df = pd.DataFrame(weekly_trend_data)

# Données pour les échantillons
sample_data = [
    {"protein": "Ginkgo Seed Protein Isolate (GSPI)", "concentration": 12, "ph": 5, "additive": "L-Theanine", "additiveConc": 0.5, "heatingTemp": 90, "heatingTime": 30, "hardness": 155.91},
    {"protein": "Whey Protein Isolate (WPI)", "concentration": 14, "ph": 7, "additive": "None", "additiveConc": 0, "heatingTemp": 90, "heatingTime": 30, "hardness": 1930.95},
    {"protein": "Potato Protein", "concentration": 18, "ph": 7, "additive": "High-acyl gellan gum", "additiveConc": 1.5, "heatingTemp": 90, "heatingTime": 30, "hardness": 1080.88},
    {"protein": "Egg ovalbumin Protein", "concentration": 25, "ph": 6.6, "additive": "None", "additiveConc": 0, "heatingTemp": 70, "heatingTime": 15, "hardness": 15610.59}
]
sample_df = pd.DataFrame(sample_data)

# Menu latéral pour la navigation
view = st.sidebar.radio("Navigation", ["Dashboard", "Workflow"])

if view == "Dashboard":
    # Statistiques de résumé
    st.subheader("Validation Summary")
    st.markdown("Protein Gel Extraction Model - April 2025")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Accuracy", f"{summary_stats['averageAccuracy']}%")
    with col2:
        st.metric("Total Samples", summary_stats['totalSamples'])
    with col3:
        st.metric("Samples Above Threshold", summary_stats['samplesAboveThreshold'])
    with col4:
        st.metric("Samples Below Threshold", summary_stats['samplesBelowThreshold'])
    
    # Onglets pour les différentes vues
    tab1, tab2, tab3 = st.tabs(["Accuracy Metrics", "Gel Hardness Data", "Trend Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Extraction Accuracy")
            accuracy_fig = px.pie(
                values=[summary_stats["samplesWithoutErrors"], summary_stats["samplesWithErrors"]],
                names=["Correct Extractions", "Extractions with Errors"],
                color_discrete_sequence=["#4ade80", "#f87171"],
                hole=0.4
            )
            st.plotly_chart(accuracy_fig, use_container_width=True)
        
        with col2:
            st.subheader("Most Frequently Incorrect Fields")
            error_fig = px.bar(
                error_frequency_df,
                x="field",
                y="count",
                color_discrete_sequence=["#f87171"]
            )
            error_fig.update_layout(xaxis_title="Field", yaxis_title="Error Count")
            st.plotly_chart(error_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Average Gel Hardness by Protein Type")
        hardness_protein_fig = px.bar(
            hardness_by_protein_df.sort_values("avgHardness", ascending=True),
            y="protein",
            x="avgHardness",
            orientation="h",
            color_discrete_sequence=["#60a5fa"]
        )
        hardness_protein_fig.update_layout(xaxis_title="Average Hardness (g)", yaxis_title="Protein Type")
        st.plotly_chart(hardness_protein_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hardness by pH")
            hardness_ph_fig = px.bar(
                hardness_by_ph_df,
                x="ph",
                y="avgHardness",
                color_discrete_sequence=["#34d399"]
            )
            hardness_ph_fig.update_layout(xaxis_title="pH", yaxis_title="Average Hardness (g)")
            st.plotly_chart(hardness_ph_fig, use_container_width=True)
        
        with col2:
            st.subheader("Hardness by Protein Concentration (%)")
            hardness_conc_fig = px.bar(
                hardness_by_concentration_df,
                x="concentration",
                y="avgHardness",
                color_discrete_sequence=["#a78bfa"]
            )
            hardness_conc_fig.update_layout(xaxis_title="Concentration (%)", yaxis_title="Average Hardness (g)")
            st.plotly_chart(hardness_conc_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Weekly Accuracy Trend")
        trend_fig = px.line(
            weekly_trend_df,
            x="week",
            y="accuracy",
            markers=True,
            color_discrete_sequence=["#3b82f6"]
        )
        trend_fig.update_layout(xaxis_title="Week", yaxis_title="Accuracy (%)")
        trend_fig.update_yaxes(range=[75, 100])
        st.plotly_chart(trend_fig, use_container_width=True)
    
    # Tableau des résultats de validation
    st.subheader("Validation Results by Sample")
    
    # Formater les données pour affichage
    display_df = validation_results_df.copy()
    display_df["incorrectFields"] = display_df["incorrectFields"].apply(lambda x: ", ".join(x) if x else "None")
    display_df["accuracy"] = display_df["accuracy"].apply(lambda x: f"{x:.1f}%")
    display_df["fields"] = display_df.apply(lambda row: f"{row['correctFields']}/{row['totalFields']}", axis=1)
    
    # Afficher le tableau
    st.dataframe(
        display_df[["id", "citation", "protein", "accuracy", "status", "fields", "incorrectFields"]],
        use_container_width=True
    )
    
    # Tableau des échantillons
    st.subheader("Protein Gel Sample Data")
    st.dataframe(sample_df, use_container_width=True)
    
    # Barre de seuil de validation
    st.subheader("Validation Threshold")
    progress_val = summary_stats["averageAccuracy"] / 100
    st.progress(progress_val)
    st.caption(f"Current accuracy: {summary_stats['averageAccuracy']}% (Threshold: 85%)")

else:  # Workflow view
    st.subheader("Continuous Validation Workflow for Protein Gel Extraction")
    
    workflow_steps = [
        {"title": "Reference Data Management", "description": "Maintain structured ground truth dataset of gel properties from scientific articles"},
        {"title": "Gemini Extraction Pipeline", "description": "Analyze scientific PDFs → Extract structured data on protein gels"},
        {"title": "Auto-Comparison & Validation", "description": "Python script compares extraction output with reference data"},
        {"title": "Alerts & Notifications", "description": "If accuracy < 85%, send alert for manual review"},
        {"title": "Review & Improvement", "description": "Update reference data and optimize Gemini prompts"}
    ]
    
    for i, step in enumerate(workflow_steps):
        st.markdown(f"### Step {i+1}: {step['title']}")
        st.markdown(f"_{step['description']}_")
        st.divider()