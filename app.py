import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import shap
from sklearn.cluster import KMeans # Although we load clusters, keep import for context

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Superconductor Explorer")
st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress Pyplot warning

# --- Load Saved Objects ---
# Wrap loading in functions with caching for performance
@st.cache_resource
def load_model_and_scaler():
    try:
        # Define RMSE for loading model
        def rmse(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        
        model = tf.keras.models.load_model('best_model.keras', custom_objects={'rmse': rmse})
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

@st.cache_data
def load_data_samples():
    try:
        X_scaled = np.load('X_test_sample_scaled_ui.npy', allow_pickle=True)
        X_unscaled = pd.read_pickle('X_test_sample_unscaled_ui.pkl')
        y_actual = pd.read_pickle('y_test_sample_ui.pkl')
        return X_scaled, X_unscaled, y_actual
    except Exception as e:
        st.error(f"Error loading data samples: {e}")
        return None, None, None

@st.cache_data
def load_shap_data():
    try:
        shap_values = np.load('shap_values_ui.npy', allow_pickle=True)
        expected_value = np.load('expected_value_ui.npy', allow_pickle=True)
        # Ensure expected_value is a float if it's a 0-d array
        if expected_value.shape == ():
             expected_value = float(expected_value)
        clusters = np.load('clusters_ui.npy', allow_pickle=True)
        return shap_values, expected_value, clusters
    except Exception as e:
        st.error(f"Error loading SHAP data: {e}")
        return None, None, None

model, scaler = load_model_and_scaler()
X_sample_scaled, X_sample_unscaled, y_sample_actual = load_data_samples()
shap_values_sample, expected_value, clusters_sample = load_shap_data()

# Check if loading failed
if model is None or X_sample_scaled is None or shap_values_sample is None:
    st.stop() # Stop execution if essential components are missing

# --- App Title ---
st.title("🔬 Superconductor T_c Prediction & Explainability Explorer")
st.markdown("Explore predictions and explanations for superconductor materials based on their chemical features.")

# --- Sidebar Controls ---
st.sidebar.header("Select Material")
# Create meaningful labels for the dropdown
material_labels = [f"Material #{i} (Cluster {clusters_sample[i]})" for i in range(len(X_sample_unscaled))]
selected_label = st.sidebar.selectbox("Choose a material from the test sample:", material_labels)

# Get the index from the selected label
selected_index = material_labels.index(selected_label)

st.sidebar.info(f"Displaying data for Material Index: {selected_index}")

# --- Main Page Display ---

# --- 1. Prediction ---
st.header("Prediction vs Actual")

# Get data for the selected material
material_scaled = X_sample_scaled[selected_index:selected_index+1] # Keep 2D shape
material_unscaled = X_sample_unscaled.iloc[selected_index]
actual_temp = y_sample_actual.iloc[selected_index]
material_cluster = clusters_sample[selected_index]

# Make prediction
predicted_temp = model.predict(material_scaled)[0][0]
error = predicted_temp - actual_temp

col1, col2, col3 = st.columns(3)
col1.metric("Predicted T_c", f"{predicted_temp:.2f} K")
col2.metric("Actual T_c", f"{actual_temp:.2f} K")
col3.metric("Prediction Error", f"{error:.2f} K")
st.info(f"This material belongs to **Cluster {material_cluster}** based on SHAP value similarity.")


# --- 2. SHAP Explanation ---
st.header("Why did the model predict this temperature?")
st.markdown("The plot below shows which features pushed the prediction **higher** (red arrows) vs **lower** (blue arrows) compared to the average prediction.")

# SHAP Force Plot for the selected instance
shap.initjs() # Initialize Javascript visualization library
st.pyplot(shap.force_plot(
    expected_value,
    shap_values_sample[selected_index, :],
    material_unscaled,
    matplotlib=True # Use Matplotlib backend for st.pyplot
), bbox_inches='tight')


# --- 3. Feature Values ---
st.header("Feature Values")
with st.expander("Show the 81 features for this material"):
    st.dataframe(material_unscaled)

# --- Add global plot for reference ---
st.header("Global Feature Importance (All Samples)")
with st.expander("Show overall feature importance"):
     # Use the full sample data loaded for the UI
     st.markdown("#### Bar Plot (Overall Ranking)")
     st.pyplot(shap.summary_plot(
         shap_values_sample,
         X_sample_unscaled,
         plot_type="bar"
     ), bbox_inches='tight')

     st.markdown("#### Bee Swarm Plot (Detailed View)")
     st.pyplot(shap.summary_plot(
         shap_values_sample,
         X_sample_unscaled
     ), bbox_inches='tight')