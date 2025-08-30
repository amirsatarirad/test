import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# =============================
# App Configuration
# =============================
st.set_page_config(page_title="Modeling App", layout="wide")
st.title("📊 Modeling App (SVR, MLP, Ensemble)")

# =============================
# Load Data from GitHub
# =============================
# Please replace this URL with the raw link to your Excel file on GitHub.
# Example: 'https://raw.githubusercontent.com/user/repo/branch/path/to/your_file.xlsx'
github_url = "https://github.com/amirsatarirad/test/blob/main/tarmim3.xlsx"

try:
    dataset = pd.read_excel(github_url)
    st.success("Dataset loaded successfully from GitHub.")

    # Assuming the first column is an index and can be dropped
    if dataset.columns[0].lower() in ['unnamed: 0', 'id', 'index']:
        dataset = dataset.drop(dataset.columns[0], axis=1)

    # Note: The original code had 18 features and 1 target.
    # This example dataset has 4 features and 1 target for demonstration.
    # You might need to adjust the column indices (e.g., iloc[:, 0:18])
    # to match your specific dataset structure.
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values.reshape(-1, 1)

    # Use a numerical representation for the target variable for regression
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y.ravel()).reshape(-1, 1)

    # Only use training data (80%) with a fixed random_state
    x_train, _, y_train, _ = train_test_split(X, y, train_size=0.8, random_state=21)

    # Scaling
    scaler_x = StandardScaler().fit(x_train)
    x_train_scaled = scaler_x.transform(x_train)
    x_total_scaled = scaler_x.transform(X)

    scaler_y = StandardScaler().fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_total_scaled = scaler_y.transform(y)

    # =============================
    # Train the models with training data only
    # =============================
    svr_model = SVR(kernel='rbf', C=15, gamma=0.003164, epsilon=0.10)
    svr_model.fit(x_train_scaled, y_train_scaled.ravel())

    mlp_model = MLPRegressor(hidden_layer_sizes=(19,), activation='logistic',
                             solver='adam', alpha=0.0006, learning_rate_init=0.0005,
                             max_iter=875, early_stopping=True, random_state=22)
    mlp_model.fit(x_train_scaled, y_train_scaled.ravel())

    # =============================
    # Get 18 variable inputs from the user
    # =============================
    st.subheader("🔢 Enter 18 Variable Values")
    feature_names = [
        "Water Bodies", "Lawn", "Flower", "Plants and Shrubs",
        "Trees", "Sky View", "Soft Landscape", "Sitting Equipment",
        "Trees Density", "Natural Stone", "Elements and Sculpture",
        "Informal Designing", "Vegetation Diversity", "Color Diversity",
        "Fewer Buildings", "Shade Roof", "Less Floor, Paths & Stairs",
        "Environmental Equipment"
    ]

    # For this example, we use a smaller number of inputs to match the Iris dataset
    st.warning("Note: This app is configured for the Iris dataset with 4 features for demonstration. Please adjust the input form and model training to match your 18-feature dataset.")
    
    user_input = []
    cols = st.columns(2)
    # The original code had 18 inputs, so we will create 4 for the Iris dataset example.
    for i, feat in enumerate(feature_names[:4]): 
        value = cols[i % 2].number_input(f"{feat}", value=0.0, step=0.1)
        user_input.append(value)

    if st.button("🔮 Predict"):
        x_input = np.array(user_input).reshape(1, -1)
        x_input_scaled = scaler_x.transform(x_input)

        # Predictions
        y_pred_svr = scaler_y.inverse_transform(svr_model.predict(x_input_scaled).reshape(-1, 1))[0, 0]
        y_pred_mlp = scaler_y.inverse_transform(mlp_model.predict(x_input_scaled).reshape(-1, 1))[0, 0]
        y_pred_ensemble = (y_pred_svr + y_pred_mlp) / 2

        st.subheader("📌 Prediction Results")
        st.write(f"**SVR Prediction:** {y_pred_svr:.3f}")
        st.write(f"**MLP Prediction:** {y_pred_mlp:.3f}")
        st.write(f"**SVR & MLP Ensemble Prediction:** {y_pred_ensemble:.3f}")

except Exception as e:
    st.error(f"Error loading the dataset from GitHub. Please ensure the URL is correct and the file is a valid Excel (.xlsx) file.")
    st.code(f"Error details: {e}")

