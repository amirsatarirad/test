import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# =============================
# تنظیمات اپ
# =============================
file_url = 'https://raw.githubusercontent.com/amirsatarirad/test/refs/heads/main/tt.xlsx' 

st.set_page_config(page_title="Modeling App", layout="wide")
st.title("📊 Modeling App (SVR, MLP, Ensemble)")
# =============================


# if uploaded_file is not None:
dataset = pd.read_excel(file_url)
st.write("*Data Loaded!*")

# dataset = dataset.drop(dataset.columns[0], axis=1)

X = dataset.iloc[:, 0:18].values
y = dataset.iloc[:, 18:19].values

# فقط داده‌های آموزش (۸۰٪) با random_state ثابت
x_train, _, y_train, _ = train_test_split(X, y, train_size=0.8, random_state=21)

# نرمال‌سازی
scaler_x = StandardScaler().fit(x_train)
x_train_scaled = scaler_x.transform(x_train)
x_total_scaled = scaler_x.transform(X)

scaler_y = StandardScaler().fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_total_scaled = scaler_y.transform(y)

# =============================
# آموزش مدل‌ها فقط با train
# =============================
Svr = SVR(kernel='rbf', C=15, gamma=0.003164, epsilon=0.10)
Svr.fit(x_train_scaled, y_train_scaled.ravel())

mlp = MLPRegressor(hidden_layer_sizes=(19,), activation='logistic',
                    solver='adam', alpha=0.0006, learning_rate_init=0.0005,
                    max_iter=875, early_stopping=True, random_state=22)
mlp.fit(x_train_scaled, y_train_scaled.ravel())

# Getting amounts of Variables
# =============================
st.subheader("🔢 Getting amounts of Variables")
feature_names = [
    "Water Bodies (0-100%)", "lawn (0-100%)", "Flower (0-100%)", "Plants and Shurbs (0-100%)",
    "Trees (0-100%)", "Sky View (0-100%)", "Soft Landscape (0-100%)", "Sitting Equipments (0-100%)",
    "Trees Density (0-4)", "Natural Stone (0-100%)", "Elements and Sculpture (0-100%)",
    "Informal Designing (1-6)", "Vegetation Diversity(1-5)", "Color Diversity (1-10)",
    "Fewer Building (0-100%)", "Shade Roof (0-100%)", "Floor, Paths & Stairs (0-100%)",
    "Environmental Equipments (0-100%)"
]

user_input = []
cols = st.columns(3)
for i, feat in enumerate(feature_names):
    value = cols[i % 3].number_input(f"{feat}", value=0.0, step=0.1)
    user_input.append(value)
if  st.button("🔮 Predict"):
    x_input = np.array(user_input).reshape(1, -1)
    x_input_scaled = scaler_x.transform(x_input)
    if user_input==[] or max(user_input) == 0:
      st.subheader("📌 Prediction Results")
      st.write('Prediction Cannot Be Done!')
    else:
      # پیش‌بینی‌ها
      y_pred_svr = scaler_y.inverse_transform(Svr.predict(x_input_scaled).reshape(-1, 1))[0, 0]
      y_pred_mlp = scaler_y.inverse_transform(mlp.predict(x_input_scaled).reshape(-1, 1))[0, 0]
      y_pred_ensemble = (y_pred_svr + y_pred_mlp) / 2
  
      st.subheader("📌 Prediction Results")
      st.write(f"**SVR Prediction:** {y_pred_svr:.3f}")
      st.write(f"**MLP Prediction:** {y_pred_mlp:.3f}")
      st.write(f"**SVR & MLP Ensemble Prediction:** {y_pred_ensemble:.3f}")
















