import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Memuat dataset
df = pd.read_csv("SaYoPillow.csv")

# Mengganti nama kolom
df.rename(columns={
    'sr': 'snoring rate',
    'rr': 'respiration rate',
    't': 'body temperature(F)',
    'lm': 'limb movement',
    'bo': 'blood oxygen',
    'rem': 'eye movement',
    'sr.1': 'sleeping hours',
    'hr': 'heart rate',
    'sl': 'stress level'
}, inplace=True)

# Judul dan deskripsi
st.title("Prediksi Tingkat Stres")
st.write("Aplikasi ini memungkinkan Anda untuk memprediksi tingkat stres berdasarkan berbagai fitur fisiologis.")

# Menampilkan dataset
if st.checkbox("Tampilkan data mentah"):
    st.write(df.head())

# Menampilkan distribusi label
st.subheader("Distribusi Label")
label_distribution = df['stress level'].value_counts()
st.bar_chart(label_distribution)

# Menampilkan heatmap korelasi
st.subheader("Heatmap Korelasi Fitur")
correlation = df.drop(columns=['stress level']).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='plasma', ax=ax)
st.pyplot(fig)

# Membagi data
X = df.drop(columns=['stress level'])
y = df['stress level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Menyimpan model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Memuat model
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Prediksi dan evaluasi
y_pred = loaded_model.predict(X_test)
st.subheader("Evaluasi Model")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))
st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

# Formulir prediksi
st.subheader("Prediksi Tingkat Stres")
snoring_rate = st.number_input("Snoring rate", min_value=0.0, step=0.1)
respiration_rate = st.number_input("Respiration rate", min_value=0.0, step=0.1)
body_temperature = st.number_input("Body temperature(F)", min_value=0.0, step=0.1)
limb_movement = st.number_input("Limb movement", min_value=0.0, step=0.1)
blood_oxygen = st.number_input("Blood oxygen", min_value=0.0, step=0.1)
eye_movement = st.number_input("Eye movement", min_value=0.0, step=0.1)
sleeping_hours = st.number_input("Sleeping hours", min_value=0.0, step=0.1)
heart_rate = st.number_input("Heart rate", min_value=0.0, step=0.1)

if st.button("Prediksi"):
    user_input = {
        'snoring rate': snoring_rate,
        'respiration rate': respiration_rate,
        'body temperature(F)': body_temperature,
        'limb movement': limb_movement,
        'blood oxygen': blood_oxygen,
        'eye movement': eye_movement,
        'sleeping hours': sleeping_hours,
        'heart rate': heart_rate,
    }
    input_df = pd.DataFrame([user_input])
    prediction = loaded_model.predict(input_df)
    stress_level_mapping = {
        0: 'rendah/normal',
        1: 'sedang rendah',
        2: 'sedang',
        3: 'sedang tinggi',
        4: 'tinggi'
    }
    prediction_str = stress_level_mapping[prediction[0]]
    st.write(f"Tingkat Stres yang Diprediksi: {prediction_str}")
