import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise import dump

# Fungsi untuk memuat data
@st.cache
def load_data():
    data = pd.read_csv('processed_data.csv')
    return data

# Fungsi untuk membuat model rekomendasi
@st.cache(allow_output_mutation=True)
def train_model(data):
    reader = Reader(rating_scale=(0, 1))  # Sesuaikan skala rating
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    trainset = dataset.build_full_trainset()
    
    algo = SVD()
    algo.fit(trainset)
    return algo

# Memuat data
data = load_data()

# Melatih model
model = train_model(data)

# Streamlit UI
st.title("Sistem Rekomendasi Musik Film")
st.write("Masukkan ID Pengguna dan ID Film untuk mendapatkan rekomendasi rating.")

# Input dari pengguna
user_id = st.text_input("Masukkan ID Pengguna")
item_id = st.text_input("Masukkan ID Film")

if st.button("Prediksi"):
    if user_id and item_id:
        pred = model.predict(user_id, item_id)
        st.write(f"Prediksi rating untuk Pengguna {user_id} dan Film {item_id} adalah {pred.est:.2f}")
    else:
        st.write("Silakan masukkan ID Pengguna dan ID Film.")

# Menyimpan model yang telah dilatih
dump.dump('model_file', algo=model)

st.write("Model telah dilatih dan disimpan.")
