from fastapi import FastAPI, Query
from typing import Optional, Dict
import pandas as pd
import numpy as np
import logging
import pickle
import joblib
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

from googletrans import Translator

app = FastAPI()
translator = Translator()

logging.basicConfig(
    filename="api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with open("woofi_model.pkl", "rb") as f:
    tfidf, cosine_sim, wisata_df = pickle.load(f)

kategori_columns = [col for col in wisata_df.columns if col.startswith("Kategori_")]
if kategori_columns:
    kategori_series = wisata_df[kategori_columns].idxmax(axis=1)
    wisata_df["Kategori"] = kategori_series.str.replace("Kategori_", "").str.strip()
else:
    raise ValueError("Kolom kategori tidak ditemukan dalam wisata_df")

if "Provinsi" not in wisata_df.columns:
    wisata_df["Provinsi"] = "Tidak diketahui"
else:
    wisata_df["Provinsi"] = wisata_df["Provinsi"].fillna("Tidak diketahui").astype(str).str.strip()

model = joblib.load("rekomendasi_wisata_model.pkl")
ohe = joblib.load("rekomendasi_encoder_kategori.pkl")
mlb = joblib.load("rekomendasi_encoder_interest.pkl")

interest_to_kategori = {
    "Mountain": "Gunung/Bukit",
    "Beach": "Pantai",
    "Lake": "Danau",
    "Waterfall": "Air Terjun",
    "Forest": "Hutan",
    "Museum": "Museum",
    "Tourist Village": "Desa Wisata",
    "Recreational Park": "Taman",
    "Peak": "Puncak",
    "Others": "Lainnya"
}

def map_interest_to_kategori(interest_list):
    return [interest_to_kategori.get(i, "Lainnya") for i in interest_list]

kategori_to_interest = {v: k for k, v in interest_to_kategori.items()}

def calculate_age(birth_date_str):
    birth_date = datetime.fromisoformat(birth_date_str.replace('Z', '+00:00'))
    today = datetime.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def recommend_for_new_user(gender, age, interest_list, top_n=None):
    valid_interest = [i if i in mlb.classes_ else "Others" for i in interest_list]
    interest_data = mlb.transform([valid_interest])
    interest_df = pd.DataFrame(interest_data, columns=[f"interest_{c}" for c in mlb.classes_])

    main_kategori = interest_to_kategori.get(valid_interest[0], "Lainnya")
    dummy_kategori = pd.DataFrame(ohe.transform([[main_kategori]]), columns=ohe.get_feature_names_out(["kategori"]))

    input_df = pd.concat([
        interest_df,
        dummy_kategori,
        pd.DataFrame([[gender, age, 1]], columns=["gender", "age", "count"])
    ], axis=1)

    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    proba = model.predict_proba(input_df)[0]
    top_indices = proba.argsort()[::-1][:top_n] if top_n else proba.argsort()[::-1]
    labels = model.classes_[top_indices]

    rekomendasi_model = wisata_df[wisata_df["NameLocation"].isin(labels)].copy()
    rekomendasi_model["skor_model"] = rekomendasi_model["NameLocation"].apply(
        lambda x: proba[labels.tolist().index(x)] if x in labels else 0
    )

    interest_query = " ".join(valid_interest)
    tfidf_query = tfidf.transform([interest_query])
    similarity_scores = cosine_sim.dot(tfidf_query.T).toarray().ravel()

    df_content = wisata_df.copy()
    df_content["score_content"] = similarity_scores
    df_content = df_content[df_content["score_content"] > 0]
    rekomendasi_content = df_content.sort_values(by="score_content", ascending=False)
    if top_n:
        rekomendasi_content = rekomendasi_content.head(top_n)

    hasil_supervised = rekomendasi_model[["NameLocation", "Kategori", "skor_model"]]
    hasil_content = rekomendasi_content[["NameLocation", "Kategori", "score_content"]]

    hasil_gabungan = pd.merge(hasil_supervised, hasil_content, on=["NameLocation", "Kategori"], how="outer")
    hasil_gabungan["skor_total"] = hasil_gabungan[["skor_model", "score_content"]].sum(axis=1, skipna=True)
    hasil_gabungan = hasil_gabungan.replace([np.inf, -np.inf], 0).fillna(0)

    hasil_final = hasil_gabungan.merge(
        wisata_df[["NameLocation", "LinkGmaps", "Rating", "Provinsi", "Foto", "Alamat", "Penjelasan_English"]],
        on="NameLocation", how="left"
    )

    hasil_final["interest"] = hasil_final["Kategori"].map(kategori_to_interest).fillna("Others")
    hasil_final = hasil_final.replace([np.inf, -np.inf], 0).fillna("")
    if top_n:
        hasil_final = hasil_final.head(top_n)

    return hasil_final[[
        "NameLocation", "LinkGmaps", "Rating", "Provinsi", "Foto", "Alamat", "Penjelasan_English", "interest"
    ]].to_dict(orient="records")

def recommend_for_existing_user(user_data, wisata_df, top_n=None):
    user_interest = map_interest_to_kategori(user_data.get("interest", []))
    searchs = user_data.get("searchs", [])
    searched_places = [s["name"] for s in searchs if isinstance(s, dict) and "name" in s]

    rekomendasi = wisata_df[
        wisata_df["Kategori"].isin(user_interest) &
        ~wisata_df["NameLocation"].isin(searched_places)
    ].copy()

    rekomendasi["interest"] = rekomendasi["Kategori"].map(kategori_to_interest).fillna("Others")
    rekomendasi = rekomendasi.fillna("")

    kolom_wajib = ["Rating", "Foto", "Provinsi", "Alamat", "Penjelasan_English", "LinkGmaps"]
    for col in kolom_wajib:
        if col not in rekomendasi.columns:
            rekomendasi[col] = ""

    if top_n:
        rekomendasi = rekomendasi.head(top_n)

    kolom_output = [
        "NameLocation", "LinkGmaps", "Rating", "Provinsi", "Foto", "Alamat", "Penjelasan_English", "interest"
    ]
    return rekomendasi[kolom_output].drop_duplicates().to_dict(orient="records")

@app.get("/rekomendasi")
def get_rekomendasi(
    q: str = Query(" ", description="Masukkan kata kunci pencarian (dalam bahasa apa saja)"),
    kategori: Optional[str] = Query(None, description="Nama kategori (misal: Gunung, Pantai)"),
    provinsi: Optional[str] = Query(None, description="Nama provinsi (misal: Bali, Sumatera Selatan)"),
    top_k: Optional[int] = Query(None, description="Jumlah maksimal hasil (kosongkan untuk tampil semua)")
):
    logging.info(f"Rekomendasi request | q='{q}' | kategori='{kategori}' | provinsi='{provinsi}'")
    try:
        translated_query = translator.translate(q, dest='en').text
    except Exception as e:
        logging.warning(f"Gagal menerjemahkan query '{q}', error: {e}")
        translated_query = q

    tfidf_query = tfidf.transform([translated_query])
    similarity_scores = cosine_sim.dot(tfidf_query.T).toarray().ravel()
    df = wisata_df.copy()
    df["score"] = similarity_scores

    if kategori:
        kolom_kategori = [col for col in df.columns if kategori.lower().strip() in col.lower() and "Kategori_" in col]
        if kolom_kategori:
            df = df[df[kolom_kategori[0]] == 1]

    if provinsi:
        df = df[df["Provinsi"].str.lower().str.strip() == provinsi.lower().strip()]

    df["interest"] = df["Kategori"].map(kategori_to_interest).fillna("Others")
    hasil = df.sort_values(by="score", ascending=False)
    if top_k:
        hasil = hasil.head(top_k)

    return {
        "rekomendasi": hasil[[
            "NameLocation", "LinkGmaps", "Rating", "Provinsi", "Foto", "Alamat", "Penjelasan_English", "interest"
        ]].to_dict(orient="records")
    }

@app.post("/rekomendasi-user")
def get_rekomendasi_user(data: Dict):
    try:
        logging.info(f"Request rekomendasi-user: {data}")
        if data.get("is_new_user", True):
            hasil = recommend_for_new_user(
                gender=data["gender"],
                age=data["age"],
                interest_list=data["interest"],
                top_n=data.get("top_n")
            )
            return {"tipe": "baru", "rekomendasi": hasil}
        else:
            hasil = recommend_for_existing_user(data, wisata_df, top_n=data.get("top_n"))
            return {"tipe": "lama", "rekomendasi": hasil}
    except Exception as e:
        logging.error(f"Error di endpoint rekomendasi-user: {str(e)}")
        return {"error": str(e)}
