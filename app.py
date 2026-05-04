# app.py
import streamlit as st
import pandas as pd
import subprocess
import os
import joblib
from gensim.models import Word2Vec
from preprocessing_word2vec_svm import preprocess_user_input, load_normalization_assets
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Tes Big Five Personality", layout="centered")

@st.cache_resource
def load_all_models():
    w2v = Word2Vec.load("models/word2vec_b5.model")
    svm_model = joblib.load("models/svm_model.pkl")
    mlb = joblib.load("models/mlb.pkl")
    return w2v, svm_model, mlb

kamus_alay, kbbi_words, kamus_inggris, badwords, noise_words = load_normalization_assets()
w2v_model, svm_model, mlb = load_all_models()

def run_analysis(combined_text):
    st.markdown(f"""
        <div style='background-color:#e3f2fd;padding:20px;border-radius:12px; 
                    font-size:16px;'>
            <b>Gabungan Teks untuk Analisis</b><br><br>{combined_text}
        </div>""", unsafe_allow_html=True)

    result = preprocess_user_input(
        combined_text,
        model=w2v_model,
        threshold=0.3,
        kamus_alay=kamus_alay,
        kbbi_words=kbbi_words,
        kamus_inggris=kamus_inggris,
        badwords=badwords,
        noise_words=noise_words,
        svm_model=svm_model,
        mlb=mlb)

    if result:
        st.markdown("#### 📌 Hasil Preprocessing")
        st.code(result["stemmed_b5"], language="python")

        if "prob_svm" in result:
            st.markdown("#### Hasil Prediksi SVM")

            tab_svm_table, tab_svm_chart = st.tabs(["📋 Tabel SVM", "📈 Bar Chart SVM"])

            with tab_svm_table:
                df_prob = pd.DataFrame({
                    "Label": result["mlb_classes"],
                    "SVM (%)": [f"{p * 100:.2f}%" for p in result["prob_svm"]],
                })
                st.markdown("###### 🔍 Persentase Prediksi SVM")
                st.dataframe(df_prob, use_container_width=True)

            with tab_svm_chart:
                df_bar = pd.DataFrame({
                    "Label": result["mlb_classes"],
                    "SVM": result["prob_svm"] * 100
                })
                fig_bar = px.bar(df_bar, x="SVM", y="Label", color="Label", orientation="h", title="SVM Prediction Score")
                st.plotly_chart(fig_bar, use_container_width=True)

            top_labels = sorted(zip(result["mlb_classes"], result["prob_svm"]), key=lambda x: -x[1])[:3]

            st.markdown(f"""
            <div style='background-color:#dcedc8;padding:15px;border-radius:10px;
                        margin-top:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1);font-size:17px;'>
                <b>Top 3 Kepribadian Berdasarkan Prediksi SVM:</b><br>
                <span style='font-size:17px;color:#33691e;font-weight:bold;'>{", ".join([lbl for lbl, _ in top_labels])}</span>
            </div>
            """, unsafe_allow_html=True)


            st.markdown("#### 📝 Feedback Kepribadian")
            for label in [lbl for lbl, _ in top_labels]:
                if label == "Neuroticism":
                    st.info("""🔴 **Neuroticism**  
                    Kamu termasuk orang yang cukup sensitif dan mungkin cenderung overthinking. Kadang-kadang perasaan cemas atau stres bisa muncul tiba-tiba. Tapi itu artinya kamu punya kepekaan yang tinggi terhadap sekitar.  
                    Coba luangkan waktu untuk hal-hal yang bikin kamu tenang, seperti nulis jurnal, olahraga ringan, atau sekadar dengerin musik favorit. Nggak apa-apa kok merasa lelah—yang penting kamu sadar dan mau jaga dirimu sendiri.""")
                elif label == "Conscientiousness":
                    st.info("""🟢 **Conscientiousness**  
                    Kamu punya sikap yang teratur dan bisa diandalkan. Kalau dikasih tanggung jawab, kamu biasanya bakal kerjain dengan serius. Orang lain mungkin melihatmu sebagai pribadi yang disiplin dan fokus.  
                    Sifat ini sangat bagus, apalagi dalam hal kerja atau belajar. Tapi jangan lupa kasih waktu juga buat diri sendiri untuk santai dan menikmati proses. Nggak semua hal harus sempurna kok.""")
                elif label == "Agreeableness":
                    st.info("""🟣 **Agreeableness**  
                    Kamu adalah tipe orang yang peduli sama orang lain dan nggak suka konflik. Dalam pertemanan atau kerja tim, kamu biasanya jadi penengah dan suka bantuin tanpa diminta.  
                    Orang-orang pasti nyaman dekat kamu. Tapi jangan lupa juga untuk jaga batasan, ya. Setiap orang punya hak buat bilang 'nggak' kalau memang merasa nggak nyaman.""")
                elif label == "Openness":
                    st.info("""🟡 **Openness**  
                    Kamu suka hal-hal baru dan punya rasa ingin tahu yang besar. Ide-ide kreatif sering banget muncul di kepala kamu, dan kamu nggak takut buat coba sesuatu yang beda.  
                    Terus pertahankan semangat eksplorasi ini, karena itu salah satu kelebihan kamu. Tapi juga penting untuk tahu kapan harus fokus dan menyelesaikan satu hal sebelum pindah ke hal lain.""")
                elif label == "Extraversion":
                    st.info("""🔵 **Extraversion**  
                    Kamu suka berinteraksi, ngobrol, dan biasanya nyaman di keramaian. Energi kamu seolah tumbuh saat ketemu orang baru atau kumpul bareng teman.  
                    Ini hal yang keren karena kamu bisa bawa semangat ke lingkungan sekitar. Tapi ingat juga, nggak apa-apa kalau sesekali ingin menyendiri—itu bagian dari jaga keseimbangan diri juga.""")

            st.markdown("""
            <hr style="margin-top:40px; margin-bottom:10px;">

            <div style='font-size:14px; color:#666; padding:10px; background-color:#f9f9f9; border-radius:8px;'>
            ⚠️ <b>Catatan Penting:</b><br>
            Hasil analisis kepribadian ini <i>bukan diagnosis profesional</i>. Sistem hanya memproses teks dan memberikan estimasi kecenderungan berdasarkan model analisis bahasa.  
            Jika kamu ingin memahami kepribadian secara lebih mendalam, sebaiknya berkonsultasilah dengan psikolog atau profesional di bidang terkait.
            </div>
            """, unsafe_allow_html=True)
                    
st.title("🧠 Tes Big Five Personality")

st.write("Tes ini mengukur lima dimensi utama dalam kepribadian Anda. Big Five Personality adalah model lima faktor kepribadian yang dirumuskan oleh Goldberg berdasarkan penelitian ilmiah di bidang psikologi kepribadian, yang bertujuan untuk memahami dan mengukur dimensi dasar kepribadian manusia. " \
"Berikut penjelasan singkatnya:")
with st.expander("Openness"): 
    st.write("Kepribadian Openness mencerminkan keinginan seseorang untuk mencoba hal baru dan terlibat dalam kegiatan kreatif serta intelektual. Individu dengan Openness atau keterbukaan yang tinggi cenderung inovatif, berpikir di luar kebiasaan, dan fleksibel. ")
with st.expander("Conscientiousness"): 
    st.write("Sifat kepribadian ini mencerminkan kemampuan individu untuk mengontrol dorongan atau keinginan sesaat demi mencapai tujuan. Individu dengan sifat conscientiousness cenderung terorganisir, bertanggung jawab, pekerja keras, dan mematuhi aturan, sehingga individu yang memiliki sifat ini kompeten, disiplin, dan fokus pada pencapaian. ")
with st.expander("Extraversion"): 
    st.write("Kepribadian extraversion mencerminkan keinginan individu untuk terlibat dalam interaksi sosial. Individu dengan tingkat ekstroversi tinggi sanfat terbuka, energik, dan senang menjadi pusat perhatian.")
with st.expander("Agreeableness"): 
    st.write("Sifat agreeableness berhubungan dengan cara soerang individu memperlakukan orang lain dalam hubungan soial. Individu dengan  agreeableness yang tinggi biasanya jujur, rendah hati, simpatik, dan pemaaf. ")
with st.expander("Neuroticism"): 
    st.write("Kepribadian neuroticsm mengukur stabilitas emosional individu dan bagaimana individu tersebut bereaksi terhadao berbagai peristiwa. Individu dengan neuroticism tinggi cenderung cemas, mudah tersinggung, dan mengalami perubahan suasana hati yang tajam.")

metode = st.radio("Pilih metode input:", ("Input Teks Manual", "Input Username Twitter"))

if metode == "Input Teks Manual":
    st.markdown("""<p style='background:#e3f2fd;padding:10px;border-radius:5px;'>Isi minimal 10 kalimat untuk analisis</p>""", unsafe_allow_html=True)
    if 'num_inputs' not in st.session_state:
        st.session_state.num_inputs = 10
    input_texts = [st.text_input(f"Kalimat {i+1}", key=f"kalimat_{i}") for i in range(st.session_state.num_inputs)]
    if st.button("➕ Tambah Kalimat"):
        st.session_state.num_inputs += 1
    if st.button("🔍 Analisis Teks"):
        valid_texts = [txt.strip() for txt in input_texts if txt.strip() != ""]
        if len(valid_texts) < 5:
            st.warning("Minimal 5 kalimat harus diisi.")
        else:
            combined_text = " ".join(valid_texts)
            st.success(f"{len(valid_texts)} kalimat akan diproses untuk analisis.")
            run_analysis(combined_text)

elif metode == "Input Username Twitter":
    st.markdown("""<p style='background:#f3e5f5;padding:10px;border-radius:5px;'>Masukkan username Twitter dan token autentikasi Anda</p>""", unsafe_allow_html=True)
    username = st.text_input("Username Twitter (tanpa @)", key="username_input")
    twitter_token = st.text_input("Token Auth Twitter", type="password", key="token_input")

    if st.button("🙇🏻‍♂️ Ambil Data dan Analisis Tweet"):
        if not username.strip() or not twitter_token.strip():
            st.warning("Mohon isi username dan token terlebih dahulu.")
            st.stop()

        tweet_file_path = os.path.abspath(os.path.join("tweets-data", f"{username}.csv"))
        script_path = os.path.join(os.getcwd(), "run_harvest.py")

        st.info("📡 Mengambil tweet secara real-time...")
        try:
            log_area = st.empty()
            process = subprocess.Popen(
                ["python", script_path, username, twitter_token],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            logs = ""
            file_ready = False
            max_wait = 120
            wait_time = 0
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    logs += output
                    log_area.code(logs, language="bash")
                if not file_ready and os.path.exists(tweet_file_path):
                    file_ready = True
                wait_time += 1
                if wait_time > max_wait:
                    break
            return_code = process.poll()

            if not os.path.exists(tweet_file_path):
                st.error("❌ File hasil crawling tidak ditemukan.")
                st.stop()
            if return_code != 0:
                st.warning("⚠️ Proses tweet-harvest error, namun file ditemukan. Melanjutkan ke preprocessing...")
            else:
                st.success("✅ Tweet berhasil diambil dan file ditemukan.")

            df_tweet = pd.read_csv(tweet_file_path)
            if "full_text" not in df_tweet.columns:
                st.error("Kolom 'full_text' tidak ditemukan dalam file tweet.")
                st.write("Kolom yang tersedia:", df_tweet.columns.tolist())
                st.stop()
            tweet_list = df_tweet["full_text"].dropna().astype(str).tolist()
            if not tweet_list:
                st.warning("❗ Tidak ada tweet yang ditemukan untuk dianalisis.")
                st.stop()
            combined_text = " ".join(tweet_list)
            st.success(f"{len(tweet_list)} tweet berhasil diambil dan akan dianalisis.")
            run_analysis(combined_text)
        except Exception as e:
            st.error("❌ Gagal mengambil tweet. Coba periksa token dan koneksi.")
            st.exception(e)
            st.stop()