import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Smart Waste Classification Dashboard",
    page_icon="♻️",
    layout="wide"
)

# ==========================================
# PATH CONFIG
# ==========================================
MODEL_PATH = Path("model_weights.h5")
CLASS_PATH = Path("class_names.txt")
ASSET_DIR = Path("gambar")
CSV_DIR = Path("[5] Csv")
IMG_SIZE = 224
NUM_CLASSES = 18

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
div[data-testid="stTabs"] button p {
    font-size: 16px !important;
    font-weight: 600 !important;
}

div[data-testid="stTabs"] button {
    padding: 8px 16px !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid #00FFAA !important;
}

p {
    font-size: 15px !important;
    line-height: 1.6 !important;
}

div[data-testid="stMarkdownContainer"] {
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL (WEIGHTS-ONLY)
# ==========================================
@st.cache_resource
def load_model():
    import keras
    from keras import layers

    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        st.stop()

    try:
        # Bangun ulang arsitektur yang sama seperti saat training
        base_model = keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights=None
        )

        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        # Build model dulu sebelum load weights (wajib di Keras 3.x)
        model.build((None, IMG_SIZE, IMG_SIZE, 3))

        # Load weights dari file .h5
        model.load_weights(str(MODEL_PATH))

        return model

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()


model = load_model()

# ==========================================
# LOAD CLASS NAMES
# ==========================================
if not CLASS_PATH.exists():
    st.error(f"File class_names.txt tidak ditemukan: {CLASS_PATH}")
    st.stop()

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# ==========================================
# TRASH CATEGORY MAP
# ==========================================
trash_map = {
    "eggshells": "Organik",
    "food_waste": "Organik",
    "coffee_grounds": "Organik",
    "tea_bags": "Organik",

    "aerosol_cans": "Anorganik / Logam",
    "aluminum_food_cans": "Anorganik / Logam",
    "aluminum_soda_cans": "Anorganik / Logam",
    "steel_food_cans": "Anorganik / Logam",

    "cardboard_boxes": "Kertas / Kardus",
    "cardboard_packaging": "Kertas / Kardus",
    "magazines": "Kertas",
    "newspaper": "Kertas",
    "office_paper": "Kertas",
    "paper_cups": "Kertas / Campuran",

    "plastic_cup_lids": "Plastik",
    "plastic_detergent_bottles": "Plastik",
    "plastic_food_containers": "Plastik",
    "plastic_shopping_bags": "Plastik",
    "plastic_soda_bottles": "Plastik",
    "plastic_straws": "Plastik",
    "plastic_trash_bags": "Plastik",
    "plastic_water_bottles": "Plastik",
    "disposable_plastic_cutlery": "Plastik",

    "glass_beverage_bottles": "Kaca",
    "glass_cosmetic_containers": "Kaca",
    "glass_food_jars": "Kaca",

    "clothing": "Tekstil",
    "shoes": "Tekstil / Karet",

    "styrofoam_cups": "Styrofoam",
    "styrofoam_food_containers": "Styrofoam"
}

# ==========================================
# HELPER IMAGE
# ==========================================
def show_image(filename, caption, description=""):
    image_path = ASSET_DIR / filename

    st.markdown(
        f"""
        <h2 style="text-align:center; margin-bottom:30px;">
            {caption}
        </h2>
        """,
        unsafe_allow_html=True
    )

    if image_path.exists():
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(str(image_path), use_container_width=True)

        with col2:
            clean_description = (
                description
                .replace("**Insight:**", "")
                .replace("</div>", "")
                .replace("<div>", "")
                .replace("```", "")
                .strip()
            )

            st.markdown(
                f"""
                <p style="
                    font-size:35px;
                    line-height:2;
                    text-align:justify;
                    color:white;
                    padding-top:10px;
                    margin:0;
                ">
                    {clean_description}
                </p>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
    else:
        st.warning(f"Gambar tidak ditemukan: {image_path}")

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction[0]) * 100)

    return predicted_class, confidence

# ==========================================
# TITLE
# ==========================================
st.markdown(
    """
    <h1 style="font-size:70px; font-weight:800; margin-bottom:10px;">
        ♻️ Smart Waste Classification Dashboard
    </h1>

    <p style="font-size:28px; color:#D1D5DB; margin-top:0px; margin-bottom:30px;">
        Dashboard klasifikasi sampah menggunakan model EfficientNetB0 untuk membantu proses identifikasi dan pemilahan sampah.
    </p>
    """,
    unsafe_allow_html=True
)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Insight Model",
    "🖼️ Prediksi Sampah",
    "📝 Kesimpulan Bisnis",
    "📁 Data Tabular"
])

# ==========================================
# TAB 1
# ==========================================
with tab1:
    st.subheader("Hasil Evaluasi Model")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Train Accuracy", "96.53%")
    col2.metric("Validation Accuracy", "93.11%")
    col3.metric("Test Accuracy", "93.33%")
    col4.metric("Test MAE", "0.0083")

    metric_df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "Accuracy": [96.53, 93.11, 93.33],
    })

    st.dataframe(metric_df, use_container_width=True)
    st.bar_chart(metric_df.set_index("Dataset")[["Accuracy"]])

    st.divider()

    st.markdown(
        """
        <h1 style="font-size:50px; text-align:center; margin-top:20px; margin-bottom:30px;">
            📊 Visualisasi Model
        </h1>
        """,
        unsafe_allow_html=True
    )

    image_sections = [
        (
            "confirus matrix.png",
            "Confusion Matrix EfficientNetB0",
            "Confusion matrix digunakan untuk mengevaluasi performa model klasifikasi dalam membedakan setiap kategori sampah berdasarkan citra yang diberikan. Matriks ini memperlihatkan hubungan antara label asli dan hasil prediksi model sehingga dapat memberikan gambaran mengenai tingkat keberhasilan maupun kesalahan klasifikasi pada masing-masing kategori."
        ),
        (
            "confidence.png",
            "Confidence Model",
            "Grafik distribusi confidence digunakan untuk menganalisis tingkat keyakinan model dalam melakukan prediksi terhadap setiap kategori sampah. Prediksi dengan confidence tinggi menunjukkan bahwa model memiliki tingkat keyakinan yang besar terhadap kelas yang dipilih."
        ),
        (
            "overal model.png",
            "Overall Model Performance",
            "Visualisasi performa keseluruhan model digunakan untuk merangkum hasil evaluasi model berdasarkan accuracy, precision, recall, dan F1-score."
        ),
        (
            "training vs validasi akurasi .png",
            "Training vs Validation Accuracy",
            "Grafik accuracy digunakan untuk memantau perkembangan kemampuan model dalam mempelajari pola data selama proses training berlangsung."
        ),
        (
            "Training vs Validation  loss.png",
            "Training vs Validation Loss",
            "Grafik loss menunjukkan tingkat kesalahan model selama proses training dan validation."
        ),
        (
            "Performa perkategori sampah.png",
            "Performa per Kategori Sampah",
            "Grafik performa per kategori menunjukkan kemampuan model dalam mengklasifikasikan setiap jenis sampah secara lebih rinci."
        ),
        (
            "perbandingan validasi akurasi .png",
            "Perbandingan Validasi Akurasi",
            "Pada tahap validasi, model EfficientNetB0 memperoleh akurasi sebesar 93.11%, lebih tinggi dibandingkan ResNet50 yang mencapai 91.78% ."
        (
            "perbandingan training akurasi .png",
            "Perbandingan Training Akurasi",
            "Pada data training, EfficientNetB0 memperoleh akurasi sebesar 96.53%, sedangkan ResNet50 mencapai 95.58%."
        ),
        (
            "validasi mae .png",
            "Perbandingan Validation MAE",
            "Nilai Validation MAE pada EfficientNetB0 sebesar 0.0091, lebih rendah dibandingkan ResNet50 yaitu 0.0129."
        ),
        (
            "training mae.png",
            "Perbandingan Training MAE",
            "Pada tahap training, EfficientNetB0 memperoleh nilai MAE sebesar 0.0064, sedangkan ResNet50 sebesar 0.0102."
        ),
        (
            "tes akurasi .png",
            "Perbandingan Test Akurasi",
            "Pada tahap pengujian akhir, EfficientNetB0 memperoleh akurasi sebesar 93.33%, sedangkan ResNet50 mencapai 92.89%."
        ),
        (
            "tes mae.png",
            "Perbandingan Test MAE",
            "Pada data pengujian, EfficientNetB0 kembali menunjukkan performa terbaik dengan nilai MAE sebesar 0.0086, lebih rendah dibandingkan ResNet50 sebesar 0.0113."
        ),
        (
            "Distribusi Jumlah Gambar per Kelas.png",
            "Distribusi Jumlah Gambar per Kelas",
            "Grafik distribusi dataset menunjukkan jumlah gambar pada setiap kategori sampah yang digunakan dalam proses training model."
        ),
        (
            "Contoh Gambar Sampah per Kelas.png",
            "Contoh Gambar Sampah per Kelas",
            "Visualisasi contoh gambar membantu memahami karakteristik visual dari setiap kategori sampah."
        ),
        (
            "contoh hasil data augmentasion.png",
            "Contoh Hasil Data Augmentation",
            "Data augmentation digunakan untuk memperbanyak variasi data training agar model lebih general dan tidak hanya menghafal gambar tertentu."
        ),
        (
            "GAMBAR BLUR.png",
            "Analisis Gambar Blur",
            "Analisis blur digunakan untuk mengevaluasi kualitas gambar dalam dataset sebelum digunakan pada proses training model klasifikasi."
        ),
        (
            "visual brighnes per kelas.png",
            "Visual Brightness per Kelas",
            "Visualisasi brightness menunjukkan tingkat kecerahan gambar pada setiap kategori sampah dalam dataset."
        ),
        (
            "visual edge perkelas.png",
            "Visual Edge per Kelas",
            "Visual edge digunakan untuk menampilkan pola tepi objek pada setiap kategori sampah."
        ),
        (
            "VISUALISASI SHARPNESS,PER KELAS.png",
            "Visualisasi Sharpness per Kelas",
            "Analisis sharpness digunakan untuk mengukur tingkat ketajaman gambar dalam dataset citra sampah."
        )
    ]

    for filename, caption, description in image_sections:
        show_image(filename, caption, description)

# ==========================================
# TAB 2
# ==========================================
with tab2:
    st.subheader("Prediksi Jenis Sampah")

    uploaded_file = st.file_uploader(
        "Upload gambar sampah",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Gambar yang diupload",
            use_container_width=True
        )

        with st.spinner("Memprediksi gambar..."):
            predicted_class, confidence = predict_image(image)

        trash_type = trash_map.get(predicted_class, "Kategori belum tersedia")

        st.success(f"Prediksi: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.warning(f"Rekomendasi tong sampah: {trash_type}")

    else:
        st.info("Silakan upload gambar sampah terlebih dahulu untuk melakukan prediksi.")

# ==========================================
# TAB 3
# ==========================================
with tab3:
    st.subheader("Jawaban Pertanyaan Bisnis")

    show_image(
        "overal model.png",
        "Pertanyaan Bisnis 1: Seberapa baik performa model dalam mengklasifikasikan jenis sampah berdasarkan citra?",
        """
        Visualisasi ini menunjukkan performa model EfficientNetB0 dalam mengklasifikasikan jenis sampah berdasarkan citra. Berdasarkan hasil pengujian, model memperoleh accuracy sebesar 95.11%, precision sebesar 95.02%, recall sebesar 94.88%, dan F1-score sebesar 94.95%. Nilai metrik yang tinggi menunjukkan bahwa model memiliki kemampuan yang baik dalam mengenali pola visual setiap kategori sampah serta menghasilkan prediksi yang akurat dan konsisten.
        """
    )

    show_image(
        "pertanyaan bisnis 2 sampah paling susah di kategorikan.png",
        "Pertanyaan Bisnis 2: Kategori sampah apa yang paling sulit diklasifikasikan dan apa penyebabnya?",
        """
        Visualisasi ini digunakan untuk mengidentifikasi kategori sampah yang paling sulit diklasifikasikan oleh model berdasarkan tingkat kesalahan prediksi. Berdasarkan hasil evaluasi, kategori yang paling sulit diklasifikasikan adalah aluminum_food_cans dengan error rate 60.53%, steel_food_cans sebesar 60.00%, dan cardboard_packaging sebesar 59.38%. Penyebabnya adalah kemiripan bentuk objek, warna yang serupa, background gambar yang tidak konsisten, serta perbedaan pencahayaan.
        """
    )

    show_image(
        "pertanyaan bisnis3.png",
        "Pertanyaan Bisnis 3: Bagaimana penerapan model ini dapat meningkatkan efisiensi dan akurasi dalam proses pemilahan sampah?",
        """
        Model dapat membantu proses pemilahan sampah dengan mempercepat identifikasi jenis sampah, mengurangi human error, meningkatkan konsistensi klasifikasi, dan memberikan rekomendasi kategori tong sampah secara otomatis. Dengan performa model yang baik, sistem ini dapat mendukung penerapan smart waste management berbasis Artificial Intelligence.
        """
    )

# ==========================================
# TAB 4
# ==========================================
with tab4:
    st.markdown(
        """
        <h1 style='text-align:center; font-size:50px;'>
            📁 Data Tabular Dataset
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 📂 Daftar File CSV")

    csv_files = {
        "raw_image_data.csv": CSV_DIR / "raw_image_data.csv",
        "assessment_image_quality.csv": CSV_DIR / "assessment_image_quality.csv",
        "image_hash_check.csv": CSV_DIR / "image_hash_check.csv",
        "duplicate_images.csv": CSV_DIR / "duplicate_images.csv",
        "clean_image_data.csv": CSV_DIR / "clean_image_data.csv",
        "data_dictionary.csv": CSV_DIR / "data_dictionary.csv",
        "eda_class_distribution.csv": CSV_DIR / "eda_class_distribution.csv",
        "split_dataset_summary.csv": CSV_DIR / "split_dataset_summary.csv",
        "data_leakage_prevention.csv": CSV_DIR / "data_leakage_prevention.csv",
        "evaluation_stage1_before_finetuning.csv": CSV_DIR / "evaluation_stage1_before_finetuning.csv",
        "final_model_evaluation_after_finetuning.csv": CSV_DIR / "final_model_evaluation_after_finetuning.csv",
        "business_questions_analysis.csv": CSV_DIR / "business_questions_analysis.csv",
    }

    selected_csv = None

    col1, col2 = st.columns(2)

    for index, (file_name, file_path) in enumerate(csv_files.items()):
        target_col = col1 if index % 2 == 0 else col2

        with target_col:
            if st.button(f"📄 {file_name}", use_container_width=True):
                st.session_state["selected_csv"] = file_name

    if "selected_csv" in st.session_state:
        selected_csv = st.session_state["selected_csv"]

    if selected_csv:
        selected_path = csv_files[selected_csv]

        st.divider()

        st.markdown(
            f"""
            <h2 style='margin-top:20px;'>
                📄 Membuka File: {selected_csv}
            </h2>
            """,
            unsafe_allow_html=True
        )

        try:
            df = pd.read_csv(selected_path)

            col1, col2, col3 = st.columns(3)

            col1.metric("Jumlah Baris", df.shape[0])
            col2.metric("Jumlah Kolom", df.shape[1])
            col3.metric("Missing Values", int(df.isnull().sum().sum()))

            st.dataframe(
                df,
                use_container_width=True,
                height=700
            )

            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=selected_csv,
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
