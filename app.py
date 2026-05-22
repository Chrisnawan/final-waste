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
MODEL_PATH = Path("fixed_model.h5")
CLASS_PATH = Path("class_names.txt")
ASSET_DIR = Path("gambar")
CSV_DIR = Path("[5] Csv")
IMG_SIZE = 224
# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Perbesar teks tab Streamlit */
div[data-testid="stTabs"] button p {
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Perbesar area klik tab */
div[data-testid="stTabs"] button {
    padding: 18px 28px !important;
}

/* Garis bawah tab aktif */
div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 4px solid #00FFAA !important;
}

</style>
""", unsafe_allow_html=True)
# SESUDAH
@st.cache_resource
def load_model():
    import tensorflow as tf
    from tensorflow.keras.layers import InputLayer

    # Patch kompatibilitas Keras versi lama vs baru
    class CompatInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            kwargs.pop("optional", None)
            if "batch_shape" in kwargs:
                kwargs["input_shape"] = kwargs.pop("batch_shape")[1:]
            super().__init__(*args, **kwargs)

    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        st.stop()

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"InputLayer": CompatInputLayer},
        compile=False
    )
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

    "plastic_soda_bottles": "Plastik",
    "plastic_food_containers": "Plastik",
    "plastic_detergent_bottles": "Plastik",
    "plastic_shopping_bags": "Plastik",
    "plastic_straws": "Plastik",
    "plastic_cup_lids": "Plastik",
    "disposable_plastic_cutlery": "Plastik",

    "glass_beverage_bottles": "Kaca",
    "glass_cosmetic_containers": "Kaca",
    "glass_food_jars": "Kaca",

    "clothing": "Tekstil",
    "shoes": "Tekstil / Karet",

    "styrofoam_cups": "Styrofoam",
    "styrofoam_food_containers": "Styrofoam",

    "paper_cups": "Kertas / Campuran"
}

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
            st.image(
                str(image_path),
                use_container_width=True
            )

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
                    font-size:18px;
                    line-height:1.8;
                    text-align:justify;
                    color:var(--text-color);
                    background-color:transparent;
                    padding-top:10px;
                    margin:0;
                    display:block;
                    width:100%;
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

    img_array = np.array(image).astype(np.float32)

    # Gunakan normalisasi ini jika model dilatih dengan rescale 1./255
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)

    return predicted_class, confidence

# ==========================================
# TITLE
# ==========================================
st.markdown(
    """
    <h1 style="
        font-size:70px;
        font-weight:800;
        margin-bottom:10px;
    ">
        ♻️ Smart Waste Classification Dashboard
    </h1>

  
        <p style="
            font-size:28px;
            margin-top:0px;
            margin-bottom:30px;
        ">
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
# TAB 1: INSIGHT MODEL
# ==========================================
with tab1:

    st.subheader("Hasil Evaluasi Model")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Train Accuracy", "97.92%")
    col2.metric("Validation Accuracy", "94.44%")
    col3.metric("Test Accuracy", "95.11%")

    metric_df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "Accuracy": [97.92, 94.44, 95.11],
    })

    st.dataframe(metric_df, use_container_width=True)

    st.bar_chart(
        metric_df.set_index("Dataset")[["Accuracy"]]
    )

    st.divider()

    st.markdown(
        """
        <h1 style="
            font-size:50px;
            text-align:center;
            margin-top:20px;
            margin-bottom:30px;
        ">
            📊 Visualisasi Model
        </h1>
        """,
        unsafe_allow_html=True
    )
    show_image(
        "confirus matrix.png",
        "Confusion Matrix EfficientNetB0",
        """
        <b></b> Confusion matrix digunakan untuk mengevaluasi performa model klasifikasi dalam membedakan setiap kategori sampah berdasarkan citra yang diberikan. Matriks ini memperlihatkan hubungan antara label asli (actual class) dan hasil prediksi model (predicted class) sehingga dapat memberikan gambaran secara rinci mengenai tingkat keberhasilan maupun kesalahan klasifikasi pada masing-masing kategori. Nilai yang dominan pada diagonal utama menunjukkan bahwa sebagian besar data berhasil diprediksi sesuai dengan kelas aslinya, yang menandakan bahwa model memiliki kemampuan klasifikasi yang baik dalam mengenali pola visual setiap jenis sampah. Semakin tinggi nilai pada diagonal utama, maka semakin tinggi pula tingkat akurasi model dalam melakukan identifikasi kategori sampah. Hasil ini menunjukkan bahwa model mampu mempelajari fitur-fitur penting dari citra, seperti bentuk, warna, tekstur, dan pola objek, sehingga dapat membedakan antar kategori dengan cukup efektif.
        """
    )

    show_image(
        "confidence.png",
        "Confidence Model",
        """
        <b></b> Grafik distribusi confidence digunakan untuk menganalisis tingkat keyakinan (confidence score) model dalam melakukan prediksi terhadap setiap kategori sampah. Grafik ini membandingkan distribusi confidence antara prediksi yang benar dan prediksi yang salah sehingga dapat memberikan gambaran mengenai kualitas keputusan model saat melakukan klasifikasi. Prediksi dengan confidence tinggi menunjukkan bahwa model memiliki tingkat keyakinan yang besar terhadap kelas yang dipilih, sedangkan confidence rendah mengindikasikan bahwa model masih ragu dalam menentukan kategori yang tepat. Pada umumnya, prediksi yang benar cenderung memiliki confidence lebih tinggi dibandingkan prediksi yang salah, yang menunjukkan bahwa model mampu mengenali pola visual data dengan cukup baik.
        """
    )

    show_image(
        "overal model.png",
        "Overall Model Performance",
        """
        <b></b> Visualisasi performa keseluruhan model digunakan untuk merangkum hasil evaluasi model berdasarkan beberapa metrik penting, seperti accuracy, precision, recall, dan F1-score. Grafik ini memberikan gambaran umum mengenai seberapa baik model dalam melakukan klasifikasi jenis sampah berdasarkan citra yang diberikan. Dengan adanya visualisasi tersebut, proses analisis performa model menjadi lebih mudah dipahami karena setiap metrik dapat dibandingkan secara langsung dalam satu tampilan. Nilai metrik yang tinggi menunjukkan bahwa model mampu mengenali dan mengklasifikasikan data dengan baik, baik dari segi ketepatan prediksi maupun kemampuan dalam mengurangi kesalahan klasifikasi antar kategori sampah.
        """
    )

    show_image(
        "training vs validasi akurasi .png",
        "Training vs Validation Accuracy",
        """
        <b></b> Grafik accuracy digunakan untuk memantau perkembangan kemampuan model dalam mempelajari pola data selama proses training berlangsung. Grafik ini biasanya menampilkan perbandingan antara train accuracy dan validation accuracy pada setiap epoch sehingga dapat memberikan gambaran mengenai peningkatan performa model dari waktu ke waktu. Train accuracy menunjukkan tingkat keberhasilan model dalam memprediksi data training, sedangkan validation accuracy menunjukkan kemampuan model dalam mengenali data baru yang tidak digunakan selama proses pelatihan. Peningkatan nilai accuracy secara bertahap menandakan bahwa model berhasil mempelajari karakteristik visual dari setiap kategori sampah dengan lebih baik seiring bertambahnya iterasi training.
        """
    )

    show_image(
        "Training vs Validation  loss.png",
        "Training vs Validation Loss",
        """
        <b></b> Grafik loss digunakan untuk menunjukkan tingkat kesalahan (error) model selama proses training dan validasi berlangsung. Grafik ini umumnya menampilkan train loss dan validation loss pada setiap epoch untuk melihat bagaimana model belajar dari data yang diberikan. Nilai loss yang tinggi menunjukkan bahwa prediksi model masih banyak mengalami kesalahan, sedangkan loss yang semakin menurun menandakan bahwa model mulai mampu mengenali pola data dengan lebih baik. Penurunan loss secara bertahap selama training menunjukkan bahwa proses pembelajaran berjalan dengan optimal dan parameter model berhasil diperbarui untuk meningkatkan performa klasifikasi.
        """
    )

    show_image(
        "Performa perkategori sampah.png",
        "Performa per Kategori Sampah",
        """
        <b></b> Grafik performa per kategori digunakan untuk menunjukkan kemampuan model dalam mengklasifikasikan setiap jenis sampah secara lebih rinci. Visualisasi ini biasanya menampilkan nilai evaluasi seperti precision, recall, atau F1-score pada masing-masing kategori sehingga dapat diketahui kelas mana yang memiliki performa terbaik maupun yang masih mengalami kendala dalam proses klasifikasi. Kategori dengan nilai yang tinggi menunjukkan bahwa model mampu mengenali ciri visual kelas tersebut dengan baik dan menghasilkan prediksi yang akurat. Sebaliknya, kategori dengan nilai evaluasi yang lebih rendah menandakan bahwa model masih mengalami kesulitan dalam membedakan kelas tersebut dari kategori lainnya.
        """
    )

    show_image(
        "perbandingan validasi akurasi .png",
        "Perbandingan Validasi Akurasi",
        """
        <b></b>Pada tahap validasi, model EfficientNetB0 memperoleh akurasi sebesar 94.44%, lebih tinggi dibandingkan ResNet50 yang mencapai 91.78%. Hasil ini menunjukkan bahwa EfficientNetB0 memiliki kemampuan generalisasi yang lebih baik terhadap data validasi dan mampu mempelajari pola citra sampah dengan lebih efektif. Selisih performa sekitar 2.66% mengindikasikan bahwa arsitektur EfficientNetB0 lebih optimal dalam mengekstraksi fitur penting pada dataset klasifikasi sampah.
        """
    )

    
    show_image(
        "perbandingan training akurasi .png",
        "Perbandingan Training Akurasi",
        """
        <b></b>Pada data training, EfficientNetB0 memperoleh akurasi sebesar 97.92%, sedangkan ResNet50 mencapai 95.58%. Nilai ini menunjukkan bahwa kedua model mampu mempelajari pola data dengan sangat baik, namun EfficientNetB0 memiliki kemampuan pembelajaran yang lebih tinggi. Tingginya training accuracy juga mengindikasikan bahwa model berhasil mengenali karakteristik setiap kategori sampah selama proses pelatihan.
        """
    )

    show_image(
        "validasi mae .png",
        "Perbandingan Validation MAE",
        """
        <b></b>Nilai Validation MAE pada EfficientNetB0 sebesar 0.0090, lebih rendah dibandingkan ResNet50 yaitu 0.0129. Semakin kecil nilai MAE menunjukkan bahwa kesalahan prediksi model semakin rendah. Hal ini membuktikan bahwa EfficientNetB0 menghasilkan prediksi yang lebih konsisten dan lebih akurat pada data validasi dibandingkan ResNet50.
        """
    )

    show_image(
        "training mae.png",
        "Perbandingan Training MAE",
        """
        <b></b>Pada tahap training, EfficientNetB0 memperoleh nilai MAE sebesar 0.0053, sedangkan ResNet50 sebesar 0.0102. Perbedaan ini menunjukkan bahwa EfficientNetB0 memiliki tingkat error pelatihan yang jauh lebih kecil. Dengan demikian, model EfficientNetB0 lebih efektif dalam menyesuaikan parameter pembelajaran untuk mengenali pola citra sampah.
        """
    )

    show_image(
        "tes akurasi .png",
        "Perbandingan Test Akurasi",
        """
        <b></b>Pada data pengujian, Pada tahap pengujian akhir, EfficientNetB0 memperoleh akurasi sebesar 95.11%, sedangkan ResNet50 mencapai 92.89%. Hasil ini membuktikan bahwa EfficientNetB0 menjadi model terbaik dalam klasifikasi citra sampah karena mampu memberikan performa paling tinggi pada data testing. Tingginya test accuracy menunjukkan bahwa model dapat diterapkan secara efektif untuk membantu proses pemilahan sampah otomatis dengan tingkat akurasi yang tinggi.
        """
    )

    show_image(
        "tes mae.png",
        "Perbandingan Test MAE",
        """
        <b></b>Pada data pengujian, EfficientNetB0 kembali menunjukkan performa terbaik dengan nilai MAE sebesar 0.0076, lebih rendah dibandingkan ResNet50 sebesar 0.0113. Hasil ini menunjukkan bahwa EfficientNetB0 memiliki kemampuan prediksi yang lebih stabil pada data baru yang belum pernah dilihat sebelumnya. Rendahnya error pada data test juga mengindikasikan bahwa model memiliki generalisasi yang baik dan tidak mengalami overfitting berlebihan.
        """
    )

    show_image(
        "Distribusi Jumlah Gambar per Kelas.png",
        "Distribusi Jumlah Gambar per Kelas",
        """
        <b></b> Grafik distribusi dataset digunakan untuk menunjukkan jumlah gambar pada setiap kategori atau kelas sampah yang digunakan dalam proses training model. Visualisasi ini membantu dalam memahami komposisi data dan melihat apakah jumlah data pada masing-masing kelas sudah seimbang atau masih terdapat perbedaan yang signifikan. Kelas dengan jumlah data yang lebih banyak biasanya akan lebih sering dipelajari oleh model selama proses training sehingga model cenderung lebih mudah mengenali pola pada kategori tersebut. Sebaliknya, kelas dengan jumlah data yang sedikit dapat menyebabkan model kesulitan memahami karakteristik visualnya secara optimal.
        """
    )

    show_image(
        "Contoh Gambar Sampah per Kelas.png",
        "Contoh Gambar Sampah per Kelas",
        """
        <b></b> Visualisasi contoh gambar digunakan untuk membantu memahami karakteristik visual dari setiap kategori sampah yang terdapat dalam dataset. Melalui visualisasi ini, dapat diamati berbagai variasi bentuk, warna, tekstur, ukuran, serta kondisi objek pada masing-masing kelas sampah. Setiap kategori umumnya memiliki ciri visual yang berbeda, seperti sampah plastik yang cenderung memiliki permukaan mengkilap, kertas dengan tekstur tipis dan datar, atau sampah organik dengan bentuk yang lebih tidak beraturan. Variasi tersebut menjadi informasi penting bagi model deep learning dalam mempelajari pola dan fitur visual selama proses training.
        """
    )

    show_image(
        "contoh hasil data augmentasion.png",
        "Contoh Hasil Data Augmentation",
        """
        <b></b> Data augmentation merupakan teknik yang digunakan untuk memperbanyak variasi data training tanpa harus menambah data baru secara manual. Teknik ini dilakukan dengan memodifikasi gambar asli melalui berbagai transformasi, seperti rotasi, flipping, zoom, shifting, cropping, brightness adjustment, dan perubahan sudut pengambilan gambar. Tujuan utama dari data augmentation adalah agar model dapat mempelajari lebih banyak variasi pola visual dari setiap kategori sampah sehingga proses pembelajaran menjadi lebih optimal. Dengan dataset yang lebih bervariasi, model tidak hanya menghafal gambar tertentu, tetapi juga mampu memahami karakteristik umum dari masing-masing kelas sampah.
        """
    )

    show_image(
        "GAMBAR BLUR.png",
        "Analisis Gambar Blur",
        """
        <b></b> Analisis blur digunakan untuk mengevaluasi kualitas gambar dalam dataset sebelum digunakan pada proses training model klasifikasi. Tingkat blur pada gambar dapat memengaruhi kejelasan detail objek, seperti bentuk, tekstur, dan tepi objek yang menjadi fitur penting dalam proses pengenalan citra. Melalui analisis ini, dapat diketahui apakah terdapat gambar dengan kualitas rendah atau tingkat ketajaman yang kurang baik sehingga berpotensi mengganggu proses pembelajaran model. Gambar yang memiliki tingkat blur tinggi biasanya kehilangan banyak informasi visual penting sehingga objek pada gambar menjadi sulit dikenali baik oleh manusia maupun model deep learning.
        """
    )

    show_image(
        "visual brighnes per kelas.png",
        "Visual Brightness per Kelas",
        """
        <b></b> Visualisasi brightness digunakan untuk menunjukkan tingkat kecerahan gambar pada setiap kategori sampah dalam dataset. Analisis ini membantu melihat variasi pencahayaan yang terdapat pada data citra, baik gambar dengan kondisi terlalu gelap, terlalu terang, maupun pencahayaan yang normal. Tingkat brightness pada gambar sangat memengaruhi kejelasan objek, warna, serta detail visual yang menjadi fitur penting dalam proses klasifikasi. Dengan visualisasi ini, dapat diketahui apakah distribusi pencahayaan antar kelas sudah cukup konsisten atau masih terdapat perbedaan signifikan yang berpotensi memengaruhi proses pembelajaran model.
        """
    )

    show_image(
        "visual edge perkelas.png",
        "Visual Edge per Kelas",
        """
        <b></b>Visual edge digunakan untuk menampilkan pola tepi (edge) dari objek pada setiap kategori sampah dalam dataset citra. Proses ini biasanya dilakukan menggunakan teknik deteksi tepi, seperti Canny Edge Detection, untuk menyoroti batas dan struktur utama objek pada gambar. Tepi objek merupakan salah satu fitur visual penting karena dapat menggambarkan bentuk, kontur, serta detail struktur dari suatu objek tanpa terlalu dipengaruhi oleh warna maupun pencahayaan. Melalui visualisasi edge, dapat diamati bagaimana karakteristik bentuk setiap jenis sampah terlihat berdasarkan pola garis dan kontur yang dihasilkan.
        """
    )

    show_image(
        "VISUALISASI SHARPNESS,PER KELAS.png",
        "Visualisasi Sharpness per Kelas",
        """
        <b></b> Analisis sharpness digunakan untuk mengukur tingkat ketajaman gambar pada dataset citra sampah. Ketajaman gambar menunjukkan seberapa jelas detail visual objek, seperti tekstur, garis tepi, bentuk, dan pola permukaan yang terdapat pada gambar. Gambar dengan tingkat sharpness yang tinggi umumnya memiliki detail objek yang lebih terlihat dan tidak kabur, sehingga fitur-fitur visual penting dapat dikenali dengan lebih baik. Visualisasi sharpness membantu dalam mengevaluasi kualitas dataset dan memastikan bahwa gambar yang digunakan memiliki tingkat kejelasan yang cukup untuk mendukung proses pembelajaran model deep learning.
        """
    )


  

# ==========================================
# TAB 2: PREDICTION
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

        trash_type = trash_map.get(
            predicted_class,
            "Kategori belum tersedia"
        )

        st.success(f"Prediksi: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.warning(f"Rekomendasi tong sampah: {trash_type}")

    else:
        st.info("Silakan upload gambar sampah terlebih dahulu untuk melakukan prediksi.")

# ==========================================
# TAB 3: BUSINESS CONCLUSION
# ==========================================
with tab3:

    st.subheader("Jawaban Pertanyaan Bisnis")

   
    show_image(
        "overal model.png",
        "Pertanyaan Bisnis 1: Seberapa baik performa model dalam mengklasifikasikan jenis sampah berdasarkan citra?",
        """

            <b></b> Visualisasi ini digunakan untuk menunjukkan seberapa baik performa model EfficientNetB0 dalam mengklasifikasikan jenis sampah berdasarkan citra menggunakan beberapa metrik evaluasi. Berdasarkan hasil pengujian, model memperoleh accuracy sebesar 95.11%, precision sebesar 95.02%, recall sebesar 94.88%, dan F1-score sebesar 94.95%. Nilai metrik yang tinggi menunjukkan bahwa model memiliki kemampuan yang sangat baik dalam mengenali pola visual setiap kategori sampah serta mampu menghasilkan prediksi yang akurat dan konsisten. Accuracy menunjukkan persentase prediksi benar terhadap seluruh data uji, precision menunjukkan tingkat ketepatan prediksi model, recall menunjukkan kemampuan model dalam mengenali data pada setiap kategori, sedangkan F1-score menunjukkan keseimbangan performa antara precision dan recall. Hasil ini membuktikan bahwa model EfficientNetB0 efektif digunakan untuk membantu proses klasifikasi dan pemilahan sampah berbasis Artificial Intelligence.
        """
    )

    show_image(
        "pertanyaan bisnis 2 sampah paling susah di kategorikan.png",
        "Pertanyaan Bisnis 2: Kategori sampah apa yang paling sulit diklasifikasikan dan apa penyebabnya?",
       """
        <b></b> Visualisasi ini digunakan untuk membantu mengidentifikasi kategori sampah yang paling sulit diklasifikasikan oleh model deep learning berdasarkan tingkat kesalahan prediksi pada setiap kelas. Berdasarkan hasil evaluasi, kategori yang paling sulit diklasifikasikan adalah aluminum_food_cans dengan error rate sebesar 60.53%, steel_food_cans sebesar 60.00%, dan cardboard_packaging sebesar 59.38%. Tingginya error rate pada kategori tersebut menunjukkan bahwa model masih mengalami kesulitan dalam membedakan karakteristik visual tertentu karena adanya kemiripan bentuk objek, warna yang serupa, background gambar yang tidak konsisten, serta perbedaan pencahayaan pada dataset. Analisis ini membantu dalam memahami kelemahan model secara lebih spesifik sehingga dapat digunakan sebagai dasar untuk pengembangan model dan perbaikan kualitas dataset pada tahap selanjutnya.
        """
    )   

    show_image(
        "pertanyaan bisnis3.png",
        "Pertanyaan Bisnis 3:Bagaimana penerapan model ini dapat meningkatkan efisiensi dan akurasi dalam proses pemilahan sampah?",
        """
        <b></b> Visualisasi performa model digunakan untuk menunjukkan kemampuan model EfficientNetB0 dalam mengklasifikasikan jenis sampah berdasarkan citra secara akurat dan konsisten. Berdasarkan hasil evaluasi, model memperoleh accuracy sebesar 95.11%, precision sebesar 95.02%, recall sebesar 94.88%, dan F1-score sebesar 94.95%. Nilai metrik yang tinggi pada seluruh aspek evaluasi menunjukkan bahwa model memiliki kemampuan yang sangat baik dalam mengenali pola visual setiap kategori sampah serta mampu menghasilkan prediksi yang stabil. Accuracy yang tinggi menunjukkan sebagian besar data berhasil diklasifikasikan dengan benar, precision yang tinggi menandakan prediksi model memiliki tingkat ketepatan yang baik, sedangkan recall menunjukkan kemampuan model dalam mengenali hampir seluruh data pada masing-masing kategori. Selain itu, nilai F1-score yang tinggi menunjukkan keseimbangan performa antara precision dan recall sehingga model dinilai efektif untuk diterapkan pada sistem smart waste management berbasis Artificial Intelligence.

        """
        )

  
# ==========================================
# TAB 4: DATA TABULAR
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
