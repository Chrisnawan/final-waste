from tensorflow import keras

# load model lama
model = keras.models.load_model(
    "best_finetune_model.keras",
    compile=False
)

# simpan ulang TANPA optimizer/training config
model.save(
    "fixed_model.h5",
    include_optimizer=False
)

print("Model berhasil disimpan ulang!")