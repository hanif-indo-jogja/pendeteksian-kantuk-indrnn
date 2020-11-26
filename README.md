Kode untuk pengambilan dan preprocessing fitur kedipan dibuat berdasarkan kode dari https://github.com/rezaghoddoosian/Early-Drowsiness-Detection

Kode untuk training IndRNN dibuat berdasarkan kode dari https://github.com/Sunnydreamrain/IndRNN_pytorch

### Drowsiness Dataset
Dataset yang digunakan adalah [UTA Real-Life Drowsiness Dataset](https://sites.google.com/view/utarldd/home).

### Cara Menjalankan Kode
Kode ini sudah dicoba pada Ubuntu 18.04.

1. Sebelum dijalankan, ekstrak UTA Real-Life Drowsiness Dataset dan taruh ke dalam folder drowsiness_dataset.

2. Jalankan script `setup.sh` untuk menginstall dependencies.

3. Jalankan script `python3 blink_feature_extractor_ui.py` untuk mengambil fitur kedipan dari drowsiness dataset.

4. Jalankan script `python3 preprocessing.py` untuk preprocessing fitur kedipan.

5. Jalankan script `run_train.sh` untuk training dengan data fitur kedipan sebelumnya.

6. Jalankan script `python3 drowsiness_detector_ui.py` untuk menjalankan aplikasi pendeteksian kantuk dengan menggunakan model IndRNN dari hasil training sebelumnya.