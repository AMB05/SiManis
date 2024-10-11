from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os
import torch

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True).to(device)

classes_dict = {
    # Bahan Daging Sapi
    "Daging-Sapi": {
        "Gadon Daging Sapi": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Daging sapi cincang: 250 gram",
                "Santan kental: 100 ml",
                "Telur ayam: 1 butir",
                "Bawang putih, haluskan: 3 siung",
                "Bawang merah, haluskan: 4 siung",
                "Daun salam: 2 lembar",
                "Serai, memarkan: 1 batang",
                "Lengkuas, memarkan: 2 cm",
                "Daun pisang untuk membungkus: secukupnya",
                "Lidi untuk menyemat: secukupnya",
                "Porsi yang dihasilkan = 5 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Campurkan daging sapi cincang dengan santan, telur, bawang putih, bawang merah, garam, gula, dan merica. Aduk rata.",
                "Ambil daun pisang, letakkan campuran daging di atasnya, bungkus rapat, dan sematkan dengan lidi.",
                "Kukus gadon selama 40 menit hingga matang.",
                "Angkat, dinginkan sejenak, dan gadon siap disajikan."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "13% AKG",
                    "Detail": "Cincang gadon, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan, ¼ sdt minyak sehat, dan kaldu secukupnya.",
                    "Kandungan Protein": "17 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Cincang gadon, campur dengan 5 sdm nasi tim, 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang, ¼ sdt minyak sehat, dan kaldu secukupnya.",
                    "Kandungan Protein": "17 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "16% AKG",
                    "Detail": "Potong sedang gadon, tambahkan nasi 100 gram, 1 mangkok kecil sayuran, 2 sdm kacang-kacangan, ½ sdt minyak sehat. Tambahkan (per resep) 1 sdt garam dan1/2 sdt gula pasir.",
                    "Kandungan Protein": "17 Gram"
                }
            ]
        },

        "Semur Bola Daging": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "BAHAN BOLA DAGING:",
                "300 gram daging sapi giling",
                "1 butir telur (ukuran besar)",
                "2 sendok makan remah roti (bread crumbs)",
                "2 siung bawang putih, cincang halus",
                "1/4 cangkir bawang bombay, cincang halus",
                "●●●●●",
                "BAHAN SAUS SEMUR:",
                "2 sendok makan minyak goreng",
                "1/2 cangkir bawang bombay, cincang kasar",
                "3 siung bawang putih, cincang kasar",
                "2 buah tomat, potong-potong",
                "1 sendok makan kecap manis",
                "½ sendok makan kecap asin",
                "1 sendok teh gula merah, serut",
                "½ sendok teh ketumbar bubuk",
                "2 cangkir kaldu sapi",
                "1 sendok makan tepung maizena, larutkan dengan 2 sendok makan air (opsional, untuk mengentalkan)",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 8 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "BOLA DAGING:",
                "Campurkan daging sapi giling, telur, remah roti, bawang putih, bawang bombay dalam mangkuk besar. Aduk rata.",
                "Bentuk campuran daging menjadi bola-bola kecil dengan diameter sekitar 2,5 cm.",
                "Rebus dalam air mendidih hingga bola mengapung. Angkat, tiriskan.",
                "●●●●●",
                "SAUS SEMUR:",
                "Panaskan minyak dalam wajan. Tumis bawang bombay dan bawang putih hingga harum dan layu.",
                "Tambahkan tomat dan masak hingga tomat mulai hancur.",
                "Masukkan kecap manis, kecap asin, gula merah, dan ketumbar bubuk. Aduk rata.",
                "Tuangkan kaldu sapi dan biarkan mendidih. Kurangi panas dan biarkan saus simmer selama 5 menit.",
                "Masukkan bola daging ke dalam saus dan biarkan memasak selama 10 menit hingga bola daging terendam dan saus mengental. Jika saus terlalu cair, tambahkan larutan tepung maizena untuk mengentalkan (opsional)."
            ],
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "6% AKG",
                    "Detail": "Cincang bola daging, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kuah semur secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "8 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Cincang bola daging, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), tambahkan kuah semur dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "8 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "40 gram",
                    "Kecukupan Protein": "8% AKG",
                    "Detail": "Potong bola daging dengan kuahnya, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran, 2 sdm kacang-kacangan cincang (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan pada adonan bola daging dan kuah nya: 1 sdt garam.",
                    "Kandungan Protein": "8 Gram"
                }
            ]
        },

        "Bistik Galantin Daging Sapi": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "BAHAN-BAHAN GALANTIN:",
                "Daging sapi giling: 500 gram",
                "Roti tawar tanpa kulit: 50 gram (rendam dengan susu cair)",
                "Susu cair: 100 ml",
                "Telur ayam: 2 butir",
                "Bawang bombay, cincang halus: 1/2 buah",
                "Bawang putih, haluskan: 3 siung",
                "Tepung panir: 50 gram",
                "Minyak goreng untuk menumis: 2 sendok makan",
                "●●●●●",
                "BAHAN-BAHAN SAUS BISTIK:",
                "Margarin: 2 sendok makan",
                "Bawang bombay, iris tipis: 1/2 buah",
                "Bawang putih, cincang halus: 2 siung",
                "Kecap manis: 3 sendok makan",
                "Saus tomat: 3 sendok makan",
                "Kecap Inggris: 2 sendok makan",
                "Kaldu sapi cair: 250 ml",
                "Tepung maizena: 1 sendok makan (larutkan dengan sedikit air)",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 6 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "PERSIAPAN ADONAN:",
                "Tumis bawang bombay dan bawang putih hingga harum. Angkat dan sisihkan.",
                "Campurkan daging sapi giling, roti tawar yang telah direndam susu, bawang bombay tumis, bawang putih, telur, merica, pala, garam, dan gula. Aduk hingga rata.",
                "Tambahkan tepung panir sedikit demi sedikit hingga adonan bisa dibentuk.",
                "Pembentukan dan Pengukusan:",
                "Ambil adonan, bentuk lonjong atau sesuai selera, dan bungkus dengan alumunium foil atau daun pisang.",
                "Kukus galantin selama 30-40 menit hingga matang. Angkat dan dinginkan.",
                "●●●●●",
                "CARA PEMBUATAN SAUS BISTIK:",
                "Panaskan margarin di atas wajan. Tumis bawang bombay dan bawang putih hingga harum dan layu.",
                "Masukkan kecap manis, saus tomat, kecap Inggris, gula, garam, merica, dan pala bubuk. Aduk rata.",
                "Tuangkan kaldu sapi cair, lalu masak hingga mendidih.",
                "Tambahkan larutan tepung maizena, aduk hingga saus mengental."
            ],
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "10% AKG",
                    "Detail": "Cincang galantin, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan saus bistik secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "13 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Cincang gadon, tambah saus bistik secukupnya, 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "13 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "40 gram",
                    "Kecukupan Protein": "13% AKG",
                    "Detail": "Potong sedang galantin, tambahkan sausnya, nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran, 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) pada galantin dan saus: garam 1 sdt dan gula pasir 1 sdt.",
                    "Kandungan Protein": "13 Gram"
                }
            ]
        },
    },

    # Bahan Daging Ayam
    "Daging-Ayam": {
        "Nugget Ayam": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Daging ayam fillet, giling halus: 500 gram",
                "Roti tawar tanpa kulit: 2 lembar (hancurkan)",
                "Susu cair: 100 ml",
                "Telur ayam: 1 butir",
                "Bawang putih, haluskan: 2 siung",
                "Bawang bombay, cincang halus: 1/2 buah",
                "Tepung panir: 100 gram (untuk pelapis)",
                "●●●●●",
                "Nugget yang dihasilkan = 20 potong"
            ],

            "Cara-Pembuatan": [
                "Campurkan daging ayam giling, roti tawar yang telah dihancurkan, susu cair, telur, bawang putih, bawang bombay, merica, garam, dan gula. Aduk rata hingga semua bahan tercampur dengan baik.",
                "Ambil sedikit adonan dan bentuk menjadi bulat pipih atau sesuai selera. Ulangi hingga semua adonan habis.",
                "Gulingkan nugget dalam tepung panir hingga rata",
                "Kukus hingga matang"
            ],
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "19% AKG",
                    "Detail": "Cincang nugget kukus, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "25 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "21% AKG",
                    "Detail": "Cincang nugget kukus, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar",
                    "Kandungan Protein": "25 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "24% AKG",
                    "Detail": "Goreng nugget, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam dan 1/2 sdt teh gula pasir.",
                    "Kandungan Protein": "25 Gram"
                }
            ]
        },

        "Dimsum Ayam": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Daging ayam fillet, cincang halus: 500 gram",
                "Bawang putih, haluskan: 3 siung",
                "Bawang bombay, cincang halus: 1/2 buah",
                "Daun bawang, cincang halus: 2 batang",
                "1/4 cangkir bawang bombay, cincang halus",
                "Kecap asin: 2 sendok makan",
                "Saus tiram: 1 sendok makan",
                "Minyak wijen: 1 sendok makan",
                "Tepung maizena: 2 sendok makan",
                "Kulit dimsum (bisa beli di toko bahan makanan Asia): 30 lembar",
                "Air: secukupnya (untuk mengukus dimsum)",
                "●●●●●",
                "Dimsum yang dihasilkan = 30 buah"
            ],

            "Cara-Pembuatan": [
                "PERSIAPAN ISIAN:",
                "Campurkan daging ayam cincang, bawang putih, bawang bombay, dan daun bawang dalam sebuah wadah.",
                "Tambahkan kecap asin, saus tiram, minyak wijen, dan tepung maizena. Aduk rata hingga semua bahan tercampur.",
                "●●●●●",
                "PEMBENTUKAN DIMSUM:",
                "Ambil selembar kulit dimsum, letakkan satu sendok makan isian di tengah kulit.",
                "Lipat kulit dimsum menjadi bentuk kantong atau sesuai selera dan rapatkan sisi-sisinya.",
                "Ulangi hingga semua isian habis.",
                "●●●●●",
                "MENGUKUS DIMSUM",
                "Panaskan kukusan yang telah dilapisi dengan daun pisang atau kertas roti agar dimsum tidak menempel.",
                "Tempatkan dimsum dalam kukusan, beri jarak agar tidak saling menempel.",
                "Kukus dimsum selama 15-20 menit hingga matang.",
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Cincang dimsum, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "8 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Cincang dimsum, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "8 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "8% AKG",
                    "Detail": "Potong dimsum, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1/2 sdt garam dan 1 sdt gula pasir.",
                    "Kandungan Protein": "8 Gram"
                }
            ]
        },

        "Rolade Ayam": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "ADONAN ROLADE",
                "Daging ayam fillet, cincang halus: 500 gram",
                "Telur ayam, kocok lepas: 1 butir",
                "Wortel, parut halus: 1 buah (sekitar 100 gram)",
                "Bawang putih, haluskan: 2 siung",
                "Bawang bombay, cincang halus: 1 buah",
                "Kecap asin: 1 sendok makan",
                "Tepung panir: 3 sendok makan",
                "●●●●●",
                "PEMBUNGKUS TELUR:",
                "Telur ayam: 4 butir",
                "Minyak untuk menggoreng: 2 sendok makan",
                "●●●●●",
                "PEMBUNGKUS ROLADE:",
                "Aluminium foil: secukupnya",
                "●●●●●",
                "Jumlah yang dihasilkan = 8 gulung"
            ],

            "Cara-Pembuatan": [
                "PERSIAPAN ADONAN ROLADE:",
                "Campurkan daging ayam cincang, wortel parut, bawang putih, bawang bombay, kecap asin dalam sebuah mangkuk. Aduk rata hingga semua bahan tercampur dengan baik.",
                "Tambahkan telur kocok dan tepung panir untuk mengikat adonan. Aduk rata.",
                "●●●●●",
                "MEMBUAT TELUR GULUNG:",
                "Panaskan wajan dengan sedikit minyak dan buat telur dadar tipis dari 4 butir telur, lalu angkat dan tiriskan.",
                "●●●●●",
                "MEMBENTUK ROLADE:",
                "Panaskan kukusan. Kukus rolade dalam foil selama 30 menit hingga matang."
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Cincang rolade, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "19 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "16% AKG",
                    "Detail": "Cincang rolade, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "19 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "18% AKG",
                    "Detail": "Potong sedang rolade, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "19 Gram"
                }
            ]
        },
    },

    # Bahan Ikan Kembung
    "Ikan-Kembung": {
        "Asem-asem Ikan Kembung": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan kembung segar: 500 gram (4 ekor ukuran sedang)",
                "Roti tawar tanpa kulit: 2 lembar (hancurkan)",
                "Air: 800 ml",
                "Tomat merah: 150 gram (2 buah), potong-potong",
                "Bawang merah: 50 gram (5 siung), iris tipis",
                "Bawang putih: 20 gram (4 siung), iris tipis",
                "Serai: 1 batang, memarkan",
                "Lengkuas: 20 gram, memarkan",
                "Daun salam: 2 lembar",
                "Asam jawa: 15 gram, larutkan dalam 50 ml air",
                "Gula pasir: 1 sendok teh (5 gram)",
                "Minyak untuk menumis: 2 sendok makan (30 ml)",
                "●●●●●",
                "Porsi yang dihasilkan = 4 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "PERSIAPAN IKAN:",
                "Bersihkan ikan kembung, buang isi perut dan insangnya.",
                "Potong masing-masing ikan menjadi dua bagian."
                "●●●●●",
                "MENUMIS BUMBU",
                "Panaskan minyak di wajan, tumis bawang merah dan bawang putih hingga harum.",
                "Tambahkan serai, lengkuas, daun salam, aduk hingga layu.",
                "●●●●●",
                "MEMASAK IKAN:",
                "Masukkan ikan kembung yang sudah dipotong ke dalam tumisan bumbu.",
                "Aduk perlahan agar ikan tidak hancur.",
                "●●●●●",
                "MEMASAK IKAN",
                "Tambahkan air ke dalam wajan, biarkan hingga mendidih. Setelah mendidih, masukkan tomat dan air asam jawa. Tambahkan gula, aduk rata.",
                "●●●●●",
                "PENYELESAIAN:",
                "Masak dengan api kecil hingga ikan matang dan bumbu meresap (sekitar 15-20 menit). Angkat dan sajikan."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Pisahkan daging ikan dengan durinya, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kuah asem-asem secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kuah hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Pisahkan ikan dengan durinya, cincang daging ikan, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kuah secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "17% AKG",
                    "Detail": "Pisahkan daging ikan dengan durinya, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kuah asem-asem), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam(5 gram).",
                    "Kandungan Protein": "18 Gram"
                }
            ]
        },

        "Ikan Kembung Masak Sarden": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan kembung segar: 500 gram (3 ekor ukuran sedang)",
                "Bawang merah: 50 gram (5 siung), cincang halus",
                "Bawang putih: 20 gram (4 siung), cincang halus",
                "Tomat merah: 150 gram (2 buah), cincang halus",
                "Saus tiram: 1 sendok makan (15 ml)",
                "Air jeruk nipis: 1 sendok makan (15 ml)",
                "Jahe: 10 gram, memarkan",
                "Lengkuas: 10 gram, memarkan",
                "Daun salam: 2 lembar",
                "Gula pasir: 1 sendok teh (5 gram)",
                "Air: 200 ml",
                "Minyak untuk menumis: 3 sendok makan (45 ml)",
                "●●●●●",
                "Porsi yang dihasilkan = 4 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "PERSIAPAN IKAN",
                "Bersihkan ikan kembung, buang isi perut dan insangnya.",
                "Lumuri ikan dengan garam dan air jeruk nipis, diamkan selama 10 menit, kemudian goreng setengah matang. Tiriskan.",
                "●●●●●",
                "MENUMIS BUMBU:",
                "Panaskan minyak di wajan, tumis bawang merah, bawang putih, jahe, lengkuas, dan daun salam hingga harum.",
                "●●●●●",
                "MEMASAK SAUS:",
                "Masukkan tomat cincang dan saus tiram. Aduk hingga tomat layu dan mengeluarkan air.",
                "Tambahkan air. Masak hingga saus mengental dan bumbu meresap.",
                "●●●●●",
                "MENGHIDANGKAN:",
                "Masukkan ikan kembung yang sudah digoreng ke dalam saus. Aduk perlahan hingga ikan terbalut saus. Masak selama 5 menit hingga bumbu meresap. Angkat dan sajikan."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Pisahkan ikan dari durinya, ambil sausnya, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Pisahkan ikan dari durinya, tambahkan saus tomat, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "17% AKG",
                    "Detail": "Pisahkan ikan dari durinya, tambahkan saus tomat dan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam",
                    "Kandungan Protein": "18 Gram"
                }
            ]
        },

        "Otak-otak Ikan Kembung": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan kembung fillet: 500 gram",
                "Santan kental: 100 ml",
                "Tepung tapioka: 150 gram",
                "Bawang merah: 50 gram (5 siung), cincang halus",
                "Bawang putih: 20 gram (4 siung), cincang halus",
                "Daun bawang: 30 gram, iris halus",
                "Putih telur: 2 butir",
                "Daun pisang: secukupnya untuk membungkus",
                "Minyak untuk menumis: 2 sendok makan (30 ml)",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 23 Bungkus"
            ],

            "Cara-Pembuatan": [
                "Bersihkan ikan kembung, fillet dagingnya, lalu haluskan daging ikan menggunakan food processor atau blender hingga halus.",
                "Tumis bawang merah dan bawang putih dengan minyak hingga harum, kemudian angkat.",
                "Campurkan daging ikan yang sudah dihaluskan dengan santan, tepung tapioka, putih telur, daun bawang, tumisan bawang merah dan bawang putih, gula, garam, dan merica. Aduk rata hingga menjadi adonan yang bisa dibentuk.",
                "Ambil selembar daun pisang, letakkan 2-3 sendok makan adonan di atasnya, lalu bungkus dan sematkan dengan tusuk gigi.",
                "Kukus otak-otak dalam kukusan yang sudah dipanaskan selama 20-30 menit hingga matang.",
                "Panaskan wajan dengan sedikit minyak dan buat telur dadar tipis dari 4 butir telur, lalu angkat dan tiriskan.",
                "Setelah dikukus, otak-otak bisa dipanggang sebentar di atas api untuk memberikan aroma panggang yang khas.",
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Cincang otak-otak, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "19 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "16% AKG",
                    "Detail": "Cincang otak-otak, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "19 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "18% AKG",
                    "Detail": "Potong sedang otak-otak, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt gula pasir dan 1 sdt garam.",
                    "Kandungan Protein": "19 Gram"
                }
            ]
        },
    },

    # Bahan Udang
    "Udang": {
        "Bola Udang": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Udang kupas (tanpa kulit dan kepala): 500 gram",
                "Tepung tapioka: 100 gram",
                "Telur ayam: 1 butir",
                "Bawang putih: 20 gram (4 siung), cincang halus",
                "Bawang putih: 20 gram (4 siung), cincang halus",
                "Wortel: 50 gram, parut halus",
                "●●●●●",
                "Porsi yang dihasilkan = 20 bola udang"
            ],

            "Cara-Pembuatan": [
                "Bersihkan udang, buang kulit dan kepalanya. Haluskan udang dengan blender atau food processor hingga menjadi pasta.",
                "Campurkan udang yang sudah dihaluskan dengan tepung tapioka, telur, bawang putih, daun bawang, wortel, garam, merica, dan gula. Aduk rata hingga menjadi adonan yang bisa dibentuk.",
                "Ambil sekitar 1 sendok makan adonan, lalu bentuk menjadi bola-bola kecil dengan tangan.",
                "Didihkan air dalam panci. Masukkan bola-bola udang ke dalam air mendidih dan rebus hingga mengapung dan matang, sekitar 3-5 menit."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Cincang bola udang, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "12% AKG",
                    "Detail": "Cincang bola udang, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Potong bola, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) garam 1 sdt dan 1 sdt gula pasir.",
                    "Kandungan Protein": "14 Gram"
                }
            ]
        },

        "Patty Udang": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                
                "Udang kupas (tanpa kulit dan kepala): 500 gram",
                "Tepung roti (breadcrumbs): 100 gram",
                "Telur ayam: 1 butir",
                "Bawang putih: 20 gram (4 siung), cincang halus",
                "Daun bawang: 30 gram, iris halus",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 12 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Bersihkan udang, buang kulit dan kepalanya. Haluskan udang dengan blender atau food processor hingga menjadi pasta.",
                "Bersihkan ikan kembung, buang isi perut dan insangnya.",
                "Campurkan udang yang sudah dihaluskan dengan tepung roti, telur, bawang putih, daun bawang, wortel. Aduk rata hingga menjadi adonan yang bisa dibentuk.",
                "Ambil sekitar 2 sendok makan adonan, lalu bentuk menjadi patty datar dengan diameter sekitar 7-8 cm dan ketebalan sekitar 1 cm.",
                "Panaskan teflon dengan sedikit minyak di atas api sedang. Panggang patty udang selama sekitar 3-4 menit per sisi atau hingga berwarna keemasan dan matang. Pastikan teflon tidak terlalu panas agar patty tidak gosong."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Cincang patty, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "15 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "13% AKG",
                    "Detail": "Cincang patty, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "15 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Potong patty, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam dan 1 sdt gula pasir.",
                    "Kandungan Protein": "15 Gram"
                }
            ]
        },

        "Lumpia Udang": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Kulit tahu (tauhu kering): 10 lembar",
                "Udang kupas (tanpa kulit dan kepala): 300 gram",
                "Bawang putih: 20 gram (4 siung), cincang halus",
                "Wortel: 50 gram, parut halus",
                "Kecap asin: 2 sendok makan (30 ml)",
                "Minyak wijen: 1 sendok makan (15 ml)",
                "Jahe: 10 gram, parut halus",
                "Air matang: secukupnya (untuk merendam kulit tahu)",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 16 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Rendam kulit tahu dalam air matang hingga lembut, kemudian tiriskan dan peras. Potong-potong kulit tahu menjadi ukuran sekitar 7x7 cm atau sesuai selera.",
                "Bersihkan udang dan buang kulit serta kepalanya. Potong udang menjadi ukuran kecil (sekitar 1-2 cm).",
                "Campurkan udang dengan bawang putih, daun bawang, wortel, kecap asin, minyak wijen, dan jahe. Aduk rata hingga semua bahan tercampur.",
                "Ambil selembar kulit tahu, letakkan satu sendok makan adonan udang di tengah kulit. Lipat sisi kiri dan kanan kulit tahu, lalu gulung hingga rapat. Ulangi hingga semua adonan dan kulit tahu habis.",
                "Panaskan panci pengukus dengan air mendidih. Letakkan kulit tahu isi udang di dalam saringan pengukus. Kukus selama sekitar 10-12 menit atau hingga udang matang dan kulit tahu lembut.",
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "8% AKG",
                    "Detail": "Cincang lumpia, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "11 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "9% AKG",
                    "Detail": "Cincang lumpia, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "11 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Potong lumpia, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1/2 sdt garam.",
                    "Kandungan Protein": "11 Gram"
                }
            ]
        },
    },

    # Bahan Ikan Nila
    "Ikan-Nila": {
        "Nila Kuah Kuning": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan nila 500 gram, potong-potong",
                "Air: 1 liter",
                "Bawang merah: 3 siung (15 gram), iris tipis",
                "Bawang putih: 3 siung (15 gram), iris tipis",
                "Jahe: 20 gram, memarkan",
                "Kunyit: 20 gram, memarkan atau parut halus",
                "Serai: 1 batang, memarkan",
                "Daun salam: 2 lembar",
                "Tomat: 1 buah (100 gram), potong-potong",
                "Daun bawang: 2 batang (20 gram), iris halus",
                "Minyak untuk menumis: 1 sendok makan (15 ml)",
                "●●●●●",
                "Porsi yang dihasilkan = 5 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Panaskan minyak dalam wajan. Tumis bawang merah, bawang putih, jahe, kunyit, dan serai hingga harum.",
                "Tambahkan air ke dalam wajan. Masukkan daun salam dan tomat. Masak hingga mendidih.",
                "Masukkan potongan ikan nila ke dalam kuah. Tambahkan garam dan gula pasir. Masak hingga ikan matang dan kuah sedikit mengental (sekitar 10-15 menit).",
                "Terakhir, tambahkan daun bawang ke dalam kuah dan aduk rata. Masak sebentar hingga daun bawang layu."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Pisahkan ikan dari durinya, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kuah kuning secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kuah hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "20 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "17% AKG",
                    "Detail": "Pisahkan ikan dari durinya, cincang ikan, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kuah kuning secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "20 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "19% AKG",
                    "Detail": "Pisahkan ikan dari durinya, tambahkan kuah, nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung)",
                    "Kandungan Protein": "20 Gram"
                }
            ]
        },

        "Nila Saus Mentega": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan nila fillet: 500 gram, potong-potong",
                "Mentega: 3 sendok makan (45 gram)",
                "Bawang putih: 3 siung (15 gram), cincang halus",
                "Bawang merah: 2 siung (10 gram), cincang halus",
                "Kecap manis: 2 sendok makan (30 ml)",
                "Saus tiram: 1 sendok makan (15 ml)",
                "Air jeruk nipis: 1 sendok makan (15 ml)",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 4 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Lumuri potongan ikan nila dengan air jeruk nipis. Diamkan selama 15 menit.",
                "Panaskan 1 sendok makan mentega dalam wajan besar di atas api sedang. Masak potongan ikan nila hingga matang dan berwarna keemasan, sekitar 3-4 menit per sisi. Angkat dan sisihkan.",
                "Dalam wajan yang sama, tambahkan sisa mentega. Tumis bawang putih dan bawang merah hingga harum dan berwarna keemasan.",
                "Tambahkan kecap manis dan saus tiram ke dalam wajan. Aduk rata hingga saus mengental dan bercampur dengan mentega.",
                "Masukkan ikan nila kembali ke dalam wajan dan aduk rata dengan saus. Masak selama 2-3 menit hingga ikan terbalut saus dan panas merata."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Cincang ikan, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "12% AKG",
                    "Detail": "Cincang ikan, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Potong sedang ikan, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1/2 sdt garam.",
                    "Kandungan Protein": "14 Gram"
                }
            ]
        },

        "Sate Lilit Ikan Nila": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan nila fillet: 500 gram",
                "Kelapa parut: 100 gram",
                "Tepung sagu: 50 gram",
                "Bawang merah: 5 butir",
                "Bawang putih: 3 siung",
                "Daun jeruk: 5 lembar, iris halus",
                "Serai: 5 batang, bagian putihnya saja",
                "Kunyit: 1 cm",
                "Jahe: 1 cm",
                "Lengkuas: 1 cm",
                "Kencur: 1 cm",
                "Ketumbar bubuk: 1 sdt",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 12 tusuk"
            ],

            "Cara-Pembuatan": [
                "Haluskan bawang merah, bawang putih, kunyit, jahe, lengkuas, kencur, dan ketumbar bubuk.",
                "Blender atau cincang halus ikan nila fillet hingga halus.",
                "Campurkan ikan nila yang sudah dihaluskan dengan bumbu yang dihaluskan tadi, kelapa parut, tepung sagu, daun jeruk. Aduk rata hingga adonan bisa dibentuk.",
                "Ambil sedikit adonan ikan, lalu lilitkan pada batang serai hingga padat dan berbentuk lonjong.",
                "Kukus sate dalam kukusan selasa kurang lebih 15 menit.",
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "12% AKG",
                    "Detail": "Cincang sate, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "16 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "13% AKG",
                    "Detail": "Cincang sate, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "16 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Potong sedang sate, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan(per resep) 1 sdt gula merah (serut halus) dan 1 sdt garam.",
                    "Kandungan Protein": "16 Gram"
                }
            ]
        },
    },

    # Bahan Ikan Lele
    "Ikan-Lele": {
        "Perkedel Ikan Lele": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan lele fillet: 500 gram",
                "Kentang: 250 gram",
                "Bawang merah: 5 butir, cincang halus",
                "Bawang putih: 3 siung, cincang halus",
                "Seledri: 2 batang, iris halus",
                "Telur: 1 butir",
                "Minyak goreng: 1 sdm (untuk olesan teflon)",
                "●●●●●",
                "Porsi yang dihasilkan = 12 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Kupas dan potong kentang, lalu kukus hingga matang dan empuk. Setelah itu, haluskan kentang yang sudah dikukus.",
                "Kukus ikan lele hingga matang, lalu pisahkan dari durinya dan suwir-suwir atau cincang halus dagingnya.",
                "Campurkan kentang yang sudah dihaluskan dengan ikan lele yang sudah dicincang, bawang merah, bawang putih, dan seledri. Aduk hingga semua bahan tercampur rata."
                "Tambahkan telur ke dalam adonan dan aduk.",
                "Ambil sedikit adonan, lalu bentuk bulat pipih sesuai selera.",
                "Panaskan teflon dan oleskan sedikit minyak goreng. Panggang perkedel di atas teflon dengan api kecil hingga kedua sisi berwarna kecoklatan dan matang. Bolak-balik perkedel secara perlahan agar matang merata tanpa gosong."
            ],
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Cincang perkedel, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "15 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "13% AKG",
                    "Detail": "Cincang perkedel, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "15 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Potong sedang gadon, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "15 Gram"
                }
            ]
        },

        "Mangut Lele": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan nila fillet: 500 gram, potong-potong",
                "Mentega: 3 sendok makan (45 gram)",
                "Bawang putih: 3 siung (15 gram), cincang halus",
                "Bawang merah: 2 siung (10 gram), cincang halus",
                "Kecap manis: 2 sendok makan (30 ml)",
                "Saus tiram: 1 sendok makan (15 ml)",
                "Air jeruk nipis: 1 sendok makan (15 ml)",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 4 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Lumuri potongan ikan nila dengan air jeruk nipis. Diamkan selama 15 menit.",
                "Panaskan 1 sendok makan mentega dalam wajan besar di atas api sedang. Masak potongan ikan nila hingga matang dan berwarna keemasan, sekitar 3-4 menit per sisi. Angkat dan sisihkan.",
                "Dalam wajan yang sama, tambahkan sisa mentega. Tumis bawang putih dan bawang merah hingga harum dan berwarna keemasan.",
                "Tambahkan kecap manis dan saus tiram ke dalam wajan. Aduk rata hingga saus mengental dan bercampur dengan mentega.",
                "Masukkan ikan nila kembali ke dalam wajan dan aduk rata dengan saus. Masak selama 2-3 menit hingga ikan terbalut saus dan panas merata."
            ],
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Cincang ikan, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "12% AKG",
                    "Detail": "Cincang ikan, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Potong sedang ikan, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1/2 sdt garam.",
                    "Kandungan Protein": "14 Gram"
                }
            ]
        },

        "Sempol Ikan Lele": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan lele fillet: 400 gram",
                "Tepung sagu: 50 gram",
                "Bawang merah: 2 siung, cincang halus",
                "Telur ayam: 1 butir",
                "Kaldu bubuk: 1/2 sendok teh (opsional)",
                "Daun bawang: 1 batang, iris halus",
                "Air es: 2 sendok makan (jika adonan terlalu kering)",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 12 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Cuci bersih ikan lele fillet, tiriskan.",
                "Rebus ikan lele dalam air mendidih selama 5-7 menit atau hingga matang. Angkat dan tiriskan.",
                "Setelah ikan dingin, suwir-suwir atau haluskan ikan lele menggunakan garpu atau food processor hingga menjadi adonan yang halus.",
                "Dalam mangkuk besar, campurkan ikan lele halus, tepung sagu, bawang putih, bawang merah, telur, kaldu bubuk, dan daun bawang. Aduk rata hingga menjadi adonan yang bisa dipulung. Jika adonan terlalu kering, tambahkan sedikit air es.",
                "Ambil sejumput adonan dan bentuk menjadi bulat atau lonjong sesuai selera.",
                "Siapkan alat pengukus yang sudah dipanaskan. Letakkan sempol di atas rak pengukus yang sudah dialasi dengan daun pisang atau kertas roti.",
                "Kukus sempol selama 15-20 menit hingga matang dan padat."
            ],         
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "13% AKG",
                    "Detail": "Cincang sate, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe kecil), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "17 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Cincang sempol, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "17 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "16% AKG",
                    "Detail": "Sajikan sempol, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "17 Gram"
                }
            ]
        },
    },

    # Bahan Ikan Patin
    "Ikan-Patin": {
        "Siomay Ikan Patin": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan patin fillet: 300 gram",
                "Tepung tapioka: 100 gram",
                "Bawang putih: 2 siung, cincang halus",
                "Daun bawang: 2 batang, iris halus",
                "Wortel: 1 buah, parut halus",
                "Telur ayam: 1 butir",
                "Kaldu bubuk: 1/2 sendok teh (opsional)",
                "Kulit pangsit: 20 lembar",
                "●●●●●",
                "Porsi yang dihasilkan = 20 buah"
            ],

            "Cara-Pembuatan": [
                "Cuci bersih ikan patin fillet, lalu tiriskan.",
                "Haluskan ikan patin dengan food processor atau blender hingga benar-benar halus.",
                "Dalam mangkuk besar, campurkan ikan patin yang sudah dihaluskan dengan tepung tapioka, bawang putih cincang, daun bawang, wortel parut, dan telur.",
                "Tambahkan kaldu bubuk (jika menggunakan). Aduk rata hingga menjadi adonan yang bisa dipulung.",
                "Ambil selembar kulit pangsit, letakkan 1 sendok makan adonan siomay di tengahnya.",
                "Lipat dan bentuk sesuai selera, pastikan adonan tertutup dengan baik oleh kulit pangsit.",
                "Panaskan alat pengukus.",
                "Letakkan siomay di atas rak pengukus yang sudah diolesi minyak atau dialasi dengan daun pisang.",
                "Kukus siomay selama 15-20 menit hingga matang dan kulit pangsit terlihat transparan."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "9% AKG",
                    "Detail": "Cincang siomay, campur dengan ¼ bagian kentang kukus, 1 sdm sayuran cincang, 1 sdm saus kacang siomay, ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "12 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "10% AKG",
                    "Detail": "Cincang siomay, campur dengan 1/3 bagian kentang kukus, 2 sdm sayuran cincang, 2 sdm saus kacang siomay, ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), saus kacang siomay dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "12 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "12% AKG",
                    "Detail": "Sajikan siomay, 1 buah kentang ukuran kecil, 1 mangkok kecil sayuran, 4 sdm saus kacang, ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam dan 1/2 sdt gula pasir.",
                    "Kandungan Protein": "12 Gram"
                }
            ]
        },

        "Garang Asem Ikan Patin": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan patin fillet: 500 gram",
                "Daun salam: 4 lembar",
                "Daun jeruk: 4 lembar, buang tulangnya",
                "Tomat merah: 2 buah, potong-potong",
                "Belimbing wuluh: 5 buah, potong-potong",
                "Bawang merah: 8 siung, iris tipis",
                "Bawang putih: 4 siung, iris tipis",
                "Serai: 2 batang, memarkan",
                "Lengkuas: 3 cm, memarkan",
                "Kaldu bubuk: 1/2 sendok teh (opsional)",
                "Santan encer: 300 ml",
                "Daun pisang: secukupnya (untuk membungkus)",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 4 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Cuci bersih ikan patin fillet, lalu potong-potong menjadi ukuran sekitar 4-5 cm.",
                "Siapkan daun pisang, jemur sebentar agar layu dan mudah dilipat. Potong daun pisang sesuai ukuran yang diperlukan untuk membungkus ikan patin.",
                "Campurkan irisan bawang merah, bawang putih, serai, lengkuas, kunyit, daun salam, dan daun jeruk.",
                "Tambahkan tomat dan belimbing wuluh ke dalam campuran bumbu, aduk rata.",
                "Masukkan ikan patin ke dalam campuran bumbu, aduk hingga ikan terbalut bumbu.",
                "Ambil selembar daun pisang, letakkan beberapa potongan ikan patin beserta bumbu di tengahnya.",
                "Tambahkan sedikit santan encer di atas ikan dan bumbunya.",
                "Bungkus rapat daun pisang, sematkan dengan tusuk gigi atau lidi agar tidak terbuka saat dikukus.",
                "Ulangi hingga semua ikan dan bumbu habis.",
                "Panaskan alat pengukus, kukus bungkusan garang asem selama 30-40 menit hingga matang."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Cincang ikan, tambahkan kuah ikan, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Cincang ikan, tambahkan kuah ikan, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "17% AKG",
                    "Detail":"Sajikan ikan bersama kuahnya, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt gula pasir dan 1 sdt garam.",
                    "Kandungan Protein": "18 Gram"
                }
            ]
        },

        "Ikan Patin Saus Tiram": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Ikan patin fillet: 500 gram",
                "Bawang putih: 4 siung, cincang halus",
                "Bawang bombay: 1/2 buah, iris tipis",
                "Saus tiram: 4 sendok makan",
                "Kecap asin: 1 sendok makan",
                "Kecap manis: 1 sendok teh",
                "Minyak goreng: 2 sendok makan (untuk menumis)",
                "Air: 100 ml",
                "Daun bawang: 1 batang, iris halus (opsional)",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 4 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Cuci bersih ikan patin fillet, kemudian potong-potong menjadi ukuran sedang (sekitar 4-5 cm).",
                "Panaskan minyak goreng dalam wajan.",
                "Goreng potongan ikan patin hingga kedua sisi berwarna kecokelatan dan matang. Angkat dan tiriskan.",
                "Dalam wajan yang sama, kurangi minyak goreng hingga tersisa sekitar 1 sendok makan.",
                "Tumis bawang putih dan bawang bombay hingga harum dan layu.",
                "Tambahkan saus tiram, kecap asin, kecap manis dan air. Aduk hingga bumbu tercampur rata.",
                "Masukkan ikan patin yang sudah digoreng, aduk perlahan hingga ikan terbalut saus.",
                "Masak dengan api kecil selama 5-7 menit hingga saus meresap dan mengental."
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Cincang ikan, campur dengan sausnya dan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Cincang ikan, tambahkan sausnya, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "17% AKG",
                    "Detail": "Sajikan ikan dengan sausnya, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt gula pasir dan garam secukupnya.",
                    "Kandungan Protein": "18 Gram"
                }
            ]
        },
    },

    # Bahan Telur Ayam Broiler
    "Telur-Ayam-Broiler": {
        "Omelet Sayuran": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Telur ayam: 3 butir (sekitar 150 gram)",
                "Keju cheddar parut: 50 gram",
                "Wortel: 50 gram, serut halus",
                "Bayam: 50 gram, cincang kasar",
                "Bawang bombay: 30 gram, cincang halus",
                "Paprika merah: 1 buah, iris tipis",
                "Minyak zaitun: 1 sendok makan (untuk menumis)",
                "Susu cair: 30 ml (opsional, untuk menambah kelembutan)",
                "●●●●●",
                "Porsi yang dihasilkan = 2 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Kocok telur ayam dalam wadah, tambahkan susu cair jika menggunakan, garam, dan merica. Aduk rata.",
                "Panaskan minyak zaitun dalam wajan dengan api sedang.",
                "Tumis bawang bombay hingga harum, kemudian masukkan wortel, bayam, dan tomat merah. Tumis sebentar hingga layu.",
                "Tuang adonan telur ke dalam wajan, ratakan sayuran di atasnya.",
                "Taburi keju parut di atas telur yang masih cair.",
                "Masak dengan api kecil hingga bagian bawah omelet matang dan berwarna keemasan.",
                "Lipat omelet menjadi dua, dan masak sebentar hingga keju meleleh sempurna.",
                "Angkat omelet dari wajan dan potong menjadi beberapa bagian sesuai selera."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Cincang omelet, campur dengan 2-3 sdm nasi lembek, ½ sdm kacang-kacangan (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "19 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "16% AKG",
                    "Detail": "Cincang omelet, campur dengan 5 sdm nasi tim (1 centong sedang), 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "19 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "18% AKG",
                    "Detail": "Potong omelet ukuran sedang, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung).",
                    "Kandungan Protein": "18 Gram"
                }
            ]
        },

        "Telur Kukus Ayam": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Telur ayam: 6 butir (sekitar 300 gram)",
                "Ayam cincang: 150 gram",
                "Wortel serut: 50 gram",
                "Bawang merah: 1 butir, cincang halus",
                "Bawang putih: 2 siung, cincang halus",
                "Daun bawang: 2 batang, iris halus",
                "Kecap asin: 1 sendok makan",
                "Minyak goreng: 1 sendok makan (untuk menumis bumbu)",
                "Air: 200 ml (untuk uap",
                "Cup aluminium foil (untuk membagi adonan)",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 6 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Kocok telur dalam mangkuk hingga rata.",
                "Cincang halus bawang merah, bawang putih, dan iris daun bawang.",
                "Panaskan minyak goreng dalam wajan.",
                "Tumis bawang merah dan bawang putih hingga harum.",
                "Tambahkan wortel serut, ayam cincang. Masak hingga ayam matang dan bumbu meresap. Angkat dan biarkan dingin sedikit.",
                "Campurkan ayam cincang tumis ke dalam telur kocok.",
                "Tambahkan daun bawang dan kecap asin. Aduk rata.",
                "Siapkan mangkuk aluminium foil.",
                "Tuang adonan telur dan ayam ke dalam masing-masing mangkuk.",
                "Siapkan alat pengukus dan panaskan air di dalamnya.",
                "Tempatkan mangkuk yang berisi adonan telur dan ayam di dalam pengukus.",
                "Kukus selama sekitar 20-25 menit hingga telur matang dan mengembang."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "11% AKG",
                    "Detail": "Cincang telur, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "12% AKG",
                    "Detail": "Cincang telur, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "14 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "17% AKG",
                    "Detail":"Potong telur, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1/2 sdt garam.",
                    "Kandungan Protein": "14 Gram"
                }
            ]
        },

        "Telur Ceplok Bumbu Kari": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Telur ayam: 6 butir (sekitar 300 gram)",
                "Santan kelapa: 200 ml (gunakan santan encer untuk rasa yang lebih ringan)",
                "Bawang merah: 2 butir, cincang halus",
                "Bawang putih: 3 siung, cincang halus",
                "Jahe: 1 cm, cincang halus",
                "Kunyit bubuk: 1/2 sendok teh",
                "Ketumbar bubuk: 1/2 sendok teh",
                "Minyak goreng: 1 sendok makan (untuk menumis bumbu)",
                "Daun bawang: 1 batang, iris halus (untuk garnish, opsional)",
                "Air: 100 ml (untuk menambah cairan)",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 13 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Ceplok telur satu per satu dalam air mendidih, angkat, lalu sisihkan, jgan buang airnya.",
                "Cincang halus bawang merah, bawang putih, dan jahe.",
                "Panaskan minyak goreng dalam wajan.",
                "Tumis bawang merah, bawang putih, dan jahe hingga harum.",
                "Tambahkan kunyit bubuk, ketumbar bubuk. Aduk rata.",
                "Tambahkan santan dan air sisa telur ceplok ke dalam wajan. Aduk rata dan biarkan mendidih.",
                "Masukkan telur ke dalam wajan. Aduk perlahan. Masak hingga matang, sekitar 5-7 menit.",
                "Jika perlu, tambahkan sedikit air jika kuah terlalu kental."
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "14% AKG",
                    "Detail": "Cincang telur, tambahkan kuah kari, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, ½ sdm kacang-kacangan (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "13 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "15% AKG",
                    "Detail": "Cincang telur, tambahkan kuah kari, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 1 sdm kacang-kacangan cincang (setara dengan ½ potong tempe sedang), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "13 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "17% AKG",
                    "Detail": "Potong telur, tambahkan kuah kari dan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 2 sdm kacang-kacangan (setara dengan 1 potong tempe sedang), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "13 Gram"
                }
            ]
        },
    },

    # Bahan Tempe
    "Tempe": {
        "Kroket Tempe": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "UNTUK ISIAN:",
                "Tempe: 300 gram (hancurkan halus)",
                "Kentang: 200 gram (rebus, haluskan)",
                "Bawang merah: 2 butir (cincang halus)",
                "Bawang putih: 2 siung (cincang halus)",
                "Wortel: 100 gram (parut halus)",
                "Daun bawang: 2 batang (cincang halus)",
                "Ketumbar bubuk: 1/2 sendok teh",
                "Minyak goreng: Secukupnya (untuk menumis)",
                "●●●●●",
                "UNTUK PELAPIS:",
                "Tepung terigu: 50 gram",
                "Telur ayam: 2 butir (kocok lepas)",
                "Tepung panir (breadcrumbs): 100 gram",
                "●●●●●",
                "Porsi yang dihasilkan = 20 bola"
            ],

            "Cara-Pembuatan": [
                "Panaskan sedikit minyak dalam wajan. Tumis bawang merah dan bawang putih hingga harum.",
                "Tambahkan wortel parut, dan masak hingga lembut.",
                "Masukkan tempe yang sudah dihancurkan dan kentang halus. Aduk rata.",
                "Tambahkan daun bawang dan ketumbar bubuk. Aduk hingga semua bahan tercampur rata.",
                "Masak hingga isian agak kering dan matang. Angkat dan dinginkan.",
                "Ambil sedikit adonan isian dan bentuk menjadi bulat lonjong.",
                "Gulingkan bola-bola kroket dalam tepung terigu, celupkan ke dalam telur kocok, dan gulingkan dalam tepung panir hingga rata.",
                "Panaskan minyak dalam wajan dengan api sedang.",
                "Goreng kroket dalam minyak panas hingga berwarna cokelat keemasan dan renyah, sekitar 3-4 menit per sisi.",
                "Angkat dan tiriskan kroket di atas kertas dapur untuk menghilangkan minyak berlebih."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "6% AKG",
                    "Detail": "Cincang kroket, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, 20 gram lauk hewani (setara dengan 1/3 bagian telur), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "10 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Cincang kroket, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 25 gram lauk hewani (setara dengan ½ bagian telur), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "10 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Potong kroket, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 30 gram lauk hewani (setara dengan ¾ bagian telur), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "10 Gram"
                }
            ]
        },

        "Tempe Bacem": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Tempe: 300 gram, potong-potong sesuai selera",
                "Air: 500 ml",
                "Kecap manis: 4 sendok makan",
                "Gula merah: 3 sendok makan, serut halus",
                "Kaldu bubuk: 1 sendok teh (opsional)",
                "Daun salam: 2 lembar",
                "Lengkuas: 2 cm, memarkan",
                "Bawang merah: 5 siung, iris tipis",
                "Bawang putih: 3 siung, iris tipis",
                "Kemiri: 3 butir, sangrai dan haluskan",
                "Jahe: 2 cm, memarkan",
                "Ketumbar bubuk: 1 sendok teh",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 5 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Potong tempe sesuai selera.",
                "Tumis bawang merah, bawang putih, dan kemiri yang sudah dihaluskan dengan sedikit minyak hingga harum. Jika tidak menggunakan minyak, tumis bumbu hingga mengeluarkan aroma harum dengan menggunakan wajan anti lengket.",
                "Dalam panci, campurkan air, kecap manis, gula merah, garam, kaldu bubuk (jika pakai), daun salam, lengkuas, jahe, dan ketumbar bubuk. Aduk hingga gula merah larut dan bumbu merata.",
                "Masukkan tempe ke dalam panci dan aduk rata. Biarkan tempe meresap bumbu dengan merebusnya dalam campuran bumbu selama 20-30 menit dengan api kecil hingga bumbu meresap dan kuah menyusut."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "10% AKG",
                    "Detail": "Cincang tempe, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, 20 gram lauk hewani (setara dengan 1/3 bagian telur), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "12% AKG",
                    "Detail": "Cincang tempe, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 25 gram lauk hewani (setara dengan ½ bagian telur), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "18 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "13% AKG",
                    "Detail":"Potong tempe, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 30 gram lauk hewani (setara dengan ¾ bagian telur), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "18 Gram"
                }
            ]
        },

        "Tempe Saus Teriyaki": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Tempe: 200 gram",
                "Bawang bombay: 1 buah (ukuran sedang, sekitar 100 gram)",
                "Jagung pipil: 50 gram",
                "Saus teriyaki: 3 sendok makan",
                "Minyak goreng: 2 sendok makan",
                "Bawang putih: 2 siung (cincang halus)",
                "Air: 100 ml",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 5 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Potong tempe menjadi dadu kecil.",
                "Iris bawang bombay tipis-tipis.",
                "Siapkan jagung pipil.",
                "Cincang halus bawang putih.",
                "Panaskan minyak goreng dalam wajan di atas api sedang.",
                "Tumis bawang putih hingga harum.",
                "Tambahkan irisan bawang bombay, tumis hingga bawang bombay layu dan beraroma harum.",
                "Masukkan potongan tempe, aduk rata hingga tempe sedikit kecoklatan.",
                "Tambahkan jagung pipil ke dalam wajan, aduk rata.",
                "Tuangkan saus teriyaki dan air, aduk hingga semua bahan tercampur rata.",
                "Masak dengan api kecil hingga jagung matang dan saus menyusut, serta tempe meresap bumbu."
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "15 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Cincang tempe dengan sausnya, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, 20 gram lauk hewani (setara dengan 1/3 bagian telur), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "12 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "8% AKG",
                    "Detail": "Cincang tempe dengan sausnya, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 25 gram lauk hewani (setara dengan ½ bagian telur), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "12 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "19% AKG",
                    "Detail": "Ambil tempe dengan sausnya, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 30 gram lauk hewani (setara dengan ¾ bagian telur), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1/2 sdt garam.",
                    "Kandungan Protein": "12 Gram"
                }
            ]
        },
    },

    # Bahan Tahu
    "Tahu": {
        "Schotel Tahu": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Tahu: 300 gram (dicincang)",
                "Keju Cheddar: 100 gram (parut)",
                "Telur: 2 butir",
                "Susu UHT: 100 ml",
                "Bawang Bombay: 1 buah (cincang halus, sekitar 100 gram)",
                "Bawang Putih: 2 siung (cincang halus)",
                "Wortel: 100 gram (parut halus)",
                "Minyak Goreng: 1 sendok makan (untuk menumis)",
                "●●●●●",
                "Porsi yang dihasilkan = 7 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Panaskan minyak goreng di wajan.",
                "Tumis bawang bombay dan bawang putih hingga harum.",
                "Tambahkan tahu yang sudah dicincang dan tumis hingga sedikit kecoklatan. Angkat dan tiriskan.",
                "Dalam mangkuk besar, kocok telur dan susu UHT hingga tercampur rata.",
                "Tambahkan keju cheddar parut. Aduk rata.",
                "Masukkan wortel parut dan tahu yang sudah ditumis ke dalam campuran telur. Aduk hingga semua bahan tercampur rata.",
                "Tuang campuran adonan ke dalam loyang yang sudah diolesi minyak atau mentega.",
                "Ratakan permukaan dan taburi dengan peterseli cincang jika menggunakan.",
                "Panggang dalam oven yang sudah dipanaskan pada suhu 180°C (350°F) selama 25-30 menit atau hingga permukaan berwarna keemasan dan set."
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "15 gram",
                    "Kecukupan Protein": "5% AKG",
                    "Detail": "Cincang tahu, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, 20 gram lauk hewani (setara dengan 1/3 bagian telur), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "9 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "6% AKG",
                    "Detail": "Cincang tahu, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 25 gram lauk hewani (setara dengan ½ bagian telur), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "9 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Potong tahu, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 30 gram lauk hewani (setara dengan ¾ bagian telur), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "9 Gram"
                }
            ]
        },

        "Tahu Gulung": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Tahu putih (keras) - 200 gram",
                "Telur ayam - 4 butir (sekitar 240 gram)",
                "Bawang daun (daun bawang) - 2 batang (20 gram), diiris halus",
                "Wortel - 50 gram, diparut halus",
                "Kecap asin - 1 sendok makan (15 gram)",
                "Minyak wijen - 1 sendok teh (5 gram)",
                "●●●●●",
                "Jumlah porsi yang dihasilkan = 5 porsi dewasa"
            ],

            "Cara-Pembuatan": [
                "Hancurkan tahu dengan garpu hingga halus dalam mangkuk besar.",
                "Masukkan 1 butir telur, bawang daun, wortel parut, kecap asin, minyak wijen, garam, dan merica. Aduk rata hingga semua bahan tercampur dengan baik.",
                "Kocok sisa telur dalam mangkuk terpisah.",
                "Buat telur dadar tipis hingga telur habis.",
                "Letakkan adonan tahu di atas telur dadar, ratakan dengan tebal sekitar 0,5 cm, lalu gulung perlahan.",
                "Panaskan panci pengukus dengan air mendidih.",
                "Kukus adonan tahu dalam panci pengukus selama 20-25 menit hingga matang dan set.",
                "Pastikan adonan matang dengan cara menusuknya dengan tusuk gigi, jika tusuk gigi keluar bersih, maka tahu gulung sudah matang.",
            ],

            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "5% AKG",
                    "Detail": "Cincang tahu, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, 20 gram lauk hewani (setara dengan 1/3 bagian telur), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "9 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "6% AKG",
                    "Detail": "Cincang tahu, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 25 gram lauk hewani (setara dengan ½ bagian telur), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "9 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "30 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail":"Potong tahu, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 30 gram lauk hewani (setara dengan ¾ bagian telur), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung). Tambahkan (per resep) 1/2 sdt garam.",
                    "Kandungan Protein": "9 Gram"
                }
            ]
        },

        "Pepes Tahu": {
            "Gambar": "menus/Background 1.png",
            "Bahan": [
                "Tahu (putih): 250 gram",
                "Daging ayam cincang: 150 gram",
                "Bawang merah: 3 siung (30 gram), iris halus",
                "Bawang putih: 2 siung (10 gram), iris halus",
                "Daun salam: 2 lembar",
                "Serai: 1 batang (10 gram), memarkan",
                "Daun kemangi: 1 ikat (20 gram), cuci bersih",
                "Minyak goreng: 1 sendok makan (15 gram) untuk menumis",
                "●●●●●",
                "Jumlah Porsi yang dihasilkan: 6 porsi dewasa"
            ],
            
            "Cara-Pembuatan": [
                "Hancurkan tahu dengan menggunakan garpu hingga teksturnya halus. Tiriskan jika ada air berlebih.",
                "Panaskan minyak dalam wajan, tumis bawang merah dan bawang putih hingga harum dan transparan.",
                "Dalam mangkuk besar, campurkan tahu yang telah dihancurkan, daging ayam cincang, bumbu tumis, daun salam, serai, daun kemangi. Aduk rata hingga semua bahan tercampur dengan baik.",
                "Ambil selembar daun pisang (atau daun pembungkus lain), letakkan sekitar 2-3 sendok makan campuran tahu dan ayam cincang di atas daun. Bungkus rapat dan sematkan dengan tusuk gigi atau tali.",
                "Siapkan panci pengukus dan panaskan air hingga mendidih. Kukus pepes selama sekitar 30-40 menit hingga matang dan bumbu meresap."
            ],
            
            "Saran-Penyajian": [
                {
                    "Usia": "6 < 9 Bulan",
                    "Porsi": "15 gram",
                    "Kecukupan Protein": "6% AKG",
                    "Detail": "Cincang tahu, campur dengan 2-3 sdm nasi lembek, 1 sdm sayuran cincang, 20 gram lauk hewani (setara dengan 1/3 bagian telur), ¼ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, tekan-tekan pada saringan kawat untuk menghilangkan potongan kasar makanan dan mendapatkan tekstur halus saring. Tambahkan kaldu hingga kental lengket (tidak encer)",
                    "Kandungan Protein": "10 Gram"
                },
                {
                    "Usia": "9 =< 12 Bulan",
                    "Porsi": "20 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Cincang tahu, campur dengan 5 sdm nasi tim (1 centong sedang), 2 sdm sayuran cincang, 25 gram lauk hewani (setara dengan ½ bagian telur), ¼ sdt minyak sehat(minyak zaitun, minyak sayur, minyak jagung), dan kaldu secukupnya. Kemudian, campur dan tekan di mangkok untuk mendapatkan tekstur bubur kasar.",
                    "Kandungan Protein": "10 Gram"
                },
                {
                    "Usia": "12 =< 24 Bulan",
                    "Porsi": "25 gram",
                    "Kecukupan Protein": "7% AKG",
                    "Detail": "Potong tahu, tambahkan nasi 100 gram (2 centong sedang, tidak munjung), 1 mangkok kecil sayuran (bisa dengan kaldu/kuah), 30 gram lauk hewani (setara dengan ¾ bagian telur), ½ sdt minyak sehat (minyak zaitun, minyak sayur, minyak jagung)/ Tambahkan (per resep) 1 sdt garam.",
                    "Kandungan Protein": "10 Gram"
                }
            ]
        },
    },
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Mengendalikan file upload
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Menjalankan model
            results = model(file_path)
            detected_classes = results.pandas().xyxy[0]['name'].unique()

            # Jika tidak ada kelas yang terdeteksi, tampilkan pesan kesalahan
            if len(detected_classes) == 0:
                error = "Gambar gagal di deteksi, silahkan perbaiki gambar."
                return render_template('index.html', img_path=file_path, error=error)

            # Get selected age_range
            age_range = request.form.get('age_range')

            # Menentukan menu yang sesuai berdasarkan detected_class
            menu_items = {}
            for detected_class in detected_classes:
                if detected_class in classes_dict:
                    menu_items.update(classes_dict[detected_class])

            # Filter saran-penyajian berdasarkan age_range
            for menu_name, menu_details in menu_items.items():
                menu_details['Saran-Penyajian'] = [
                    saran for saran in menu_details['Saran-Penyajian'] if saran['Usia'] == age_range
                ]

            return render_template('index.html', img_path=file_path, menu_items=menu_items)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)