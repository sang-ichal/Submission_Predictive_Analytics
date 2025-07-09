# Laporan Proyek Machine Learning - Faizal Riza

# Predictive Analysis

## Domain Proyek
Di tengah melimpahnya data pendidikan, proyek ini berinovasi dengan menggunakan analisis prediktif untuk mengoptimalkan intervensi siswa. Pendekatan tradisional yang reaktif seringkali terlambat, sehingga kami mengembangkan solusi proaktif. Kami akan menerapkan dan membandingkan empat model machine learning (Random Forest, Naive Bayes, SVM, dan XGBoost) pada dataset performa siswa dari Kaggle untuk memprediksi hasil belajar dengan presisi tinggi. Pendekatan ini mengintegrasikan keunggulan dari berbagai model untuk membandingkan dan menemukan algoritma terbaik dalam memprediksi performa siswa berdasarkan dataset *students-performance-dataset* yang diperoleh dari kaggel. Prediksi yang akurat ini diharapkan dapat memfasilitasi pengambilan keputusan yang lebih baik dalam merancang kurikulum dan memberikan dukungan personalisasi.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang tersebut, maka rincian permasalahan yang dapat dibahas pada proyek ini yakni:
1. Berapa durasi belajar mingguan (Study Time Weekly) yang paling efektif untuk meningkatkan GPA siswa?
2. Apakah frekuensi absensi siswa (Absences) berhubungan negatif dengan GPA?
3. Apakah keikutsertaan dalam bimbingan belajar (Tutoring) meningkatkan GPA siswa?
4. Apakah terdapat perbedaan GPA antara siswa laki-laki dan perempuan?
5. Bagaimana pengaruh partisipasi dalam kegiatan ekstrakurikuler (Sports, Music, Volunteering) terhadap GPA siswa?
6. Apakah dukungan orang tua (Parental Support) berhubungan dengan peningkatan GPA siswa?
7. Faktor apa yang paling berpengaruh terhadap GPA siswa berdasarkan seluruh fitur yang tersedia?
8. Model machine learning mana yang paling akurat dalam memprediksi GPA siswa?

### Goals
Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini.
1. Menemukan durasi belajar mingguan optimal untuk peningkatan GPA.
2. Menganalisis hubungan absensi dengan penurunan GPA.
3. Mengevaluasi dampak bimbingan belajar terhadap GPA.
4. Mengidentifikasi perbedaan GPA berdasarkan gender.
5. Menilai pengaruh aktivitas ekstrakurikuler terhadap GPA.
6. Menganalisis pengaruh dukungan orang tua terhadap GPA.
7. Menentukan faktor utama yang memengaruhi GPA siswa.
8. Menemukan model machine learning terbaik untuk prediksi GPA berdasarkan akurasi.
9. Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements
1. Melakukan EDA untuk menganalisis hubungan antar fitur seperti Study Time Weekly, Absences, Tutoring, Gender, Parental Support, dan Extracurricular Activities dengan GPA siswa.
2. Membangun model machine learning (XGBoost, SVM, Naive Bayes, Random Forest) untuk memprediksi GPA, lalu membandingkan performa masing-masing model menggunakan confusion matrix dan F1 score.
3. Mengidentifikasi fitur yang paling berpengaruh terhadap GPA berdasarkan hasil modeling.

## Data Understanding
Dataset yang digunakan untuk mempredisksi kinerja siswa diambil dari platform kaggle yang dapat diakses pada tautan (https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset) yang dipublikasikan oleh Rabie El Kharoua pada tanggal 13 Juni 2024. Kumpulan data ini berisi informasi lengkap tentang 2.392 siswa sekolah menengah, yang merinci demografi, kebiasaan belajar, keterlibatan orang tua, kegiatan ekstrakurikuler, dan prestasi akademik mereka. Variabel target, GradeClass, mengklasifikasikan nilai siswa ke dalam kategori yang berbeda, sehingga menyediakan kumpulan data yang kuat untuk penelitian pendidikan, pemodelan prediktif, dan analisis statistik. Dataset ini terdiri dari 1 file csv.
Infromasi dataset tersebut dapat dilihat pada gambar dibawah ini:

![informasi-data](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/informasi-dataset.png?raw=true)

Dari gambar yang ditampilkan, terdapat 12 variabel bertipe int64 dan 3 variabel bertipe fload64

### Deskripsi Variabel
Dataset ini memiliki 15 variabel dengan keterangan sebagai berikut.
Variabel | Keterangan
----------|----------
StudentID | Pengidentifikasi unik yang diberikan kepada setiap siswa (1001 hingga 3392)
Age | Usia siswa berkisar antara 15 hingga 18 tahun
Gender | Jenis kelamin siswa, di mana 0 mewakili Laki-laki dan 1 mewakili Perempuan
Ethnicity | Etnis siswa, dikodekan sebagai berikut: 0(Kaukasia), 1(Afrika Amerika), 2(Asia), 3(Lainnya)
ParentalEducation | Tingkat pendidikan orang tua, dikodekan sebagai berikut: 0(Tidak Ada), 1(Sekolah Menengah Atas), 2(Beberapa Perguruan Tinggi), 3(Sarjana), 4(Lebih Tinggi)
StudyTimeWeekly | Waktu belajar mingguan dalam jam, berkisar antara 0 hingga 20
Absences | Jumlah ketidakhadiran selama tahun ajaran, berkisar antara 0 hingga 30
Tutoring | Status bimbingan belajar, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
ParentalSupport | Tingkat dukungan orang tua, dikodekan sebagai berikut: 0(Tidak Ada), 1(Rendah), 2(Sedang), 3(Tinggi), 4(Sangat Tinggi)
Extracurricular | Partisipasi dalam kegiatan ekstrakurikuler, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
Sports | Partisipasi dalam olahraga, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
Music	| Partisipasi dalam kegiatan musik, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
Volunteering	| Partisipasi dalam kesukarelaan, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
GPA	| Nilai Rata-rata Poin pada skala 2,0 hingga 4,0
GradeClass | Klasifikasi nilai siswa berdasarkan IPK (0: 'A' (IPK >= 3,5)), (1: 'B' (3,0 <= IPK < 3,5)), (2: 'C' (2,5 <= IPK < 3,0)), (3: 'D' (2,0 <= IPK < 2,5)), (4: 'F' (IPK < 2,0))

### Menangani Missing Value dan Duplicate Data (Duplikasi Data)
Pada tahap ini kita akan mengecek data yang tidak valid pada dataset. Setelah diperiksa apakah terdapat kolom yang bernilai null, hasilnya adalah tidak ada. Sedangkan data duplikat atau data ganda juga tidak ada. Maka dengan demikian data siapa untuk dianalisis pada tahap selanjutnya.

### Konversi nilai numerik kategorikal ke objek(string)
Pada tahap ini, karena dataset kita tipe kategorikal sudah dalam bentuk nilai numerik, maka kita perlu membuat fungsi konversi nilai number kategorikal ke objek (string). Tujuan dari langkah ini yaitu untuk menampilkan label fitur visualisasi dalam proses analisis data dengan teknik Univariate EDA dan Multivariate EDA.

### Univariate Analysis EDA
Ada beberapa tahap yang akan kita lakukan, yakni:
Tahap pertama, membagi variabel-variabel menjadi 2 jenis, yaitu variabel numerikal dan variabel kategorikal. Berikut merupakan kolom-kolom yang termasuk dalam variabel numerikal maupun kategorikal. <br>
Semua numerikal: ["Age", "StudyTimeWeekly", "Absences", "GPA"] <br>
Semua kategorikal: ["Gender", "Ethnicity", "ParentalEducation", "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering", "GradeClass"]

Tahap kedua, kita akan melihat nilai berbeda pada kolom kategorikal pada gambar tabel dibawah ini:

![univariate-kategorikal](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/univariate-kategorikal.png?raw=true)><br>

Dapat dilihat pada tabel nilai berbeda pada:
1. Kolom gender = 2;
2. Kolom Etnicity = 4;
3. Kolom ParentalEducation = 5;
4. Kolom Tutoring = 2;
5. Kolom ParentalSupport = 5;
6. Kolom Extracurricular = 2;
7. Kolom Sports = 2;
8. Kolom Music = 2;
9. Kolom Volunteering = 2;
10. Kolom GradeClass(Variabel Target) = 5;
    
Tahap ketiga, Pada tahap ini, kita akan membuat visualisasi data kategorikal dalam bentuk grafik dengan menggunakan library python matplotlib. Hasilnya seperti gambar dibawah ini:

![grafik-kategorikal](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/grafik-kategorikal.png?raw=true)<br>
Interpretasi:
1. Grafik jenis kelamin, menunjukan jumlah merata antara laki-laki dan perempuan.
2. Grafik Etnis, menunjukan mayoritas siswa berasal dari etnis kaukasia.
3. Grafisk pendidikan orangtua, menunjukan mayoritas pendidikan orang tua yakni pensisikan tinggi dan sarjana
4. Grafik bimbingan belajar, menunjukan mayoritas siswa tidak mengikuti bimbingan belajar.
5. Grafik dukungan orang tua, menunjukan mayoritas dukungan orang tua berada di level sdang dan tinggi.
6. Grafik Ekstrakulikuler(EKtrakulikuler, Olahraga, Musik dan Sukrelawan), menujukan rendahnya minat siswa pada kegiatan diluar sekolah.

Tahap keempat, kita akan membuat visualisasi data numerikal dalam bentuk grafik dengan menggunakan library python `matplotlib`. Hasilnya seperti gambar dibawah ini:

![visualisasi-numerikal](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/visualisasi-numerikal.png?raw=true)<br>
Interprestasi:
1. Pada kolom Age, dapat dilihat rata-rata siswa berumur 15-17 tahun. Dapat disimpulkan tidak ada Outlier yang tersebar.
2. Pada kolom StudyTimeWeekly, dapat dilihat bahwa rata-rata siswa memiliki waktu belajar 5-14 jam per minggu.
3. Pada kolom absences, dapat dilihat bahwa rata-rata siswa memiliki jumlah ketidakhadiran 6 - 23 hari. Dapat disimpulkan 1. Pada kolom
4. Pada kolom GPA, dapat dilihat bahwa rata-rata prestasi siswa diantara 1,2 - 2,7 dan tidak memiliki outlier.

Tahap yang kelima, kita akan melihat lebih detail mengenai jumlah dari masing-masing tingkat kelas terbaik yang menjadi target kita untuk mengetahui jumlah secara umum.

![data-kelas-terbaik](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/data-kelas-terbaik.png?raw=true)<br>
Interpretasi: 
Kelas terbaik (GradeClass) yang ditampilkan menununjukan mayoritas prestasi siswa berada di kategori Grade F(Prestasi terendah) yakni 1211 siswa sedangkan minioritas siswa berada pada kategori Grade A (Kelas tertinggi) yakni 107

Tahap keenam, kita akan melihat persebaran data dari masing-masing kategori kelas pada kolom GradeClass:

![distribusi-prestasi-siswa](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/distribusi-prestasi-siswa.png?raw=true)<br>
Interpretasi:
1. Siswa berada pada Grade F (kelas terendah) memiliki presentasi terbayak yakni 50.6%.
2. Siswa berada pada Grade A (kelas tertinggi) memiliki presentasi sedikit yakni 4.5%.
3. Siswa yang lainnya berada pada Grade B (11.2%), Grade C(16.3%) dan Grade D (17.3%)

Langkah terakhir, kita akan membentuk histogram dari variabel-variabel numerikal untuk melihat persebaran data:

![histogram-data](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/histogram-data.png?raw=true)<br>
Interpretasi: 
Usia, waktu belajar setiap minggu, absen dan nilai siswa cukup berdistribusi normal.

### Multivariate Analysis EDA

Pada bagian ini, akan ditunjukan hubungan antara dua variabel biasa disebut sebagai bivariate EDA. Selanjutnya, kita akan melakukan analisis data pada fitur kategori dan numerik.

#### 1. Ananlisis data pada fitur numerik StudyTimeWeekly (Waktu belajar setiap minggu) dengan GPA (Nilai Prestasi)

![analisis-fitur-numerik](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/analisis-fitur-numerik-1.png?raw=true)<br>
Interpretasi:
Siswa yang waktu belajaranya banyak mempengaruhi naiknya prestasi belajar(GPA).

#### 2. Ananlisis data pada fitur numerik Absences (Ketidakhadiran) dengan GPA (Nilai Prestasi)

![analisis-fitur-numerik](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/analisis-fitur-numerik-2.png?raw=true)<br>
Interpretasi:
Absen(ketidakhadiran) siswa sangat mempengaruhi turun prestasinya(GPA).

#### 3. Ananlisis data pada fitur kategori Tutoring (Bimbingan Belajar) dengan GradeClass (Kategori Kelas)

![histogram-bimbingan-belajar](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/histogram-bimbingan-prestasi.png?raw=true)<br>
Interpretasi:
Banyak siswa yang tidak mengikuti bimbingan belajar yang mendapat prestasi rendah (Grade F)

#### 4. Ananlisis data pada fitur kategori Gender (Jenis Kelamin) dengan GradeClass (Kategori Kelas)

![gender-prestasi](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/histogram-gender-prestasi.png?raw=true)<br>
Interpretasi:
Jenis kelamin pria lebih dominan memiliki prestasi lebih tinggi dibandingkan dengan wanita

#### 5. Ananlisis data pada fitur kategori kegiatan non akademik (Extracurricular, Sports, Music, Volunteering) dengan GPA (Nilai Prestasi)

![eskul-prestasi](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/histogram-eskul-prestasi.png?raw=true)<br>
Interpretasi:
Lebih banyak siswa yang tidak mengikuti kegiatan ekstrakulikuler, olahraga dan musik mempengaruhi turunya nilai pretasi(GPA) mereka

#### 6. Ananlisis data pada fitur kategori ParentalSupport (Dukungan Orang Tua) dengan GradeClass (Kategori Kelas)

![dukungan-orangtua](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/histogram-dukunganortu-prestasi.png?raw=true)<br>
Interpretasi:
Mayoritas dukungan orang tua sangat mempengaruhin nilai prestasi siswa (GPA). Semakin tinggi dukungan orang tua, maka semakin meningkat nilai prestasi dari anaknya.

#### 7. Melihat Korelasi Variabel dengan Menggunakan Heatmap

![heatmap-prestasi](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/heatmap-prestasi-siswa.png?raw=true)<br>
Interpretasi:

Nilai Prestasi Siswa memiliki
1. Korelasi negatif yang cukup kuat dengan ketidakhadiran(Absences).
2. Korelasi positif yang cukup lemah dengan waktu belajar setiap minggu(StudyTimeWeekly).

#### 8. Melihat Plot Scatter yang Memiliki Nilai Korelasi Positif dan Negatif

![plot-scatter-korelasi](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/plot-scatter-nilai-korelasi.png?raw=true)<br>
Interpretasi:

Nilai prestasi siswa (GPA) memiliki  korelasi negatif yang kuat pada ketidakhadiran (garis regresi menurun ke kanan bawah) dan korelatif positif cukup lemah pada waktu belajar setiap minggu (garis regresi naik ke kanan atas)

## Data Data Preparation

Pada tahap ini kita akan melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahap persiapan data perlu dilakukan, yaitu:
1. Drop kolom yang tidak digunakan dalam pemrosesan data
2. Encoding fitur kategori
3. Pembagian dataset dengan fungsi train_test_split dari library sklearn.

### Drop kolom yang tidak digunakan dalam pemrosesan data

Pada tahap ini ada beberapa kolom pada dataset yang tidak perlu digunakan dalam pemrosesan data yakni `StudentID`, `Ethnicity` dan `ParentalEducation`. Kolom-kolom ini akan dihapus menggunakan fungsi `drop()`. Hasilnya dapat dilihat pada gambar dibawah ini:

![data-preparation](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/drop-kolom-unused.png?raw=true)<br>
Hasilnya menampilkan variabel kolom `StudentID`, `Ethnicity` dan `ParentalEducation` sudah terhapus. Dataset siswa yang akan kita proses saat ini terdiri dari 2 tipe data float64, 2 tipe data int64 dan 8 tipe data objek(string).

### Encoding Fitur Kategori

Pada bagian ini, karena dataset fitur kategori kita sebelumnya sudah diubah dalam bentuk objek (string) pada tahap eksplorasi data analis maka kita perlu mengubah data kategori (yang berbentuk teks atau label) menjadi format numerik agar dapat diproses oleh algoritma machine learning. Encoding Fitur Kategorikal dilakukan 3 bagian, yakni:

1. *Label Encoding* yaitu, mengonversi nilai kategori menjadi angka integer (`0` dan `1`)). Variabel yang akan diproses yakni:  <br>
    a. *Tutoring* (Apakah siswa mengikuti bimbingan belajar?) <br>
    b. *Extracurricular* (Apakah siswa mengikuti kegiatan ektrakulikuler?) <br>
    c. *Sports* (Apakah siswa mengikuti kegiatan olahraga? <br>
    d. *Music* (Apakah siswa mengikuti kegiatan musik?) <br>
    e. *Volunteering* (Apakah siswa mengikuti kegiatan sukarelaan?)
2. *One Hot Ecoding* yaitu mengubah setiap kategori menjadi kolom biner terpisah untuk data tidak terurut). Variabel yang akan diproses yakni Gender.
3. *Ordinal Encoding* yaitu memberikan nilai integer berdasarkan hierarki atau urutan kategori). Variabel yang akan diproses yakni ParentalSupport.
Hasil setelah dilakukan data preprocessing dapat dilihat pada gambar berikut:

![data-encoding](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/data-encoding.png?raw=true)<br>

### Train-Test-Split
Langkah awal kita mengubah data objek ke data numeri dengan memanggil fungsi konversi objek to numerik. Selanjutnya, karena target kita adalah variabel GradeClass untuk mengetahui akurasi prediksi dari kategori kelas prestasi terbaik, maka kita akan membuang kolom tersebut dari data dan assign kolom tersebut ke variabel baru. Data training digunakan untuk melatih model dengan data yang ada, sedangkan data testing digunakan untuk menguji model yang dibuat menggunakan data yang belum dilatih. Pembagian data ini dilakukan dengan perbandingan 80% : 20% untuk data training dan data testing menggunakan train_test_split dari library sklearn. Berikut adalah data traning yang akan diproses (ditampilkan contoh 5 baris teratas):

![data-train-test](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/data-training.png?raw=true)<br>

Kemudian, kita melihat jumlah masing-masing *GradeClass* (Kategori Kelas) pada data testing untuk selanjutnya ditransformasikan menggunakan `LabelEncoder()`. `LabelEncoder()` berfungsi untuk memetakan setiap kategori unik dalam kolom *GradeClass* menjadi angka integer mulai dari `0`

## Modeling

Pada bagian ini, kita akan membangun 4 model machine learning untuk menguji sebarapa baik akurasi model, sehingga model tersebut yang disarankan untuk memprediksi prestasi siswa.

### 1. Model Development dengan Random Forest

Algoritma pembelajaran ensemble yang sangat populer untuk tugas klasifikasi dan regresi. Ini bekerja dengan membuat sejumlah pohon keputusan selama pelatihan dan menggabungkan hasilnya (melalui voting untuk klasifikasi atau rata-rata untuk regresi) untuk meningkatkan akurasi dan mengurangi overfitting.. <br>
    
Pada pemodelan ini, *Random Forest* diimplementasikan menggunakan `RandomForestClassifier` dari library `sklearn.ensemble` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `n_estimators` yaitu jumlah tree yang akan dibuat, `criterion` yaitu fungsi untuk menentukan kualitas *splitting data*, `max_depth` yaitu kedalaman maksimum setiap tree, dan `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi. Pada proyek ini, parameter yang digunakan adalah `n_estimators = 200`, `criterion = "entropy"`, `max_depth = 10`, `random_state = 50`.

### 2. Model Development dengan Extreme Gradient Boosting (XGBoost)

Algoritma Extreme Gradient Boosting merupakan salah satu algoritma boosting yang sangat kuat untuk tugas klasifikasi dan regresi. XGBoost dirancang untuk efisiensi, fleksibilitas, dan performa tinggi, serta sering digunakan dalam kompetisi machine learning. <br>

Pada pemodelan ini, XGBoost diimplementasikan menggunakan `XGBClassifier` dari library `xgboost` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `max_depth` yaitu kedalaman maksimum setiap tree, `n_estimators` yaitu jumlah tree yang akan dibuat, `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi, `learning rate` yaitu mengatur langkah setiap iterasi ketika meminimumkan *loss function*, dan `n_jobs` yaitu mengatur jumlah CPU threads untuk menjalankan XGBoost. Pada proyek ini, parameter yang digunakan adalah `max_depth = `6`, `n_estimators = 125`, `random_state = 30`, `learning_rate = 0.01`, `n_jobs = -1`.

### 3. Model Development dengan Support Vector Machine (SVM)

Algoritman ini sangat efektif untuk klasifikasi dan regresi. SVM bekerja dengan mencari hyperplane optimal yang memisahkan data dalam ruang fitur, serta mendukung kernel untuk menangani data non-linear. <br>

Pada pemodelan ini, SVM diimplementasikan menggunakan `SVC` dari library `sklearn.svm` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `kernel` yaitu tipe kernel yang digunakan untuk mentransformasikan input data, `gamma` yaitu pengaruh dari sebuah contoh training, dan `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi. Pada proyek ini, parameter yang digunakan adalah `kernel = 'rbf'`, `gamma = 'auto'`, `random_state = 50`.

### 4. Model Development dengan Naive Bayes

Algoritman ini merupakan algoritma klasifikasi berbasis probabilistik yang didasarkan pada Teorema Bayes. Algoritma ini bekerja dengan asumsi bahwa semua fitur saling independen (meskipun dalam kenyataan sering tidak sepenuhnya demikian). <br>

Pada pemodelan ini, Naive Bayes diimplementasikan menggunakan `GaussianNB` dari library `sklearn.naive_bayes` karena datanya numerik dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter `var_smoothing` berfungsi menambahkan nilai kecil (`var_smoothing`) ke varians dari setiap fitur. Sedangkan Nilai `1e-9` adalah representasi ilmiah untuk angka `0.000000001` (atau `10⁻⁹`). Ini digunakan untuk menambahkan nilai kecil pada varians, sehingga tidak ada nilai varians yang terlalu kecil untuk menghasilkan masalah numerik.

## Evaluation

Pada proyek ini, penilaian model menggunakan confusion matrix, akurasi, dan f1 score sebagai metrik evaluasi untuk masing-masing model. Akan dijelaskan terlebih dahulu bagaimana cara mendapatkan akurasi dan f1 score serta bagaimana cara menggunakan confusion matrix.

### Matriks Confusion, Akurasi, dan Skor f1

1. Matriks Confusion merupakan matriks yang menunjukkan jumlah prediksi benar dan salah untuk setiap kelas. Contoh dari Matriks Confusion beserta labelnya dapat dilihat pada gambar di bawah ini.

<img src="https://github.com/user-attachments/assets/0b200762-9005-4765-9924-8076faf96046" align="center"><br>
Formatnya:
[[TP, FP],
 [FN, TN]]

Terdapat 4 label pada matriks confusion seperti yang terlihat di gambar, yaitu TP, TN, FP, dan FN.
    a. *True Positive* (TP) merupakan jumlah data pada positif yang ditebak dengan benar.
    b. *True Negative* (TN) merupakan jumlah data pada negatif yang ditebak dengan benar.
    c. *False Positive* (FP) merupakan jumlah data yang ditebak dengan salah karena diprediksi positif, sedangkan aslinya adalah negatif.
    d. *False Negative* (FN) merupakan jumlah data yang ditebak dengan salah karena diprediksi negatif, sedangkan aslinya adalah positif.
    
2. Akurasi merupakan Persentase prediksi benar terhadap total prediksi.<br>
Formatnya:

<img src="https://github.com/user-attachments/assets/689a2934-4adb-42e8-b46a-e59f2e6b0508" align="center"><br>
4. Skor F1 merupakan rata-rata harmonik dari precision dan recall.
Formatnya:

<img src="https://github.com/user-attachments/assets/de176d91-a6b6-40a7-adc4-dd0d755eaa16" align="center"><br>
5. Precision merupakan proporsi prediksi positif yang benar-benar benar.<br>
Rumusnya:

<img src="https://github.com/user-attachments/assets/12b1ad68-e216-4bde-bce7-cfea6652e7e7" align="center"><br>
*Contoh*: Jika model memprediksi 10 data sebagai positif, tetapi hanya 7 yang benar-benar positif, maka precision adalah 7/10 = 0.7.

5. Recall (Sensitivity) merupakan proporsi data positif yang terdeteksi dengan benar oleh model.<br>
Rumusnya:

<img src="https://github.com/user-attachments/assets/2f6d9e5d-bd59-4999-84ae-9e85146e385c" align="center"><br>
*Contoh*: Jika model memprediksi 10 data sebagai positif, tetapi hanya 7 yang benar-benar positif, maka precision adalah 7/10 = 0.7.
 
### Penerapan Matriks Confusion, Akurasi, dan Skor f1
Berikut merupakan matriks confusion, akurasi, dan skor f1 dari seluruh model
![confusion-matrix](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/heatmap-confusion-matrix.png?raw=true)<br>

#### 1. Model Development dengan Random Forest
![performance-random-forest](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/performance-random-forest.png?raw=true)
Dari gambar di atas, model Random Forest menunjukkan **akurasi keseluruhan yang sangat tinggi, yaitu 93%**, menandakan kemampuannya yang kuat dalam memprediksi _grade_ siswa secara tepat.
Secara spesifik, model ini **sangat unggul dalam mengidentifikasi siswa dengan _grade_ menengah hingga rendah (Grade C, D, dan terutama Grade F)**. Untuk Grade F, model mencapai _recall_ 98% dan _precision_ 94%, yang berarti hampir semua siswa yang gagal berhasil terdeteksi dan sebagian besar prediksi "gagal" adalah benar. Ini sangat berharga untuk intervensi dini.
Namun, **kelemahan signifikan terletak pada prediksi untuk Grade A (_grade_ tertinggi)**. Dengan _recall_ hanya 62%, model seringkali **gagal mengidentifikasi siswa yang seharusnya masuk Grade A**, meskipun _precision_ (87%) cukup baik saat memprediksi Grade A. Hal ini kemungkinan besar disebabkan oleh **jumlah _sample_ Grade A yang sangat sedikit** (hanya 21), yang membuat model kesulitan belajar pola yang spesifik untuk kelas ini.

#### 2. Model Development dengan XGBoots
![performance-xgboost](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/performance-xgboost.png?raw=true)
Dari gambar di atas, model XGBoost Anda menunjukkan **akurasi keseluruhan yang sangat tinggi sebesar 93%**, setara dengan Random Forest dan jauh melampaui SVM. Keunggulan utama XGBoost adalah **keseimbangan kinerja di semua _grade_**. Model ini tidak hanya **sangat efektif dalam mendeteksi siswa yang berisiko gagal (Grade F)** dengan _recall_ 98%, tetapi juga menunjukkan **peningkatan signifikan dalam mengidentifikasi siswa berprestasi tinggi (Grade A)**, mencapai _recall_ 76%. ** XGBoost adalah model dengan kinerja terbaik** yang diuji sejauh ini. Model inimemberikan **prediksi yang akurat dan seimbang untuk semua kategori _grade_ siswa**.

#### 3. Model Model Development dengan SVM
![performance-svm](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/performance-svm.png?raw=true)
Dari gambar di atas, model SVM Anda mencapai **akurasi 78%**, yang secara signifikan **lebih rendah dibandingkan Random Forest (93%)** pada _dataset_ yang sama. Kinerja terbaik SVM terletak pada **prediksi siswa yang berisiko gagal (Grade F)**, dengan _recall_ 94% dan _precision_ 89%, menunjukkan efektivitas tinggi dalam mendeteksi kelompok ini. Namun, model ini **sangat buruk dalam mengidentifikasi siswa berprestasi tinggi (Grade A)**, hanya mampu mendeteksi 24% dari data(_recall_ 0.24). Kinerja pada _grade_ lainnya (B, C, D) berada di tingkat moderat. Meskipun SVM bagus untuk mendeteksi kegagalan kinerja siswa, **kinerja keseluruhannya jauh di bawah Random Forest**, terutama karena **kelemahannya dalam memprediksi Grade A**. Ini menyiratkan SVM **kurang cocok** untuk tujuan proyek Anda yang membutuhkan prediksi akurat untuk semua _grade_, kecuali jika fokus utamanya hanya pada deteksi dini siswa berisiko.

#### 4. Model Model Development dengan Naive Bayes
![performance-naive-bayes](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/performance-naive-bayes.png?raw=true)
Dari gambar di atas, model Naive Bayes Anda menunjukkan **akurasi 79%**, lebih rendah dari Random Forest dan XGBoost, tapi sedikit lebih baik dari SVM.
Kekuatan model ini adalah kemampuannya yang **sangat baik dalam mendeteksi siswa yang berisiko gagal (Grade F)**, dengan _recall_ 90%. Namun model ini memiliki kekurangan karena  **tidak mampu mengidentifikasi siswa berprestasi tinggi (Grade A)**, di mana model hanya berhasil mendeteksi 10% dari grade A (_recall_ 0.10). Model Naive Bayes **kurang cocok** untuk **memprediksi Grade A**, meskipun cukup baik untukmemprediksi  Grade F. Model ini tidak memberikan prediksi yang seimbang dan komprehensif untuk semua _grade_.

### Hasil Evaluasi
Dari seluruh akurasi yang diketahui dari keempat model, dibentuk bar plot untuk melihat perbandingan nilai akurasi model sebagai berikut. 

![evaluasi-model](https://github.com/sang-ichal/Submission_Predictive_Analytics/blob/main/image/result-akurasi-model.png?raw=true)><br>
Berdasarkan gambar di atas dan evaluasi masing-masing model untuk mengetahui skor akurasi, skor F1, dan jumlah kesalahan klasifikasi pada masing-masing model, didapat model *XGBoots* merupakan model terbaik karena memiliki skor akurasi dan skor F1 tertinggi, serta jumlah kesalahan klasifikasi yang paling sedikit, terutama pada Grade A. 

## Kesimpulan
Berdasarkan hasil yang diperoleh setelah melakukan proses EDA dan pengujiaan model terbaik untuk peningkatan prestasi siswa dapat dismpulkan bahwah:
1. Terdapat hubungan positif antara durasi belajar mingguan yang lebih tinggi dengan performa akademik (GPA dan GradeClass), namun tidak terlalu signifikan kenaikannya. Oleh karena itu, direkomendasikan waktu belajar berada diatas 20 jam per minggu.
2. Tingkat absensi yang tinggi secara konsisten menunjukkan korelasi negatif dengan performa akademik. Siswa dengan absensi tinggi cenderung memiliki GPA lebih rendah. Oleh karena iu, perlu diidentifikasi siswa dengan pola absensi tinggi untuk intervensi dini.
3. Siswa yang tidak mengikuti bimbingan belajar menunjukkan GPA yang lebih rendah dibandingkan mereka yang mengikuti. Olehkarena itu siswa perlu disiplin mengikuti bimbingan belajar.
4. Jenis kelamin yang berbeda memiliki prestasi yang tidak jauh berbeda. Dari data yang diperoleh, mayoritas perempuan mengalami penurunan prestasi pada Grade F, Grade D, Grade B dan Grade A dibandingkan dengan laki-laki.
5. Terjadi penurunan sedikit nilai pretasinya(GPA) siswa yang tidak mengikuti kegiatan ekstrakulikuler, olahraga dan musik. Sedangkan untuk kegiatan sukarelaan terlihat merata atau seimbang. 
6. Siswa yang menerima dukungan orang tua memiliki nilai rata-rata lebih tinggi dibandingkan yang tidak.
7. *Exploratory Data Analysis* (EDA) menunjukkan bahwa performa akademik siswa dipengaruhi oleh kombinasi faktor internal (seperti durasi belajar dan pola absensi) serta faktor eksternal (seperti keterlibatan orang tua dan partisipasi dalam kegiatan non-akademik)
8. Setelah menguji data menggunakan 4 model *machine learning*, yaitu ***Extreme Gradient Boosting* (XGBoost)**, ***Support Vector Machine* (SVM)**, ***Naive Bayes*** dan ***Random Forest*** untuk memprediksi performa siswa, diperoleh:
    * ***XGBoost*** adalah model terbaik untuk memprediksi performa siswa pada dataset ini, dengan akurasi dan F1-Score tertinggi.
    * ***Random Forest*** memberikan hasil yang hampir setara dengan XGBoost dan lebih mudah diimplementasikan.
    * ***SVM*** memberikan performa baik tetapi memerlukan penyesuaian parameter untuk hasil optimal.
    * ***Naive Bayes*** adalah model tercepat namun memiliki performa yang jauh lebih rendah karena asumsi independensi antar fitur yang tidak sesuai dengan dataset.

## Referensi
1. Nurrohman, Aji. "Klasifikasi Performa Akademik Siswa Menggunakan Metode Decision Tree dan Naive Bayes", Vol. 13 No.1 (2023) 22-31, ISSN 2503-3247. SINTA Peringkat 4, diakses pada 28 November 2024.
2. Dicoding. Diakses pada 6 Juli 2024 dari https://www.dicoding.com/academies/319-machine-learning-terapan
