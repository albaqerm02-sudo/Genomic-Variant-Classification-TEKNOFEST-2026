TEKNOFEST 2026 SAĞLIKTA YAPAY ZEKÂ YARIŞMASI
ÜNİVERSİTE VE ÜZERİ SEVİYESİ
MODEL ÇALIŞTIRMA VE YENİDEN ÜRETİLEBİLİRLİK BİLGİLERİ

Takım Adı: Biyoinformatikçiler
Takım ID: 882005
Başvuru ID: 4878966

============================================================
1. SİSTEM BİLGİSİ
============================================================

İşletim Sistemi: Windows
Python Sürümü: Python 3.13.15

Gerekli Python kütüphaneleri ve sürümleri
requirements.txt dosyasında belirtilmiştir.

Model çalıştırma ortamı proje klasörü içerisindeki
.venv klasöründe hazır bulunmaktadır.

============================================================
2. MODEL DOSYALARI
============================================================

Eğitilmiş model dosyaları aşağıdaki klasörde bulunmaktadır:

models/

Panel bazında model dosyaları:

models/MASTER/
models/CANCER/
models/PAH/
models/CFTR/

MASTER modeli için taşınabilir XGBoost model dosyası da
models/MASTER/ klasörü içerisinde bulunmaktadır.

============================================================
3. TEST VERİLERİNİN YERLEŞTİRİLMESİ
============================================================

Yarışma sırasında organizasyon tarafından sağlanan dört test
CSV dosyası aşağıdaki klasöre yerleştirilmelidir:

TEKNOFEST_4_DATASETS_NO_LABEL/

Test verileri aşağıdaki dört paneli temsil etmelidir:

MASTER
CANCER / KANSER
PAH
CFTR

NOT:
Yarışma test verileri bu model paketinin bir parçası değildir
ve USB teslim paketine dahil edilmemelidir.

============================================================
4. TEST DOSYASI İSİMLERİNİN TANIMLANMASI
============================================================

predict_final.py dosyasının üst kısmında bulunan aşağıdaki
dört değişken, organizasyon tarafından verilen gerçek test
dosyalarının isimlerine göre düzenlenmelidir:

MASTER_FILE = "..."
CANCER_FILE = "..."
PAH_FILE    = "..."
CFTR_FILE   = "..."

Bu dört dosya adı dışında model kodunda herhangi bir değişiklik
yapılmamalıdır.

============================================================
5. MODELİN ÇALIŞTIRILMASI
============================================================

Ana inference dosyası:

predict_final.py

VS Code kullanılıyorsa proje Python yorumlayıcısı olarak
aşağıdaki sanal ortam seçilmelidir:

.venv\Scripts\python.exe

Ardından predict_final.py dosyası çalıştırılmalıdır.

Alternatif olarak komut satırından:

.venv\Scripts\python.exe predict_final.py

============================================================
6. MODEL ÇALIŞMA AKIŞI
============================================================

Program sırasıyla:

1. Dört panel modelini yükler.
2. Test CSV dosyalarını okur.
3. Eğitim sırasında kaydedilen veri ön işleme adımlarını uygular.
4. Özellik seçimi ve ölçeklendirme işlemlerini uygular.
5. Her varyant için Patojenik sınıf olasılığını hesaplar.
6. Panel bazlı karar eşiğini uygular.
7. predicted_class ve predicted_prob değerlerini üretir.
8. Sonuçları TEKNOFEST JSON formatında kaydeder.

============================================================
7. ÇIKTI DOSYASI
============================================================

Model başarılı şekilde çalıştırıldığında sonuç dosyası proje
ana klasöründe oluşturulur:

TEAM_882005_FINAL.json

JSON dosyası UTF-8 formatında oluşturulur.

Dosya aşağıdaki bilgileri içerir:

team_name
team_id
application_id
competition_level
predictions

Her tahmin kaydı:

id
panel
predicted_class
predicted_prob

alanlarını içerir.

predicted_class yalnızca "0" veya "1" değerini alır.

predicted_prob değeri 0 ile 1 arasındadır ve
Patojenik sınıf (Sınıf 1) olasılığını temsil eder.

============================================================
8. HATA KONTROLLERİ
============================================================

Program aşağıdaki durumlarda işlemi durdurur:

- Test verisinde Label sütunu bulunması
- Variant_ID sütununun bulunmaması
- Eksik Variant_ID bulunması
- Aynı panel içerisinde tekrarlanan Variant_ID bulunması
- Beklenen özellik sütunlarının eksik olması
- Beklenmeyen ek özellik sütunlarının bulunması
- NaN veya Infinity tahmin üretilmesi
- Olasılık değerlerinin 0-1 aralığı dışında olması
- Üretilen tahmin sayısının giriş satırı sayısıyla eşleşmemesi

============================================================
9. SONUÇ
============================================================

Başarılı çalıştırma sonunda terminalde:

SUCCESS

mesajı görüntülenir ve aşağıdaki dosya oluşturulur:

TEAM_882005_FINAL.json