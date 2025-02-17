# Custom Intel Image Classification Projesi

## 📌 Proje İçeriği
- **Dataset**: Custom Intel Image dataset kullanıldı.
- **Model**: 7 farklı label üzerinden sıfırdan bir sınıflandırma modeli oluşturuldu, eğitildi ve test edildi.
- **Tahminleme**: Model, yeni veriler üzerinde tahmin yapmak için kullanıldı.

## 🚀 Kullanılan Teknolojiler
- **Flask**: Modeli bir web uygulaması üzerinden kullanmak için Flask tabanlı bir API geliştirildi.
- **MLflow**: Model eğitim sürecini takip etmek ve kayıt altına almak için MLflow kullanıldı.
- **DVC**: Veri ve model versiyonlaması için DVC (Data Version Control) kullanıldı.
- **Dagshub**: MLflow ve DVC, Dagshub reposuna entegre edildi.

## 📊 Model Sonuçları
- Modelin performans metrikleri ve eğitim sürecine ait detaylar **MLflow görselleri** ile kayıt altına alındı.
- MLflow görselleri proje içerisine eklenmiştir.

## 📌 Kullanım
1. Gerekli bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt

2. MLflow URI: Kendi local MLflow URI'nizi belirlemek için, aşağıdaki gibi ayar yapmalısınız:
   - Local için: http://localhost:5000
   - Dagshub için uygun URL’yi düzenleyin.

3. Flask uygulamasını çalıştırın:
    ```cmd
    python app.py

4. Uygulama: Flask uygulaması, localhost:8000 adresinde çalışacaktır. Bu adrese giderek modelin tahminleme işlemlerini yapabilirsiniz.


