# 1. Temel imaj olarak Python 3.8'in resmi slim sürümünü kullanıyoruz.
FROM python:3.8-slim

# 3. Çalışma dizinimizi /app olarak belirliyoruz.
WORKDIR /app

# Proje dosyalarının workdire kopyalanması
COPY . /app

# 5. pip güncellemesi ve proje bağımlılıklarının kurulması.
RUN pip install --upgrade pip && pip install -r requirements.txt

# 7. Model eğitim script'ini çalıştırmak için varsayılan komut.
CMD ["python", "app.py"]

