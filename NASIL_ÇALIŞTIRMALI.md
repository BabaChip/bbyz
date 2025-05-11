# Görüntü Sınıflandırıcı - Nasıl Çalıştırılır?

Bu kılavuz, görüntü sınıflandırıcı uygulamasının kurulumu ve çalıştırılması için adım adım talimatlar içerir.

## Gereksinimler

- Python 3.7 veya daha yeni sürüm
- pip (Python paket yöneticisi)
- İnternet bağlantısı (CIFAR-10 veri setini indirmek için)

## Kurulum Adımları

1. **Python Yükleme**

Eğer Python yüklü değilse, [python.org](https://www.python.org/downloads/) adresinden işletim sisteminize uygun sürümü indirip yükleyin.

2. **Projeyi İndirme**

Projeyi bilgisayarınıza kopyalayın.

3. **Gerekli Kütüphaneleri Yükleme**

Terminal veya komut istemcisinde proje klasörüne gidin ve aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

## Uygulamayı Çalıştırma

### 1. Modeli Eğitme

İlk olarak, modeli eğitmeniz gerekiyor. Bu adım, CIFAR-10 veri setini indirecek ve modeli eğitecektir:

```bash
python train.py
```

Bu işlem donanımınıza bağlı olarak 5-30 dakika sürebilir. Eğitim sırasında:
- Her epoch ve batch için ilerleme çubukları gösterilir
- Her epoch sonunda eğitim ve doğrulama kayıpları ile doğruluk oranı görüntülenir
- Eğitim tamamlandığında sonuç metrikleri ve karmaşıklık matrisi oluşturulur

Eğitim tamamlandığında aşağıdaki dosyalar oluşturulacaktır:
- `cifar10_model.pth` (eğitilmiş model)
- `sample_images.png` (örnek eğitim görselleri)
- `training_metrics.png` (eğitim metrikleri grafiği)
- `confusion_matrix.png` (karmaşıklık matrisi)

### 2. Web Uygulamasını Başlatma

Model eğitimi tamamlandıktan sonra, web uygulamasını başlatmak için:

```bash
streamlit run app.py
```

Bu komut, uygulamayı başlatacak ve tarayıcınızda otomatik olarak açılacaktır. Eğer otomatik açılmazsa, tarayıcınızda aşağıdaki adresi açın:

```
http://localhost:8501
```

### 3. Uygulamayı Kullanma

1. "Bir görüntü yükleyin..." bölümüne tıklayarak bir görsel yükleyin (JPG, JPEG veya PNG formatında).
2. Görsel yüklendikten sonra "Tahmin Et" butonuna tıklayın.
3. Model, görselinizi işleyecek ve hangi sınıfa ait olduğunu tahmin edecektir.
4. Sonuçlar, tahmin edilen sınıf ve tüm sınıflar için olasılık değerleri şeklinde gösterilecektir.

## Eğitim Sürecini İzleme

Eğitim sırasında, terminal ekranında model eğitim sürecini detaylı bir şekilde takip edebilirsiniz:

- **Genel İlerleme Çubuğu**: Toplam epoch sayısı üzerinden genel ilerleme durumunu gösterir.
- **Epoch İlerleme Çubuğu**: Her epoch içindeki batch'lerin işlenme durumunu gösterir.
- **Doğrulama İlerleme Çubuğu**: Her epoch sonundaki doğrulama sürecinin durumunu gösterir.
- **Metrik Bilgileri**: Her epoch sonunda eğitim kaybı, doğrulama kaybı, doğruluk oranı ve öğrenme oranı bilgileri gösterilir.

Bu sayede model eğitim sürecinin ne kadar süreceğini ve ne kadar ilerlediğini kolayca takip edebilirsiniz.

## Hata Giderme

- **ModuleNotFoundError**: Gerekli kütüphaneler eksikse, `pip install -r requirements.txt` komutunu tekrar çalıştırın.
- **CUDA hatası**: GPU ile eğitim sırasında hata alırsanız, `train.py` dosyasında eğitim kısmını CPU'ya zorlamak için kodu düzenleyebilirsiniz.
- **Model dosyası bulunamadı hatası**: Web uygulaması çalıştırmadan önce `train.py` dosyasını çalıştırdığınızdan emin olun.

## Demo Görüntüler

Model, CIFAR-10 veri setindeki görsellere benzer görüntülerde en iyi performansı gösterecektir. Aşağıdaki sınıflardan herhangi birine ait görselleri deneyebilirsiniz:

- Uçak
- Araba
- Kuş
- Kedi
- Geyik
- Köpek
- Kurbağa
- At
- Gemi
- Kamyon 