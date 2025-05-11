
# Yapay Zeka Destekli Görüntü Sınıflandırıcı

Bu proje, CIFAR-10 veri setinde eğitilmiş bir CNN modeli kullanarak görüntü sınıflandırma yapan bir uygulamadır. Sistem, kullanıcıların yüklediği görüntüleri sınıflandırarak hangi kategoriye (uçak, araba, kuş, kedi, geyik, köpek, kurbağa, at, gemi, kamyon) ait olduğunu tahmin eder.

## Proje Özellikleri

- PyTorch ile oluşturulmuş özel CNN mimarisi
- CIFAR-10 veri seti üzerinde eğitim
- Görüntü ön işleme (yeniden boyutlandırma, normalizasyon)
- Kullanıcı dostu Streamlit web arayüzü
- Model performans metrikleri (doğruluk, kesinlik, duyarlılık, F1 skoru)
- Görsel sonuç analizleri (eğitim/doğrulama grafikleri, karmaşıklık matrisi)

## Kurulum

1. Projeyi klonlayın veya indirin:

```bash
git clone https://github.com/kullanici/yapay-zeka-goruntu-siniflandirici.git
cd yapay-zeka-goruntu-siniflandirici
```

2. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Modeli Eğitme

Modeli eğitmek için aşağıdaki komutu çalıştırın:

```bash
python train.py
```

Bu işlem:
- CIFAR-10 veri setini indirecek
- Modeli eğitecek (varsayılan olarak 20 epoch)
- Eğitim metriklerini ve sonuçları grafik olarak kaydedecek
- Eğitilmiş modeli `cifar10_model.pth` olarak kaydedecek

> **Not:** Model eğitimi donanıma bağlı olarak 5-30 dakika sürebilir. GPU varsa otomatik olarak kullanılacaktır.

### 2. Web Arayüzünü Başlatma

Modeli eğittikten sonra web arayüzünü başlatmak için:

```bash
streamlit run app.py
```

Tarayıcınızda http://localhost:8501 adresine giderek uygulamayı kullanabilirsiniz.

### 3. Görüntü Sınıflandırma

1. Arayüzden "Bir görüntü yükleyin..." bölümüne tıklayarak bir görsel dosyası seçin (JPG, JPEG veya PNG).
2. "Tahmin Et" butonuna tıklayın.
3. Sonuçları görüntüleyin:
   - Tahmin edilen sınıf
   - İşlenmiş görüntü (32x32)
   - Sınıf olasılıkları

## Teknik Detaylar

### Model Mimarisi

```
CIFAR10CNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=2048, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=10, bias=True)
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

### Eğitim Parametreleri

- Optimizer: Adam
- Learning Rate: 0.001 (ReduceLROnPlateau ile düşürülür)
- Batch Size: 64
- Epochs: 20
- Loss Function: CrossEntropyLoss

## Proje Yapısı

```
.
├── README.md                   # Bu dosya
├── requirements.txt            # Gerekli kütüphaneler
├── data.py                     # Veri işleme fonksiyonları
├── model.py                    # CNN model tanımı
├── train.py                    # Model eğitim kodu
├── app.py                      # Streamlit web uygulaması
├── cifar10_model.pth           # Eğitilmiş model (eğitimden sonra oluşturulur)
├── sample_images.png           # Örnek eğitim görselleri
├── training_metrics.png        # Eğitim metrikleri grafiği
└── confusion_matrix.png        # Karmaşıklık matrisi
```


