import streamlit as st
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

from model import load_model
from data import preprocess_image

# Page config
st.set_page_config(
    page_title="Görüntü Sınıflandırıcı",
    page_icon="🔍",
    layout="centered"
)

# Title and description
st.title('Görüntü Sınıflandırıcı')
st.markdown("""
Bu uygulama, yüklediğiniz görüntüleri CIFAR-10 veri setine göre sınıflandırır. 
Desteklenen sınıflar: uçak, araba, kuş, kedi, geyik, köpek, kurbağa, at, gemi, ve kamyon.
""")

# Function to load the model
@st.cache_resource
def get_model():
    model_path = 'cifar10_model.pth'
    
    # Check if model exists, if not, inform user to train first
    if not os.path.exists(model_path):
        st.error("Model dosyası bulunamadı. Lütfen önce 'train.py' dosyasını çalıştırarak modeli eğitin.")
        st.stop()
        
    return load_model(model_path)

# Class names
class_names = ('uçak', 'araba', 'kuş', 'kedi', 'geyik', 'köpek', 'kurbağa', 'at', 'gemi', 'kamyon')

# Load model
try:
    model = get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Bir görüntü yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklenen görüntü', use_container_width=True)
    
    # Add a predict button
    predict_button = st.button('Tahmin Et')
    
    if predict_button:
        try:
            with st.spinner("Görüntü işleniyor ve tahmin yapılıyor..."):
                # Save the uploaded image temporarily
                temp_path = "temp_image.jpg"
                image.save(temp_path)
                
                # Preprocess the image and make prediction
                processed_image = preprocess_image(temp_path)
                processed_image = processed_image.to(device)
                
                # Display the preprocessed image
                plt.figure(figsize=(3, 3))
                plt.imshow(np.transpose(processed_image[0].cpu().numpy(), (1, 2, 0)))
                plt.axis('off')
                plt.title('İşlenmiş görüntü (32x32)')
                
                # Save the figure to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Display the preprocessed image
                st.image(buf, caption='İşlenmiş görüntü (32x32)', width=150)
                
                # Make prediction
                model.eval()
                with torch.no_grad():
                    outputs = model(processed_image)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
                
                # Display results
                st.subheader("Tahmin Sonucu:")
                st.markdown(f"<h1 style='text-align: center; color: #1e88e5;'>{class_names[predicted_class]}</h1>", unsafe_allow_html=True)
                
                # Display probability bar chart
                st.subheader("Sınıf Olasılıkları:")
                probs_data = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                
                # Sort probabilities for better visualization
                sorted_probs = dict(sorted(probs_data.items(), key=lambda x: x[1], reverse=True))
                
                # Display as bar chart
                st.bar_chart(sorted_probs)
                
                # Clean up
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"Tahmin yapılırken bir hata oluştu: {e}")

# Add information about the model
with st.expander("Model Hakkında Bilgi"):
    st.markdown("""
    ### Model Mimarisi
    Bu uygulamada kullanılan model, aşağıdaki mimariyi içeren özel bir CNN modelidir:
    
    - 3 konvolüsyonel katman (32, 64, ve 128 filtre)
    - Batch normalizasyon
    - MaxPooling
    - Dropout (0.25) düzenleştirme
    - 512 nöronlu tam bağlantılı katman
    - 10 sınıflı çıkış katmanı
    
    ### CIFAR-10 Veri Seti
    Model, 60,000 renkli görüntü (32x32 piksel) içeren CIFAR-10 veri seti üzerinde eğitilmiştir. 
    Bu veri seti, 10 farklı sınıfa ait 50,000 eğitim ve 10,000 test görüntüsü içerir.
    """)

# Add footer
st.markdown("---")
st.markdown("Bu uygulama PyTorch ve Streamlit kullanılarak geliştirilmiştir.") 