!pip install pyngrok
!ngrok authtoken 2oyL7yq36iuQopwiUgC1BvZ8Btb_2ra5gxQBfjwXT23cv2kS8

pip install streamlit torch torchvision transformers huggingface-hub

!pkill -f ngrok

from pyngrok import ngrok
import os

# Inicia Streamlit en segundo plano
os.system("streamlit run app.py &")

# Crea un túnel público con Ngrok
public_url = ngrok.connect(8501)  # Puerto 8501 es el predeterminado de Streamlit
print(f"Tu aplicación está disponible en: {public_url}")
