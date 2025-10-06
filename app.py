import streamlit as st
import gdown #baixar modelo do google drive
import tensorflow as tf
import io
from PIL import Image
import numpy as np #converter a imagem para um formato que o TensorFlow consiga entender
import pandas as pd
import plotly.express as px


@st.cache_resource #armazena o modelo em cache evitando downloads repetidos
def carrega_modelo():
    url = ''

    gdown.download(url,'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem ou clique para selecionar', type=['png','jpg','jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)) #pega os dados em forma de binaria passando para o formato de imagem

        st.image(image)
        st.success('Imagem carregada com sucesso')

        #pré processamento da imagem e retorno
        image = np.array(image,dtype=np.float32)
        image = image/ 255.0
        image = np.expand_dms(image,axis=0)

        return image

def previsao(interpreter,image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'],image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles', 'BlackRot','HealthyGrapes','LeafBlight']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df,y='classes', x = 'probalidades (%)', orientation = 'h', text='probalidades (%)',
                 title='Probabilidade de Classes de Doenças de Uvas')
    st.plotly_chart(fig)

def main():
    
    st.set_page_config(
        page_title="Classifica Folhas Videiras"
    )

    #carrega modelo
    interpreter = carrega_modelo()
    
    #carrega imagem
    

    #classifica
    if image is not None:
        previsao(interpreter,image)

if __name__=="__main__":
    main()