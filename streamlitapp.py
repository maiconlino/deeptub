import pandas as pd
import numpy as np
import pickle 
import streamlit as st
import lime
import lime.lime_tabular
from pickle import load
from PIL import Image
img_alerta = Image.open('alerta.png')


loaded_model = pickle.load(open('SVM03-11-2022_02-50-37.sav','rb'))
scalerfile = 'scaler2_03-19-2022_02-19-47.pkl'
scaler = load(open(scalerfile, 'rb'))
X_train = pd.read_csv('X_trainLIME_03-19-2022_03-59-34.csv', sep=';', index_col=False)
Y_train = pd.read_csv('Y_trainLIME_03-19-2022_03-59-34.csv', sep=';', index_col=False)


colunas = ('NU_IDADE_N','TRATAMENTO','RAIOX_TORA','TESTE_TUBE','FORMA','AGRAVDOENC','BACILOSC_E','BACILOS_E2','HIV','BACILOSC_6','DIAS')

def prognosis_tuberculosis(input_data):

    input_data_numpy = np.asarray(input_data)
    input_reshape = input_data_numpy.reshape(1,-1)
    #print(input_reshape)
    base_x = pd.DataFrame(input_reshape, columns=colunas)
    #print(base_x.head())
    
    test_scaled_set = scaler.transform(base_x)
    test_scaled_set = pd.DataFrame(test_scaled_set, columns=colunas)
    #print(test_scaled_set.head())

    class_names = ["Cura","Óbito"]
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, training_labels=Y_train,class_names=class_names, feature_names=X_train.columns, kernel_width=3, discretize_continuous=True, verbose=False)
    exp = explainer.explain_instance(test_scaled_set.values[0], loaded_model.predict_proba, num_features=11)
    #exp.show_in_notebook()

    predictions = loaded_model.predict(test_scaled_set)
    predictions
    if(predictions[0]==3):
        #retorno = "A probabilidade de **óbito** no prognósitco da Tuberculose é de: {}%"
        #retorno = retorno.format(round(exp.predict_proba[1]*100,2))
        return (predictions[0],round(exp.predict_proba[1]*100,2))
    else:
        #retorno = "A probabilidade de **cura** no prognósitco da Tuberculose é de: {}%"
        #retorno = retorno.format(round(exp.predict_proba[0]*100,2))
        return (predictions[0],round(exp.predict_proba[0]*100,2))

def main():

    #title
    st.title("DeepTub ++ Tuberculosis prediction Web App")

   
    NU_IDADE_N = st.sidebar.slider(
        'Idade do paciente',
        0, 125, (30)
    )
    TRATAMENTO = st.sidebar.selectbox(
        'Tipo de Entrada do Paciente no tratamento',
        ('1. Caso novo', '2. Recidiva', '3. Reingresso após abandono', '4. Não sabe', '5.Transferência', '6. Pós-óbito')
    )
    RAIOX_TORA = st.sidebar.selectbox(
        'Radiografia do tórax',
        ('1. Suspeito', '2. Normal', '3. Outra patologia', '4. Não realizado')
    )
    TESTE_TUBE = st.sidebar.selectbox(
        'Teste tuberculíneo',
        ('1. Não reator', '2. Reator fraco', '3. Reator forte', '4. Não realizado')
    )
    FORMA = st.sidebar.selectbox(
        'Forma clínica da tuberculose',
        ('1. Pulmonar', '2. Extrapulmonar', '3. Pulmonar + Extrapulmonar')
    )
    AGRAVDOENC = st.sidebar.selectbox(
        'Agravos associados a Doença Mental',
        ('1. Sim', '2. Não', '3. Ignorado')
    )
    BACILOSC_E = st.sidebar.selectbox(
        'Baciloscopia de escarro – 1ª amostra',
        ('1. Positiva', '2. Negativa', '3. Não realizada', '4. Não se aplica')
    )
    BACILOS_E2 = st.sidebar.selectbox(
        'Baciloscopia de escarro – 2ª amostra',
        ('1. Positiva', '2. Negativa', '3. Não realizada', '4. Não se aplica')
    )
    HIV = st.sidebar.selectbox(
        'Resultado da sorologia para o vírus da imunodeficiência HIV',
        ('1. Positivo', '2. Negativo', '3. Em andamento', '4. Não se aplica')
    )
    BACILOSC_6 = st.sidebar.selectbox(
        '11. Baciloscopia no 6º mês ',
        ('1. Positiva', '2. Negativa', '3. Não realizada', '4. Não se aplica')
    )
    DIAS = st.sidebar.slider(
        'Dias em que o paciente está em tratamento (desde o início do diagnóstico)',
        0, 1095, (60)
    )

    prognosis = ''
    if st.button('Clique para ver o resultado do prognóstico da tuberculose'):
        if TRATAMENTO=="1. Caso novo":
            TRATAMENTO = 1
        elif TRATAMENTO=="2. Recidiva":
            TRATAMENTO = 2
        elif TRATAMENTO=="3. Reingresso após abandono":
            TRATAMENTO = 3
        elif TRATAMENTO=="4. Não sabe":
            TRATAMENTO = 4
        elif TRATAMENTO=="5.Transferência":
            TRATAMENTO = 5
        elif TRATAMENTO=="6. Pós-óbito":
            TRATAMENTO = 6

        if RAIOX_TORA=="1. Suspeito":
            RAIOX_TORA = 1
        elif RAIOX_TORA=="2. Normal":
            RAIOX_TORA = 2
        elif RAIOX_TORA=="3. Outra patologia":
            RAIOX_TORA = 3
        elif RAIOX_TORA=="4. Não realizado":
            RAIOX_TORA = 4
        
        if TESTE_TUBE=="1. Não reator":
            TESTE_TUBE = 1
        elif TESTE_TUBE=="2. Reator fraco":
            TESTE_TUBE = 2
        elif TESTE_TUBE=="3. Reator forte":
            TESTE_TUBE = 3
        elif TESTE_TUBE=="4. Não realizado":
            TESTE_TUBE = 4
        
        if FORMA=="1. Pulmonar":
            FORMA = 1
        elif FORMA=="2. Extrapulmonar":
            FORMA = 2
        elif FORMA=="3. Pulmonar + Extrapulmonar":
            FORMA = 3

        if AGRAVDOENC=="1. Sim":
            AGRAVDOENC = 1
        elif AGRAVDOENC=="2. Não":
            AGRAVDOENC = 2
        elif AGRAVDOENC=="3. Ignorado":
            AGRAVDOENC = 3

        if BACILOSC_E=="1. Positiva":
            BACILOSC_E = 1
        elif BACILOSC_E=="2. Negativa":
            BACILOSC_E = 2
        elif BACILOSC_E=="3. Não realizada":
            BACILOSC_E = 3
        elif BACILOSC_E=="4. Não se aplica":
            BACILOSC_E = 4

        if BACILOS_E2=="1. Positiva":
            BACILOS_E2 = 1
        elif BACILOS_E2=="2. Negativa":
            BACILOS_E2 = 2
        elif BACILOS_E2=="3. Não realizada":
            BACILOS_E2 = 3
        elif BACILOS_E2=="4. Não se aplica":
            BACILOS_E2 = 4

        if HIV=="1. Positivo":
            HIV = 1
        elif HIV=="2. Negativo":
            HIV = 2
        elif HIV=="3. Em andamento":
            HIV = 3
        elif HIV=="4. Não se aplica":
            HIV = 4

        if BACILOSC_6=="1. Positiva":
            BACILOSC_6 = 1
        elif BACILOSC_6=="2. Negativa":
            BACILOSC_6 = 2
        elif BACILOSC_6=="3. Não realizada":
            BACILOSC_6 = 3
        elif BACILOSC_6=="4. Não se aplica":
            BACILOSC_6 = 4

        prognosis = prognosis_tuberculosis([NU_IDADE_N, TRATAMENTO, RAIOX_TORA, TESTE_TUBE, FORMA, AGRAVDOENC, BACILOSC_E, BACILOS_E2, HIV, BACILOSC_6, DIAS])
    
        if prognosis[0]==1:
            st.info('A probabilidade de **CURA** por tuberculose é de: {}'.format(prognosis[1])+'%')
        else:
            st.info('A probabilidade de **ÓBITO** por tuberculose é de: {}'.format(prognosis[1])+'%')
            st.image(img_alerta)


if __name__ == '__main__':
    main()