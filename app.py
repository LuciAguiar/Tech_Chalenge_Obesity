import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA (Deve ser a 1ª linha)
# ==========================================
st.set_page_config(page_title="Tech Challenge 4 - Obesidade", page_icon="🩺", layout="wide")

# ==========================================
# 2. INICIALIZAÇÃO E CARREGAMENTO INTELIGENTE (Auto-Treino se faltar arquivo)
# ==========================================
@st.cache_resource
def carregar_arquivos():
    nome_modelo = 'modelo_obesidade.pkl'
    nome_encoder = 'label_encoder.pkl'
    nome_colunas = 'colunas_modelo.pkl'
    
    # Se os três arquivos existirem, carrega direto (Início Instantâneo)
    if os.path.exists(nome_modelo) and os.path.exists(nome_encoder) and os.path.exists(nome_colunas):
        modelo = joblib.load(nome_modelo)
        le = joblib.load(nome_encoder)
        colunas = joblib.load(nome_colunas)
        return modelo, le, colunas
    
    # Fallback de Segurança: Se faltar algum arquivo, treina automaticamente na inicialização
    else:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Leitura com o tratamento de ponto e vírgula e encoding correto
        df = pd.read_csv('Obesity.csv', dtype=str, sep=';', encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # Aplicação da lógica de Winsorização/Clip homologada no Colab (73.50% acurácia)
        idade_num = pd.to_numeric(df['Age'].str.split('.').str[0], errors='coerce')
        df['Age'] = idade_num.clip(lower=14, upper=61).fillna(idade_num.median())

        def limpar_categorica_clip(coluna_nome, limite_inf, limite_sup):
            num = pd.to_numeric(df[coluna_nome], errors='coerce').round()
            num_clipado = num.clip(lower=limite_inf, upper=limite_sup)
            return num_clipado.fillna(num_clipado.mode()[0])

        df['FCVC'] = limpar_categorica_clip('FCVC', 1, 3)
        df['NCP']  = limpar_categorica_clip('NCP', 1, 4)
        df['CH2O'] = limpar_categorica_clip('CH2O', 1, 3)
        df['FAF']  = limpar_categorica_clip('FAF', 0, 3)   
        df['TUE']  = limpar_categorica_clip('TUE', 0, 2)
        
        colunas_inteiras = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for coluna in colunas_inteiras:
            df[coluna] = df[coluna].astype('Int64')

        df = df.fillna(df.mode().iloc[0]) 
        
        X = df.drop(['Obesity', 'Height', 'Weight'], axis=1)
        y = df['Obesity']
        
        X_num = pd.get_dummies(X, drop_first=True)
        le_novo = LabelEncoder()
        y_num = le_novo.fit_transform(y)
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(X_num, y_num, test_size=0.3, random_state=42)
        
        modelo_rf_novo = RandomForestClassifier(random_state=42)
        modelo_rf_novo.fit(X_treino, y_treino)
        
        # Salva os arquivos para os próximos carregamentos serem instantâneos
        joblib.dump(modelo_rf_novo, nome_modelo)
        joblib.dump(le_novo, nome_encoder)
        joblib.dump(list(X_treino.columns), nome_colunas)
        
        return modelo_rf_novo, le_novo, list(X_treino.columns)

@st.cache_data
def carregar_dados():
    df = pd.read_csv('Obesity.csv', sep=';', encoding='latin1')
    df.columns = df.columns.str.strip()
    return df

# Garante que as variáveis globais do modelo estejam prontas assim que o app abre
modelo_rf, le, colunas_modelo = carregar_arquivos()

# ==========================================
# 3. MENU LATERAL (SIDEBAR)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=100)
st.sidebar.title("Menu de Navegação")
opcao_menu = st.sidebar.radio(
    "Selecione a página:",
    ["🔮 Análise Preditiva", "📊 Fonte de Dados", "⚙️ Pipeline Machine Learning", "📖 Story Telling"]
)

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido para o Tech Challenge 4 - FIAP.")

# ==========================================
# PÁGINA 1: ANÁLISE PREDITIVA 
# ==========================================
if opcao_menu == "🔮 Análise Preditiva":
    
    map_genero = {'Feminino': 'Female', 'Masculino': 'Male'}
    map_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    map_freq = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
    map_transporte = {'Automóvel': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike', 'Transporte Público': 'Public_Transportation', 'A pé': 'Walking'}

    map_diagnostico = {
        'Insufficient_Weight': 'Abaixo do Peso', 'Normal_Weight': 'Peso Normal',
        'Overweight_Level_I': 'Sobrepeso (Nível I)', 'Overweight_Level_II': 'Sobrepeso (Nível II)',
        'Obesity_Type_I': 'Obesidade (Tipo I)', 'Obesity_Type_II': 'Obesidade (Tipo II)', 'Obesity_Type_III': 'Obesidade (Tipo III)'
    }

    st.title("🩺 Sistema Preditivo de Grau de Obesidade")
    st.markdown("Preencha os dados do paciente abaixo para obter a previsão comportamental do modelo.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dados Biométricos Básicos")
        gender_pt = st.radio("Gênero", ['Feminino', 'Masculino'])
        age = st.number_input("Idade", min_value=14, max_value=100, value=25)
        family_history_pt = st.selectbox("Histórico Familiar de Excesso de Peso?", ['Sim', 'Não'])

    with col2:
        st.subheader("Hábitos e Estilo de Vida")
        favc_pt = st.selectbox("Consome alimentos calóricos c/ frequência?", ['Sim', 'Não'])
        fcvc_pt = st.radio("Frequência de consumo de vegetais", ["1 - Raramente", "2 - Às vezes", "3 - Sempre"], index=1)
        ncp_pt = st.radio("Refeições principais por dia", ["1 - Uma refeição", "2 - Duas", "3 - Três", "4 - Quatro ou mais"], index=2)
        caec_pt = st.selectbox("Consome lanches entre as refeições?", ['Não', 'Às vezes', 'Frequentemente', 'Sempre'], index=1)
        smoke_pt = st.selectbox("Fumante?", ['Sim', 'Não'], index=1)

    st.subheader("Informações Adicionais")
    col3, col4 = st.columns(2)
    with col3:
        ch2o_pt = st.selectbox("Consumo diário de água", ["1 - Menos de 1 Litro", "2 - De 1 a 2 Litros", "3 - Mais de 2 Litros"], index=1)
        scc_pt = st.selectbox("Monitora ingestão calórica?", ['Sim', 'Não'], index=1)
        faf_pt = st.selectbox("Atividade Física Semanal", ["0 - Nenhuma", "1 - 1 a 2 dias", "2 - 3 a 4 dias", "3 - 5 ou mais dias"], index=1)
        
    with col4:
        tue_pt = st.selectbox("Tempo diário em telas/dispositivos", ["0 - Até 2 hours", "1 - De 3 a 5 horas", "2 - Mais de 5 horas"], index=1)
        calc_pt = st.selectbox("Consumo de bebida alcoólica", ['Não', 'Às vezes', 'Frequentemente', 'Sempre'], index=1)
        mtrans_pt = st.selectbox("Meio de transporte habitual", ['Automóvel', 'Moto', 'Bicicleta', 'Transporte Público', 'A pé'], index=3)

    if st.button("🔮 Gerar Diagnóstico Preditivo", type="primary"):
        df_paciente_num = pd.DataFrame(0, index=[0], columns=colunas_modelo)
        
        df_paciente_num['Age'] = age
        df_paciente_num['FCVC'] = int(fcvc_pt[0])
        df_paciente_num['NCP'] = int(ncp_pt[0])
        df_paciente_num['CH2O'] = int(ch2o_pt[0])
        df_paciente_num['FAF'] = int(faf_pt[0])
        df_paciente_num['TUE'] = int(tue_pt[0])
        
        categorias_selecionadas = {
            'Gender': map_genero[gender_pt], 'family_history': map_sim_nao[family_history_pt],
            'FAVC': map_sim_nao[favc_pt], 'CAEC': map_freq[caec_pt],
            'SMOKE': map_sim_nao[smoke_pt], 'SCC': map_sim_nao[scc_pt],
            'CALC': map_freq[calc_pt], 'MTRANS': map_transporte[mtrans_pt]
        }
        
        for prefixo, valor in categorias_selecionadas.items():
            nome_coluna_esperada = f"{prefixo}_{valor}"
            if nome_coluna_esperada in colunas_modelo:
                df_paciente_num[nome_coluna_esperada] = 1

        previsao_num = modelo_rf.predict(df_paciente_num)[0]
        diagnostico_ingles = le.inverse_transform([previsao_num])[0]
        diagnostico_final = map_diagnostico[diagnostico_ingles]
        
        st.success("Análise concluída!")
        st.metric(label="Diagnóstico Previsto pelo Modelo", value=diagnostico_final)
        
        if "Obesidade" in diagnostico_final:
            st.error("Atenção: O modelo indica um perfil associado à obesidade. Recomenda-se acompanhamento multidisciplinar.")
        elif "Sobrepeso" in diagnostico_final:
            st.warning("Alerta: O modelo indica um perfil de sobrepeso. É recomendado ajuste de hábitos.")
        elif "Abaixo" in diagnostico_final:
             st.warning("Alerta: Paciente com indícios de peso insuficiente.")
        else:
            st.info("Paciente apresenta um perfil associado aos parâmetros de peso esperado.")
            
        st.write("---")
        with st.expander("🔍 Verifique os dados enviados para o modelo preditivo"):
            st.subheader("🛠️ Modo de Depuração (Debug)")
            debug_visual = {"Idade": age, "FCVC": int(fcvc_pt[0]), "NCP": int(ncp_pt[0]), "CH2O": int(ch2o_pt[0]), "FAF": int(faf_pt[0]), "TUE": int(tue_pt[0])}
            debug_visual.update(categorias_selecionadas)
            st.write("**1. As 14 variáveis capturadas do formulário:**")
            st.json(debug_visual)
            st.write("**2. Tabela final processada enviada ao modelo:**")
            st.dataframe(df_paciente_num)

# ==========================================
# PÁGINA 2: FONTE DE DADOS E DASHBOARDS
# ==========================================
elif opcao_menu == "📊 Fonte de Dados":
    st.title("📊 Dashboards Analíticos")
    st.markdown("Exploração da base de dados original utilizada para treinar a Inteligência Artificial.")
    try:
        df = carregar_dados()
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.subheader("Distribuição por Gênero")
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x='Gender', palette=['#ff9ff3', '#3498db'], ax=ax1)
            st.pyplot(fig1)

        with col_graf2:
            st.subheader("Histórico Familiar")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x='family_history', palette=['#e74c3c', '#2ecc71'], ax=ax2)
            st.pyplot(fig2)
            
        st.markdown("---")
        st.subheader("Distribuição dos Níveis de Obesidade")
        ordem = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
        fig3, ax3 = plt.subplots(figsize=(10,5))
        sns.countplot(data=df, y='Obesity', order=ordem, palette='magma', ax=ax3)
        st.pyplot(fig3)
        
        with st.expander("Ver Tabela de Dados Original (CSV)"):
            st.dataframe(df)
    except FileNotFoundError:
        st.error("⚠️ Ficheiro 'Obesity.csv' não encontrado. Certifique-se de que está na mesma pasta no GitHub.")

# ==========================================
# PÁGINA 3: PIPELINE MACHINE LEARNING
# ==========================================
elif opcao_menu == "⚙️ Pipeline Machine Learning":
    st.title("⚙️ Pipeline de Machine Learning")
    st.markdown("Abaixo, pode forçar o re-treinamento manual da Inteligência Artificial em tempo real se novos dados forem inseridos no CSV.")

    if st.button("🚀 Forçar Re-treinamento do Modelo", type="primary"):
        with st.spinner("A ler dados, a aplicar Winsorização e a re-treinar a Inteligência Artificial..."):
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            try:
                df = pd.read_csv('Obesity.csv', dtype=str, sep=';', encoding='latin1')
                df.columns = df.columns.str.strip()
                
                idade_num = pd.to_numeric(df['Age'].str.split('.').str[0], errors='coerce')
                df['Age'] = idade_num.clip(lower=14, upper=61).fillna(idade_num.median())

                def limpar_categorica_clip(coluna_nome, limite_inf, limite_sup):
                    num = pd.to_numeric(df[coluna_nome], errors='coerce').round()
                    num_clipado = num.clip(lower=limite_inf, upper=limite_sup)
                    return num_clipado.fillna(num_clipado.mode()[0])

                df['FCVC'] = limpar_categorica_clip('FCVC', 1, 3)
                df['NCP']  = limpar_categorica_clip('NCP', 1, 4)
                df['CH2O'] = limpar_categorica_clip('CH2O', 1, 3)
                df['FAF']  = limpar_categorica_clip('FAF', 0, 3)   
                df['TUE']  = limpar_categorica_clip('TUE', 0, 2)
                
                colunas_inteiras = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
                for coluna in colunas_inteiras:
                    df[coluna] = df[coluna].astype('Int64')

                df = df.fillna(df.mode().iloc[0]) 
                
                X = df.drop(['Obesity', 'Height', 'Weight'], axis=1)
                y = df['Obesity']
                
                X_num = pd.get_dummies(X, drop_first=True)
                le_novo = LabelEncoder()
                y_num = le_novo.fit_transform(y)
                
                X_treino, X_teste, y_treino, y_teste = train_test_split(X_num, y_num, test_size=0.3, random_state=42)
                
                modelo_rf_novo = RandomForestClassifier(random_state=42)
                modelo_rf_novo.fit(X_treino, y_treino)
                
                joblib.dump(modelo_rf_novo, 'modelo_obesidade.pkl')
                joblib.dump(le_novo, 'label_encoder.pkl')
                joblib.dump(list(X_treino.columns), 'colunas_modelo.pkl')
                
                acuracia = modelo_rf_novo.score(X_teste, y_teste)
                
                st.success(f"✅ Modelo re-treinado com sucesso! (Base de {len(df)} pacientes mantida intacta)")
                st.metric("Acurácia do Novo Modelo no Teste", f"{acuracia * 100:.2f}%")
                
                # Força a atualização do estado global do Streamlit
                st.rerun()
            except Exception as e:
                st.error(f"Ocorreu um erro durante o treinamento: {e}")

# ==========================================
# PÁGINA 4: STORY TELLING (PDF)
# ==========================================
elif opcao_menu == "📖 Story Telling":
    st.title("📖 Story Telling do Projeto")
    st.markdown("Consulte abaixo a documentação completa e o dicionário de dados (Tech Challenge 4).")
    
    caminho_pdf = "dicionario_obesity_fiap_tc4.pdf"
    
    if os.path.exists(caminho_pdf):
        with open(caminho_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        with open(caminho_pdf, "rb") as f:
            st.download_button(
                label="📥 Fazer Download do PDF",
                data=f,
                file_name=caminho_pdf,
                mime="application/pdf"
            )
    else:
        st.error(f"⚠️ Ficheiro '{caminho_pdf}' não encontrado. Por favor, coloque o seu PDF na mesma pasta do GitHub.")