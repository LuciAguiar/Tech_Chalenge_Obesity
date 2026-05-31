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
# 2. INICIALIZAÇÃO E CARREGAMENTO INTELIGENTE 
# ==========================================
@st.cache_resource
def carregar_arquivos():
    nome_modelo = 'modelo_obesidade.pkl'
    nome_encoder_y = 'label_encoder.pkl'
    nome_encoders_x = 'feature_encoders.pkl' 
    nome_colunas = 'colunas_modelo.pkl'
    
    if os.path.exists(nome_modelo) and os.path.exists(nome_encoder_y) and os.path.exists(nome_encoders_x) and os.path.exists(nome_colunas):
        modelo = joblib.load(nome_modelo)
        le_y = joblib.load(nome_encoder_y)
        le_dict = joblib.load(nome_encoders_x)
        colunas = joblib.load(nome_colunas)
        return modelo, le_y, le_dict, colunas
    
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        url = 'https://raw.githubusercontent.com/LuciAguiar/Tech_Chalenge_Obesity/refs/heads/main/Obesity_Limpo.csv'
        
        df = pd.read_csv(url, sep=',', decimal='.')
        
        X = df.drop(columns=['Weight', 'Height', 'Obesity'])
        y = df['Obesity']
        
        le_dict = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        modelo_rf_novo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        modelo_rf_novo.fit(X_treino, y_treino)
        
        joblib.dump(modelo_rf_novo, nome_modelo)
        joblib.dump(le_y, nome_encoder_y)
        joblib.dump(le_dict, nome_encoders_x)
        joblib.dump(list(X.columns), nome_colunas)
        
        return modelo_rf_novo, le_y, le_dict, list(X.columns)

@st.cache_data
def carregar_dados():
    url = 'https://raw.githubusercontent.com/LuciAguiar/Tech_Chalenge_Obesity/refs/heads/main/Obesity_Limpo.csv'
    df = pd.read_csv(url, sep=',', decimal='.')
    return df

modelo_rf, le_y, le_dict, colunas_modelo = carregar_arquivos()

# ==========================================
# 3. MENU LATERAL (SIDEBAR)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=100)
st.sidebar.title("Menu de Navegação")

# ALTERAÇÃO AQUI: "Documentação Executiva" em vez de "Story Telling"
opcao_menu = st.sidebar.radio(
    "Selecione a página:",
    ["🔮 Análise Preditiva", "📊 Fonte de Dados", "⚙️ Pipeline Machine Learning", "📖 Documentação Executiva"]
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
        tue_pt = st.selectbox("Tempo diário em telas/dispositivos", ["0 - Até 2 horas", "1 - De 3 a 5 horas", "2 - Mais de 5 horas"], index=1)
        calc_pt = st.selectbox("Consumo de bebida alcoólica", ['Não', 'Às vezes', 'Frequentemente', 'Sempre'], index=1)
        mtrans_pt = st.selectbox("Meio de transporte habitual", ['Automóvel', 'Moto', 'Bicicleta', 'Transporte Público', 'A pé'], index=3)

    if st.button("🔮 Gerar Diagnóstico Preditivo", type="primary"):
        input_data = {
            'Gender': map_genero[gender_pt],
            'Age': age,
            'family_history': map_sim_nao[family_history_pt],
            'FAVC': map_sim_nao[favc_pt],
            'FCVC': int(fcvc_pt[0]),
            'NCP': int(ncp_pt[0]),
            'CAEC': map_freq[caec_pt],
            'SMOKE': map_sim_nao[smoke_pt],
            'CH2O': int(ch2o_pt[0]),
            'SCC': map_sim_nao[scc_pt],
            'FAF': int(faf_pt[0]),
            'TUE': int(tue_pt[0]),
            'CALC': map_freq[calc_pt],
            'MTRANS': map_transporte[mtrans_pt]
        }
        
        df_paciente = pd.DataFrame([input_data])[colunas_modelo]
        
        for col, le_feat in le_dict.items():
            if col in df_paciente.columns:
                df_paciente[col] = le_feat.transform(df_paciente[col].astype(str))

        previsao_num = modelo_rf.predict(df_paciente)[0]
        diagnostico_ingles = le_y.inverse_transform([previsao_num])[0]
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
            st.write("**1. Variáveis capturadas do formulário:**")
            st.json(input_data)
            st.write("**2. Tabela final processada enviada ao modelo (Label Encoded):**")
            st.dataframe(df_paciente)

# ==========================================
# PÁGINA 2: FONTE DE DADOS E DASHBOARDS
# ==========================================
elif opcao_menu == "📊 Fonte de Dados":
    st.title("📊 Dashboards Analíticos")
    st.markdown("Exploração da base de dados limpa oficial utilizada para treinar a Inteligência Artificial.")
    
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #f4f4f4;
        border-radius: 8px;
        padding: 8px 12px;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e6e6e6;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.4rem !important; 
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important; 
    }
    </style>
    """, unsafe_allow_html=True)

    try:
        df = carregar_dados()
        total_amostra = len(df)
        
        col_smoke = next((col for col in df.columns if 'SMOKE' in str(col).strip().upper()), None)
        col_favc = next((col for col in df.columns if 'FAVC' in str(col).strip().upper()), None)
        
        def contar_positivos(coluna):
            if coluna is None:
                return 0
            valores_limpos = df[coluna].astype(str).str.strip().str.lower().str.replace('.0', '', regex=False)
            return valores_limpos.isin(['yes', '1', 'sim', 'true', 'y', 's']).sum()

        total_fumantes = contar_positivos(col_smoke)
        total_favc = contar_positivos(col_favc)
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric(label="📊 Tamanho da Amostra", value=total_amostra)
        with col_metric2:
            st.metric(label="🚬 Total de Fumantes", value=total_fumantes)
        with col_metric3:
            st.metric(label="🍔 Consumo alimentos calóricos", value=total_favc)
                    
        st.markdown("---")
        
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.markdown("#### Distribuição por Gênero")
            fig1, ax1 = plt.subplots(figsize=(4, 2.8)) 
            
            sns.countplot(data=df, x='Gender', order=['Female', 'Male'], palette=['#ff9ff3', '#3498db'], ax=ax1)
            ax1.set_xlabel("Gênero", fontsize=8)
            ax1.set_ylabel("") 
            ax1.set_xticklabels(['Feminino', 'Masculino'], fontsize=8) 
            ax1.tick_params(axis='y', labelsize=8)
            sns.despine()
            st.pyplot(fig1)

        with col_graf2:
            st.markdown("#### Histórico Familiar")
            fig2, ax2 = plt.subplots(figsize=(4, 2.8))
            
            sns.countplot(data=df, x='family_history', order=['yes', 'no'], palette=['#e74c3c', '#2ecc71'], ax=ax2)
            ax2.set_xlabel("Histórico Familiar de Excesso de Peso", fontsize=8)
            ax2.set_ylabel("") 
            ax2.set_xticklabels(['Sim', 'Não'], fontsize=8) 
            ax2.tick_params(axis='y', labelsize=8)
            sns.despine()
            st.pyplot(fig2)
            
        st.markdown("---")
        st.markdown("#### Distribuição dos Níveis de Obesidade")
        ordem = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
        
        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        sns.countplot(data=df, y='Obesity', order=ordem, palette='magma_r', ax=ax3)
        
        for container in ax3.containers:
            labels_personalizadas = [f"{int(barra.get_width())} ({(barra.get_width() / total_amostra) * 100:.1f}%)" for barra in container]
            ax3.bar_label(container, labels=labels_personalizadas, padding=5, fontsize=8, color='black', weight='bold')
            
        ax3.set_xlabel(f"Quantidade de Pacientes (Total da Amostra: {total_amostra})", fontsize=9, labelpad=8)
        ax3.set_ylabel("") 
        ax3.set_yticklabels(['Abaixo do Peso', 'Peso Normal', 'Sobrepeso (Nível I)', 'Sobrepeso (Nível II)', 'Obesidade (Tipo I)', 'Obesidade (Tipo II)', 'Obesidade (Tipo III)'], fontsize=8)
        ax3.tick_params(axis='x', labelsize=8)
        sns.despine() 
        st.pyplot(fig3)
        
        st.markdown("---")
        st.markdown("#### Grau de Obesidade vs Histórico Familiar")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        
        sns.countplot(data=df, y='Obesity', hue='family_history', order=ordem, hue_order=['yes', 'no'], palette=['#e74c3c', '#2ecc71'], ax=ax4)
        
        ax4.set_yticklabels(['Abaixo do Peso', 'Peso Normal', 'Sobrepeso (Nível I)', 'Sobrepeso (Nível II)', 'Obesidade (Tipo I)', 'Obesidade (Tipo II)', 'Obesidade (Tipo III)'], fontsize=8)
        ax4.set_xlabel("Quantidade de Pacientes", fontsize=9, labelpad=8)
        ax4.set_ylabel("")
        ax4.tick_params(axis='x', labelsize=8)
        
        for container in ax4.containers:
            ax4.bar_label(container, padding=5, fontsize=7, color='black')
            
        legenda = ax4.get_legend()
        if legenda:
            legenda.set_title("Histórico Familiar", prop={'size': 8})
            for texto in legenda.texts:
                texto.set_fontsize(8)
                if texto.get_text() == 'yes':
                    texto.set_text('Sim')
                elif texto.get_text() == 'no':
                    texto.set_text('Não')

        sns.despine()
        st.pyplot(fig4)
        
        with st.expander("Ver Tabela de Dados Original (CSV Limpo)"):
            st.dataframe(df)
    except Exception as e:
        st.error(f"⚠️ Erro ao carregar os dados. Verifique a conexão de internet ou a URL do GitHub. Erro: {e}")

# ==========================================
# PÁGINA 3: PIPELINE MACHINE LEARNING
# ==========================================
elif opcao_menu == "⚙️ Pipeline Machine Learning":
    st.title("⚙️ Pipeline de Machine Learning")
    st.markdown("Abaixo, pode forçar o re-treinamento manual da Inteligência Artificial usando a nova lógica de Label Encoding com 80.38% de precisão.")

    if st.button("🚀 Forçar Re-treinamento do Modelo", type="primary"):
        with st.spinner("A ler dados limpos, a aplicar Label Encoding e a re-treinar a IA..."):
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score
            
            try:
                url = 'https://raw.githubusercontent.com/LuciAguiar/Tech_Chalenge_Obesity/refs/heads/main/Obesity_Limpo.csv'
                
                df = pd.read_csv(url, sep=',', decimal='.')
                
                X = df.drop(columns=['Weight', 'Height', 'Obesity'])
                y = df['Obesity']
                
                le_dict_novo = {}
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict_novo[col] = le

                le_y_novo = LabelEncoder()
                y_encoded = le_y_novo.fit_transform(y)
                
                X_treino, X_teste, y_treino, y_teste = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                modelo_rf_novo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                modelo_rf_novo.fit(X_treino, y_treino)
                
                joblib.dump(modelo_rf_novo, 'modelo_obesidade.pkl')
                joblib.dump(le_y_novo, 'label_encoder.pkl')
                joblib.dump(le_dict_novo, 'feature_encoders.pkl') 
                joblib.dump(list(X.columns), 'colunas_modelo.pkl')
                
                previsoes = modelo_rf_novo.predict(X_teste)
                acuracia = accuracy_score(y_teste, previsoes)
                
                st.success(f"✅ Modelo re-treinado com sucesso usando a Lógica Avançada!")
                st.metric("Acurácia do Novo Modelo no Teste", f"{acuracia * 100:.2f}%")
                
                st.rerun()
            except Exception as e:
                st.error(f"Ocorreu um erro durante o treinamento: {e}")

# ==========================================
# PÁGINA 4: DOCUMENTAÇÃO EXECUTIVA (PDF)
# ==========================================
# ALTERAÇÃO AQUI: Correspondência com o novo nome no menu
elif opcao_menu == "📖 Documentação Executiva":
    st.title("📖 Documentação Executiva do Projeto")
    st.info("A redirecionar... O documento PDF deve abrir automaticamente numa nova aba do seu navegador.")
    
    url_pdf = "https://cdn.jsdelivr.net/gh/LuciAguiar/Tech_Chalenge_Obesity@main/Obesidade_AnaliseComportamental.pdf"
    
    import streamlit.components.v1 as components
    components.html(
        f"""
        <script>
            window.open('{url_pdf}', '_blank');
        </script>
        """,
        height=0
    )
    
    st.warning("⚠️ Se o seu navegador bloqueou a abertura automática (Bloqueador de Pop-ups), clique no botão abaixo:")
    
    st.markdown(
        f'''
        <a href="{url_pdf}" target="_blank" style="
            display: inline-block;
            padding: 12px 24px;
            background-color: #FF4B4B;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
            margin-top: 10px;
        ">📄 Abrir PDF Diretamente no Navegador</a>
        ''',
        unsafe_allow_html=True
    )