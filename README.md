# Tech Challenge 4 - FIAP: Predição Comportamental de Obesidade

Este repositório contém o estudo desenvolvido para o Tech Challenge 4 do curso Data Analytics da FIAP.

Utilizamos a base de dados Obesity.csv fornecida pela FIAP para desenvolver um modelo de machine learning que faça predição de estado clínico relacionado a obesidade, baseado em um formulário de perguntas.

Foi utilizado o modelo Random Forest, que teve uma boa acuracidade, acima de 80% no acerto do diagnóstico preditivo, utilizando apenas os dados comportamentais, como hábitos alimentares e histórico familiar, descartando altura e peso para evitar vazamento de dados.

🔗 **Link para a aplicação na nuvem:** https://techchalengeobesity.streamlit.app/

## Estrutura do Projeto
- `app.py`: Código fonte da aplicação Streamlit.
- `Obesity_Limpeza.ipnb`: Notebook do Colag onde consta a limpeza dos dados do dataset original;
- `Obesity_EDA.ipynb`: Notebook utilizado para analisar os dados já tratados, gerar os insigths que constam na documentação executiva e treinar o modelo de machine learning;
- `Obesidade_AnaliseComportamental.pdf`: Documentação Executiva.
- `Obesity.csv`: Dataset original.
- `Obesity_Limpo.csv`: Dataset tratado, sem ruidos.
- `dicionario_obesity_fiap_tc4.pdf`: Dicionário de dados.
- `.pkl`: Arquivos binários do modelo treinado.
