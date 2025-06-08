import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Configuração
st.set_page_config(page_title="Dashboard Agrícola Interativo", layout="wide")
st.title("🌾 Dashboard de Produção Agrícola")

# Dados
@st.cache_data
def carregar_dados():
    df = pd.read_excel("Conjunto_de_Dados_Agricolas.xlsx")
    df["Eficiência (ton/acre)"] = df["Produção(toneladas)"] / df["Área_da_Fazenda(acres)"]
    return df

df = carregar_dados()

# Filtros
st.sidebar.header("🔍 Filtros")
culturas = st.sidebar.multiselect("Tipo de Cultura", df["Tipo_de_Cultura"].unique(), default=df["Tipo_de_Cultura"].unique())
estacoes = st.sidebar.multiselect("Estação do Ano", df["Estação"].unique(), default=df["Estação"].unique())
solos = st.sidebar.multiselect("Tipo de Solo", df["Tipo_de_Solo"].unique(), default=df["Tipo_de_Solo"].unique())

df_filtrado = df[
    (df["Tipo_de_Cultura"].isin(culturas)) &
    (df["Estação"].isin(estacoes)) &
    (df["Tipo_de_Solo"].isin(solos))
]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("🌽 Produção Total", f"{df_filtrado['Produção(toneladas)'].sum():,.2f} ton")
col2.metric("💧 Média de Água", f"{df_filtrado['Uso_de_Água(metros cúbicos)'].mean():,.2f} m³")
col3.metric("📈 Eficiência Média", f"{df_filtrado['Eficiência (ton/acre)'].mean():.2f} ton/acre")

# Gráficos
st.subheader("📊 Produção por Cultura")
fig1 = px.bar(df_filtrado.groupby("Tipo_de_Cultura")["Produção(toneladas)"].sum().reset_index(),
              x="Tipo_de_Cultura", y="Produção(toneladas)", title="Produção por Cultura", text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("💧 Consumo de Água por Irrigação")
fig2 = px.box(df_filtrado, x="Tipo_de_Irrigação", y="Uso_de_Água(metros cúbicos)",
              title="Distribuição de Água por Tipo de Irrigação")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📈 Fertilizante vs Produção")
fig3 = px.scatter(df_filtrado, x="Fertilizante_Usado(toneladas)", y="Produção(toneladas)",
                  color="Tipo_de_Cultura", size="Área_da_Fazenda(acres)", title="Fertilizante x Produção")
st.plotly_chart(fig3, use_container_width=True)

# Estatísticas Avançadas
st.header("📊 Análises Estatísticas Avançadas")

# Distribuição
st.subheader("Distribuição da Produção")
fig4, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df_filtrado["Produção(toneladas)"], kde=True, bins=20, color='skyblue', ax=ax1)
ax1.set_title("Distribuição da Produção")
st.pyplot(fig4)

# Correlação
st.subheader("Correlação entre Variáveis Numéricas")
fig5, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(df_filtrado.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
ax2.set_title("Matriz de Correlação")
st.pyplot(fig5)

# Regressão
st.subheader("Regressão Linear: Fertilizante x Produção")
fig6, ax3 = plt.subplots(figsize=(8, 4))
sns.regplot(data=df_filtrado, x="Fertilizante_Usado(toneladas)", y="Produção(toneladas)", line_kws={"color": "red"}, ax=ax3)
ax3.set_title("Relação Linear entre Fertilizante e Produção")
st.pyplot(fig6)

# Tabela
st.subheader("📄 Dados Filtrados")
st.dataframe(df_filtrado)