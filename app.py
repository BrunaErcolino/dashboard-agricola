import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import norm

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

# Filtros laterais
st.sidebar.header("Filtros")
cultura = st.sidebar.multiselect("Tipo de Cultura", df["Tipo_de_Cultura"].unique())
estacao = st.sidebar.multiselect("Estação", df["Estação"].unique())
solo = st.sidebar.multiselect("Tipo de Solo", df["Tipo_de_Solo"].unique())

# Aplicar filtros
df_filtrado = df.copy()
if cultura:
    df_filtrado = df_filtrado[df_filtrado["Tipo_de_Cultura"].isin(cultura)]
if estacao:
    df_filtrado = df_filtrado[df_filtrado["Estação"].isin(estacao)]
if solo:
    df_filtrado = df_filtrado[df_filtrado["Tipo_de_Solo"].isin(solo)]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("🌽 Produção Total", f"{df_filtrado['Produção(toneladas)'].sum():,.2f} ton")
col2.metric("💧 Média de Água", f"{df_filtrado['Uso_de_Água(metros cúbicos)'].mean():,.2f} m³")
col3.metric("📈 Eficiência Média", f"{df_filtrado['Eficiência (ton/acre)'].mean():.2f} ton/acre")

# Gráficos
st.subheader("📊 PRODUÇÃO POR CULTURA")
fig1 = px.bar(df_filtrado.groupby("Tipo_de_Cultura")["Produção(toneladas)"].sum().reset_index(),
              x="Tipo_de_Cultura", y="Produção(toneladas)", title="Produção por Cultura", text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("💧 CONSUMO DE ÁGUA POR IRRIGAÇÃO")
fig2 = px.box(df_filtrado, x="Tipo_de_Irrigação", y="Uso_de_Água(metros cúbicos)",
              title="Distribuição de Água por Tipo de Irrigação")
st.plotly_chart(fig2, use_container_width=True)

# Estatísticas Avançadas
st.header("📊 Análises Estatísticas Avançadas")

st.subheader("CORRELAÇÃO ENTRE VARIÁVEIS NUMÉRICAS")
variaveis_corr = st.multiselect("Selecione variáveis para matriz de correlação:",
                                 df_filtrado.select_dtypes(include=np.number).columns.tolist(),
                                 default=df_filtrado.select_dtypes(include=np.number).columns.tolist())
if len(variaveis_corr) >= 2:
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_filtrado[variaveis_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Matriz de Correlação")
    st.pyplot(fig_corr)
else:
    st.warning("Selecione pelo menos duas variáveis numéricas para gerar a matriz de correlação.")

st.subheader("DISTRIBUIÇÃO DE VARIÁVEIS QUANTITATIVAS")
variaveis_quantitativas = df_filtrado.select_dtypes(include=np.number).columns.tolist()
variavel_selecionada = st.selectbox("Selecione a variável para análise", variaveis_quantitativas)

dados = df_filtrado[variavel_selecionada].dropna()
media = dados.mean()
desvio = dados.std()

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(dados, bins=20, stat="density", color="skyblue", alpha=0.5, ax=ax, label="Distribuição dos dados")
sns.kdeplot(dados, color="blue", linewidth=2, ax=ax)
x = np.linspace(dados.min(), dados.max(), 100)
curva_normal = norm.pdf(x, media, desvio)
ax.plot(x, curva_normal, 'r-', linewidth=2, label="Curva Normal")
ax.set_title(f"Distribuição Normal - {variavel_selecionada}")
ax.set_ylabel("Densidade")
ax.legend()
st.pyplot(fig)

st.subheader("📊 CORRELAÇÃO")
variaveis_numericas = df_filtrado.select_dtypes(include=np.number).columns.tolist()
colx, coly = st.columns(2)
with colx:
    var_x = st.selectbox("Variável X", variaveis_numericas, index=0)
with coly:
    var_y = st.selectbox("Variável Y", variaveis_numericas, index=1)

fig_corr_filtros = px.scatter(df_filtrado, x=var_x, y=var_y,
                              title=f"Correlação: {var_x} x {var_y}",
                              trendline="ols")
st.plotly_chart(fig_corr_filtros, use_container_width=True)

# 🔁 Regressão Linear Simples com faixa de confiança
st.subheader("📏 REGRESSÃO LINEAR COM FAIXA DE CONFIANÇA")
variavel_independente_fc = st.selectbox("Escolha a variável independente (X) para regressão com faixa de confiança",
                                        [col for col in variaveis_numericas if col != "Produção(toneladas)"],
                                        key="faixa")

fig_fc, ax_fc = plt.subplots(figsize=(8, 4))
sns.regplot(
    data=df_filtrado,
    x=variavel_independente_fc,
    y="Produção(toneladas)",
    ci=95,
    line_kws={"color": "red", "label": "Linha de Regressão"},
    scatter_kws={"alpha": 0.7},
    ax=ax_fc
)
ax_fc.set_title(f"Relação Linear entre {variavel_independente_fc} e Produção(toneladas)")
ax_fc.set_xlabel(variavel_independente_fc)
ax_fc.set_ylabel("Produção(toneladas)")
ax_fc.legend()
st.pyplot(fig_fc)

# Regressão Linear Múltipla
st.header("📉 Análise Preditiva da Produção")

X = df_filtrado[["Área_da_Fazenda(acres)", "Fertilizante_Usado(toneladas)", "Pesticida_Usado(kg)", "Uso_de_Água(metros cúbicos)"]]
y = df_filtrado["Produção(toneladas)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

st.markdown("### 📋 AVALIAÇÃO DO MODELO")
st.markdown(f"**R² (coeficiente de determinação):** {r2_score(y_test, y_pred):.2f}")
st.markdown(f"**Erro Médio Absoluto (MAE):** {mean_absolute_error(y_test, y_pred):.2f} toneladas")

st.markdown("### 🔍 IMPORTÂNCIA DAS VARIÁVEIS (COEFICIENTES)")
coef_df = pd.DataFrame({
    "Variável": X.columns,
    "Coeficiente": modelo.coef_
})
st.dataframe(coef_df)

st.markdown("### 📎 Equação da Regressão Linear")
equacao = " + ".join([f"({coef:.2f} × {col})" for coef, col in zip(modelo.coef_, X.columns)])
st.latex(f"\\text{{Produção Prevista}} = {modelo.intercept_:.2f} + {equacao}")

st.subheader("COMPARAÇÃO: PRODUÇÃO REAL VS PREVISTA")
df_resultado = pd.DataFrame({"Real": y_test, "Prevista": y_pred})
fig7 = px.scatter(df_resultado, x="Real", y="Prevista", trendline="ols",
                  title="Produção Real vs Produção Prevista",
                  labels={"Real": "Produção Real", "Prevista": "Produção Prevista"})
fig7.update_traces(marker=dict(color='blue', size=10, line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig7, use_container_width=True)

st.subheader("🗓️ PREVISÃO COM NOVOS DADOS (SIMULAÇÃO)")
with st.form("form_previsao"):
    st.markdown("Insira os valores estimados para um novo cenário futuro:")
    area = st.number_input("Área da Fazenda (acres)", min_value=0.0, step=1.0)
    fertilizante = st.number_input("Fertilizante Usado (toneladas)", min_value=0.0, step=0.1)
    pesticida = st.number_input("Pesticida Usado (kg)", min_value=0.0, step=0.1)
    agua = st.number_input("Uso de Água (m³)", min_value=0.0, step=10.0)
    submit = st.form_submit_button("Prever Produção")

    if submit:
        entrada = np.array([[area, fertilizante, pesticida, agua]])
        previsao = modelo.predict(entrada)[0]
        st.success(f"A produção estimada é: {previsao:.2f} toneladas")

        # Adiciona ponto de simulação ao gráfico
        fig7.add_scatter(
            x=[df_resultado["Real"].min() - 5],
            y=[previsao],
            mode='markers',
            marker=dict(symbol='star', size=18, color='red', line=dict(color='black', width=1.5)),
            name="Simulação"
        )
        fig7.add_annotation(
            x=df_resultado["Real"].min() - 5,
            y=previsao,
            text=f"{previsao:.2f} ton",
            showarrow=True,
            arrowhead=2,
            ax=60,
            ay=-40,
            font=dict(color="red", size=14),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1
        )
        st.plotly_chart(fig7, use_container_width=True)
