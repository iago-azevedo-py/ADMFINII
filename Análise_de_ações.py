# -*- coding: utf-8 -*-
"""
Sistema Oráculo de Análise de Ações Americanas - Cloud Ready
"""
import streamlit as st
import os
import warnings
import json
import asyncio
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import requests

import yfinance as yf
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import google.generativeai as genai

# Configuração da página
st.set_page_config(
    page_title="Assistente para Análise de Ações",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")

# CONSTANTES
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
GNEWS_BASE_URL = "https://gnews.io/api/v4"

# FUNÇÕES PARA ACESSAR CHAVES (simplificadas)
def get_google_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        return None

def get_fmp_api_key():
    try:
        return st.secrets["FMP_API_KEY"]
    except:
        return None

def get_gnews_api_key():
    try:
        return st.secrets["GNEWS_API_KEY"]
    except:
        return None

def get_serpapi_key():
    try:
        return st.secrets["SERPAPI_KEY"]
    except:
        return None

def configure_gemini_auto():
    """Configura Gemini automaticamente"""
    api_key = get_google_api_key()
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except:
        return False

# FUNÇÕES AUXILIARES (mantidas iguais, mas usando config)
@st.cache_resource
def initialize_session_service():
    """Inicializa o serviço de sessão em memória para o agente."""
    return InMemorySessionService()

async def call_agent(agent: Agent, message_text: str, session_service) -> str:
    """Executa um agente com uma mensagem de entrada e retorna a resposta textual final."""
    session = await session_service.create_session(app_name=agent.name, user_id="user1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    current_message = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response_text = ""
    async for event in runner.run_async(user_id="user1", session_id=session.id, new_message=current_message):
        if event.content:
            for part in event.content.parts:
                if part.text:
                    final_response_text += part.text
    return final_response_text

def formatar_numero(valor, formato="{:,}", default="N/A"):
    """Formata número com vírgulas, retorna default se não for numérico"""
    try:
        if isinstance(valor, (int, float)) and valor != 0:
            return formato.format(valor)
        return default
    except Exception:
        return default

def formatar_moeda_usd(valor):
    """Formata valores monetários em USD"""
    try:
        if not isinstance(valor, (int, float)) or valor == 0:
            return "$ 0.00"
        
        if abs(valor) >= 1_000_000_000:
            return f"$ {valor/1_000_000_000:,.2f}B"
        elif abs(valor) >= 1_000_000:
            return f"$ {valor/1_000_000:,.2f}M"
        elif abs(valor) >= 1_000:
            return f"$ {valor/1_000:,.2f}K"
        else:
            return f"$ {valor:,.2f}"
    except:
        return "$ 0.00"

def inicializar_perfil_investidor():
    """Inicializa o perfil do investidor"""
    if 'perfil_investidor' not in st.session_state:
        st.session_state.perfil_investidor = {
            'tipo': 'investidor',
            'preferencias': [],
            'historico_interacoes': [],
            'setores_interesse': [],
            'criterios_importantes': ['dividend_yield', 'debt_ratio', 'pe_ratio', 'stability']
        }

def atualizar_perfil_investidor(pergunta, contexto_empresa):
    """Atualiza o perfil do investidor baseado nas perguntas"""
    perfil = st.session_state.perfil_investidor
    
    pergunta_lower = pergunta.lower()
    
    if 'dividend' in pergunta_lower or 'dividendo' in pergunta_lower:
        if 'dividend_yield' not in perfil['criterios_importantes']:
            perfil['criterios_importantes'].append('dividend_yield')
    
    if 'risco' in pergunta_lower or 'risk' in pergunta_lower:
        if 'risk_analysis' not in perfil['criterios_importantes']:
            perfil['criterios_importantes'].append('risk_analysis')
    
    if 'crescimento' in pergunta_lower or 'growth' in pergunta_lower:
        if 'growth_metrics' not in perfil['criterios_importantes']:
            perfil['criterios_importantes'].append('growth_metrics')
    
    if contexto_empresa and 'profile' in contexto_empresa:
        sector = contexto_empresa['profile'].get('sector', '')
        if sector and sector not in perfil['setores_interesse']:
            perfil['setores_interesse'].append(sector)
    
    perfil['historico_interacoes'] = [pergunta]
    st.session_state.perfil_investidor = perfil

def obter_noticias_empresa(ticker_symbol, company_name):
    """Obtém notícias recentes sobre a empresa usando GNews"""
    gnews_api_key = get_gnews_api_key()
    if not gnews_api_key:
        return []
    
    try:
        params = {
            'q': f'"{company_name}" OR "{ticker_symbol}"',
            'lang': 'en',
            'max': 10,
            'token': gnews_api_key
        }
        response = requests.get(f"{GNEWS_BASE_URL}/search", params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('articles', [])
        elif response.status_code == 401:
            st.warning("⚠️ Chave GNews inválida ou expirada")
        elif response.status_code == 429:
            st.warning("⚠️ Limite de requisições GNews atingido")
        return []
    except Exception as e:
        st.warning(f"Erro ao obter notícias: {str(e)[:50]}")
        return []

def obter_noticias_setor(sector):
    """Obtém notícias do setor usando GNews"""
    gnews_api_key = get_gnews_api_key()
    if not gnews_api_key:
        return []
    try:
        params = {
            'q': f'"{sector}" AND (stocks OR investment OR market)',
            'lang': 'en',
            'max': 5,
            'token': gnews_api_key
        }
        response = requests.get(f"{GNEWS_BASE_URL}/search", params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('articles', [])
        return []
    except Exception:
        return []

def fmp_get(endpoint, symbol):
    fmp_api_key = get_fmp_api_key()
    
    if not fmp_api_key:
        st.warning("⚠️ Chave FMP não encontrada. Usando dados limitados do Yahoo Finance.")
        return None
    
    url = f"{FMP_BASE_URL}/{endpoint}/{symbol}?apikey={fmp_api_key.strip()}"
    
    try:
        r = requests.get(url, timeout=15)
        
        if r.status_code == 200:
            data = r.json()
            return data if data else None
        elif r.status_code == 401:
            st.error("❌ Chave FMP inválida ou expirada")
        elif r.status_code == 429:
            st.error("❌ Limite de requisições FMP atingido")
        return None
    except Exception as e:
        st.error(f"❌ Erro de conexão com FMP: {str(e)[:50]}")
        return None

def obter_dados_fmp_completos(ticker_symbol):
    """Obtém dados completos do FMP incluindo histórico de 3 anos"""
    income = fmp_get("income-statement", ticker_symbol)
    balance = fmp_get("balance-sheet-statement", ticker_symbol)
    cashflow = fmp_get("cash-flow-statement", ticker_symbol)
    profile = fmp_get("profile", ticker_symbol)
    historical = fmp_get("historical-price-full", ticker_symbol)
    ratios = fmp_get("ratios", ticker_symbol)
    key_metrics = fmp_get("key-metrics", ticker_symbol)
    
    return {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "profile": profile[0] if profile and isinstance(profile, list) and len(profile) > 0 else {},
        "historical": historical,
        "ratios": ratios,
        "key_metrics": key_metrics
    }

def calcular_indicadores_fmp(dados_fmp):
    """Calcula os 10 principais indicadores financeiros usando dados FMP"""
    if not dados_fmp:
        return None
        
    income = dados_fmp.get("income", [])
    balance = dados_fmp.get("balance", [])
    ratios = dados_fmp.get("ratios", [])
    
    if not income or not balance or len(income) == 0 or len(balance) == 0:
        return None
    
    income_atual = income[0] if income else {}
    balance_atual = balance[0] if balance else {}
    ratios_atual = ratios[0] if ratios else {}
    
    indicadores = {}
    
    try:
        receita = income_atual.get('revenue', 0)
        custo_vendas = income_atual.get('costOfRevenue', 0)
        lucro_bruto = receita - custo_vendas
        indicadores['Margem Bruta (%)'] = (lucro_bruto / receita * 100) if receita > 0 else 0
        
        lucro_operacional = income_atual.get('operatingIncome', 0)
        indicadores['Margem Operacional (%)'] = (lucro_operacional / receita * 100) if receita > 0 else 0
        
        lucro_liquido = income_atual.get('netIncome', 0)
        indicadores['Margem Líquida (%)'] = (lucro_liquido / receita * 100) if receita > 0 else 0
        
        patrimonio_liquido = balance_atual.get('totalStockholdersEquity', 0)
        indicadores['ROE (%)'] = (lucro_liquido / patrimonio_liquido * 100) if patrimonio_liquido > 0 else 0
        
        ativo_total = balance_atual.get('totalAssets', 0)
        indicadores['ROA (%)'] = (lucro_liquido / ativo_total * 100) if ativo_total > 0 else 0
        
        indicadores['P/L Ratio'] = ratios_atual.get('priceEarningsRatio', 0)
        
        divida_total = balance_atual.get('totalDebt', 0)
        indicadores['Debt-to-Equity'] = (divida_total / patrimonio_liquido) if patrimonio_liquido > 0 else 0
        
        ativo_circulante = balance_atual.get('totalCurrentAssets', 0)
        passivo_circulante = balance_atual.get('totalCurrentLiabilities', 0)
        indicadores['Current Ratio'] = (ativo_circulante / passivo_circulante) if passivo_circulante > 0 else 0
        
        indicadores['EPS (USD)'] = income_atual.get('eps', 0)
        
        if len(income) >= 2:
            receita_anterior = income[1].get('revenue', 0)
            indicadores['Crescimento Receita YoY (%)'] = ((receita - receita_anterior) / receita_anterior * 100) if receita_anterior > 0 else 0
        else:
            indicadores['Crescimento Receita YoY (%)'] = 0
            
    except Exception as e:
        st.error(f"Erro ao calcular indicadores: {e}")
        return None
    
    return indicadores

def criar_graficos_historicos_fmp(dados_fmp, ticker_symbol):
    """Cria gráficos históricos usando dados FMP"""
    historical = dados_fmp.get("historical", {})
    
    if not historical or 'historical' not in historical:
        st.warning("Dados históricos não disponíveis no FMP. Usando Yahoo Finance.")
        return criar_graficos_yahoo_fallback(ticker_symbol)
    
    hist_data = historical['historical']
    df = pd.DataFrame(hist_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    tres_anos_atras = datetime.now() - timedelta(days=3*365)
    df_3y = df[df['date'] >= tres_anos_atras]
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Preço de Fechamento', 'Volume', 'Candlestick', 'Variação %', 'High vs Low', 'Média Móvel'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"colspan": 2}, None],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df_3y['date'], y=df_3y['close'], name='Preço', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df_3y['date'], y=df_3y['volume'], name='Volume', marker_color='rgba(0,100,80,0.6)'),
        row=1, col=2
    )
    
    df_1y = df_3y.tail(252)
    fig.add_trace(
        go.Candlestick(
            x=df_1y['date'],
            open=df_1y['open'],
            high=df_1y['high'],
            low=df_1y['low'],
            close=df_1y['close'],
            name='OHLC'
        ),
        row=2, col=1
    )
    
    df_3y['variacao_pct'] = df_3y['changePercent']
    fig.add_trace(
        go.Scatter(x=df_3y['date'], y=df_3y['variacao_pct'], name='Variação %', line=dict(color='red')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_3y['date'], y=df_3y['high'], name='Máxima', line=dict(color='green')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_3y['date'], y=df_3y['low'], name='Mínima', line=dict(color='red')),
        row=3, col=2
    )
    
    fig.update_layout(
        height=1000,
        title=f'Análise Histórica Completa - {ticker_symbol} (Últimos 3 Anos)',
        showlegend=True
    )
    
    return fig

def criar_graficos_yahoo_fallback(ticker_symbol):
    """Fallback para Yahoo Finance se FMP não estiver disponível"""
    empresa = yf.Ticker(ticker_symbol)
    hist = empresa.history(period="3y")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='Preço de Fechamento (Yahoo Finance)',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title=f'Histórico de Preços - {ticker_symbol} (3 Anos - Yahoo Finance)',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        height=400
    )
    
    return fig

def buscar_google_serpapi(query, serpapi_key=None):
    """Busca resultados no Google usando SerpAPI"""
    if not serpapi_key:
        serpapi_key = get_serpapi_key()
    
    if not serpapi_key:
        return []
    
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "hl": "pt",
        "gl": "br",
        "api_key": serpapi_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [
                f"{item['title']}: {item.get('snippet','')}\nFonte: {item['link']}"
                for item in data.get("organic_results", [])[:3]
            ]
    except Exception as e:
        st.warning(f"Erro ao buscar no Google (SerpAPI): {str(e)[:50]}")
    return []

async def chat_consultor_acoes(pergunta, dados_contexto, noticias, session_service):
    """Chat especializado para análise de ações"""
    
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError("Google API Key não encontrada")
    
    genai.configure(api_key=api_key)
    
    atualizar_perfil_investidor(pergunta, dados_contexto)
    perfil = st.session_state.perfil_investidor

    contexto_financeiro = json.dumps(dados_contexto, indent=2, ensure_ascii=False, default=str)
    contexto_noticias = "\n".join([
        f"- {noticia['title']}: {noticia.get('description','')}\nFonte: {noticia.get('url','')}"
        for noticia in noticias[:5]
    ])

    serpapi_key = get_serpapi_key()
    profile = dados_contexto.get('profile', {})
    nome_empresa = profile.get('companyName') or profile.get('name') or ""
    resultados_google = buscar_google_serpapi(f"{nome_empresa} ações", serpapi_key) if serpapi_key and nome_empresa else []
    contexto_google = "\n".join(resultados_google)

    chat_agent = Agent(
        name="consultor_acoes",
        model="gemini-2.5-flash",
        instruction=f"""
        Você é um CONSULTOR FINANCEIRO EXPERIENTE e DIRECIONADOR, especializado em orientar investidores sobre análise de ações.
        Responda sempre em português do Brasil, de forma educativa, prática e orientativa.

        SEU PAPEL:
        - EDUCAR e DIRECIONAR o investidor para tomadas de decisão mais informadas
        - Ser FLEXÍVEL nas análises, não se limitando apenas aos dados fornecidos
        - CONTEXTUALIZAR as informações com conhecimento geral do mercado
        - ORIENTAR sobre próximos passos e análises adicionais
        - Ser um MENTOR que ensina conceitos financeiros de forma prática

        PERFIL DO CLIENTE:
        - Investidor focado em análise de ações
        - Busca orientação prática e educativa
        - Interessado em: {', '.join(perfil['criterios_importantes'])}
        - Setores de interesse: {', '.join(perfil['setores_interesse']) if perfil['setores_interesse'] else 'Ainda identificando'}

        DADOS DISPONÍVEIS (use como base, mas seja flexível):
        {contexto_financeiro}

        NOTÍCIAS RECENTES:
        {contexto_noticias}

        INFORMAÇÕES ADICIONAIS DA WEB:
        {contexto_google}

        COMO RESPONDER:
        1. ANALISE os dados disponíveis, mas vá além - use seu conhecimento geral
        2. EDUQUE sobre conceitos financeiros relevantes à pergunta
        3. DIRECIONE para análises complementares ou ações práticas
        4. CONTEXTUALIZE com tendências do mercado e setor
        5. Seja HONESTO sobre limitações dos dados, mas ofereça alternativas
        6. NUNCA dê conselhos específicos de compra/venda - oriente sobre processo de análise
        7. ENCORAJE o investidor a buscar mais informações quando necessário
        8. Use EXEMPLOS práticos quando apropriado

        ESTRUTURA SUGERIDA (seja flexível):
        - Resposta direta à pergunta
        - Contexto educativo relevante
        - Análise dos dados (quando disponíveis)
        - Orientações para próximos passos
        - Considerações adicionais importantes
        """
    )

    resposta = await call_agent(chat_agent, pergunta, session_service)
    return resposta

def main():
    st.title("📈 Assistente para Análise de Ações")
    st.markdown("Consultoria especializada para investidores com análise de notícias em tempo real")
    st.markdown("---")
    
    # Debug das APIs na sidebar
    with st.sidebar:
        st.markdown("### 🔍 Status das APIs")
        st.write(f"🤖 Google API: {'✅' if get_google_api_key() else '❌'}")
        st.write(f"📊 FMP API: {'✅' if get_fmp_api_key() else '❌'}")
        st.write(f"📰 GNews API: {'✅' if get_gnews_api_key() else '❌'}")
        st.write(f"🔎 SerpAPI: {'✅' if get_serpapi_key() else '❌'}")
        
        # Configurar Gemini automaticamente
        if configure_gemini_auto():
            st.success("✅ Gemini configurado!")
        else:
            st.warning("⚠️ Google API não configurada")

    # Validar chaves obrigatórias
    if not get_google_api_key() or not get_fmp_api_key():
        st.error("⚠️ Chaves obrigatórias não configuradas (Google API e FMP API)")
        st.info("""
        **Configure suas chaves no painel de secrets do Streamlit Cloud:**
        
        Settings → Secrets:
        ```
        GOOGLE_API_KEY = "sua_chave_google"
        FMP_API_KEY = "sua_chave_fmp"
        GNEWS_API_KEY = "sua_chave_gnews"
        SERPAPI_KEY = "sua_chave_serpapi"
        ```
        """)
        st.stop()
    
    # Inicializar serviços
    session_service = initialize_session_service()
    inicializar_perfil_investidor()
    
    # NOVA PÁGINA INICIAL - EXPLICAÇÃO DO PROJETO
    if not st.session_state.get("mostrar_ferramenta", False):
        st.markdown("""
        # 🎯 Como Funciona Este Assistente?
        
        ### 🤔 **O que faz este sistema?**
        Este assistente foi criado para **ajudar investidores** a analisar ações de empresas americanas de forma simples e inteligente. 
        
        Ele coleta informações financeiras, notícias e gera análises personalizadas - como ter um consultor financeiro particular!
        
        ---
        
        ### 🔧 **Como foi construído?**
        
        **🤖 Inteligência Artificial (IA)**
        - Usa o **Google Gemini** (a IA do Google) para entender suas perguntas e dar respostas personalizadas
        - É como conversar com um especialista em investimentos!
        
        **📊 Dados Financeiros**
        - Conecta com **APIs** (fontes de dados na internet) para buscar:
          - Balanços patrimoniais das empresas
          - Demonstrativos de resultados
          - Fluxo de caixa
          - Histórico de preços
        
        **📰 Notícias em Tempo Real**
        - Busca notícias atualizadas sobre a empresa e o setor
        - Inclui essas informações nas análises
        
        **🔍 Pesquisas no Google**
        - Pode buscar informações adicionais na internet
        - Traz contexto mais amplo sobre a empresa
        
        ---
        
        ### 🚀 **Como começar?**
        
        1. **Digite um ticker** de uma empresa (ex: AAPL, MSFT, TSLA)
        2. **Clique em "Analisar"** para ver os dados
        3. **Use o chat** para fazer perguntas específicas
        4. **Explore as diferentes abas** com análises detalhadas
        
        ---
        """)
        
        # Seção de início
        st.markdown("### 🎯 **Pronto para começar?**")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 Começar a Usar o Assistente", 
                type="primary", 
                use_container_width=True,
                help="Clique para acessar a ferramenta de análise"
            ):
                st.session_state.mostrar_ferramenta = True
                st.rerun()
        
        return  # Para aqui, não mostra o resto da ferramenta
    
    # Input do ticker da empresa
    ticker_input = st.text_input("Digite o ticker da empresa (ex: AAPL, MSFT, TSLA)", key="ticker_input")

    # Área principal
    if not ticker_input:
        st.warning("⚠️ Por favor, digite o ticker de uma empresa para analisar.")
        return

    # Validar ticker input
    if not ticker_input or len(ticker_input.strip()) < 1:
        st.warning("⚠️ Por favor, digite um ticker válido.")
        return
    
    ticker_input = ticker_input.strip().upper()
    if not re.match(r'^[A-Z]{1,5}$', ticker_input):
        st.warning("⚠️ Ticker deve conter apenas letras (1-5 caracteres).")
        return

    # Se já realizou a análise, mantenha os dados e as abas abertas
    if st.session_state.get("analise_realizada") and st.session_state.get("ticker_analise") == ticker_input:
        dados_fmp = st.session_state.get("dados_analise", {}).get("dados_fmp")
        profile = st.session_state.get("dados_analise", {}).get("profile")
        
        # Armazenar dados no session_state para o chat
        st.session_state.dados_analise = {
            'ticker': ticker_input,
            'profile': profile,
            'dados_fmp': dados_fmp
        }
        
        st.success(f"✅ Dados coletados para {profile.get('companyName', ticker_input)}")
        
        # CRIAR ABAS ORGANIZADAS
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Gráficos Históricos", 
            "📈 Indicadores Financeiros",
            "💰 DRE (Demonstrativo)",
            "🏦 Balanço Patrimonial", 
            "💸 Fluxo de Caixa (DFC)",
            "ℹ️ Perfil da Empresa",
            "📰 Notícias & Análise",
            "💬 Consultor de Ações"
        ])

        # TAB 1: GRÁFICOS HISTÓRICOS
        with tab1:
            st.subheader(f"📊 Análise Gráfica - {ticker_input}")
            st.markdown("**Fonte:** Financial Modeling Prep (FMP)")
            
            fig_historicos = criar_graficos_historicos_fmp(dados_fmp, ticker_input)
            st.plotly_chart(fig_historicos, use_container_width=True)
            
            # Métricas resumo
            historical = dados_fmp.get("historical", {})
            if historical and 'historical' in historical:
                dados_recentes = historical['historical'][0]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Preço Atual", f"${dados_recentes.get('close', 0):.2f}")
                col2.metric("Variação Dia", f"{dados_recentes.get('changePercent', 0):.2f}%")
                col3.metric("Volume", f"{dados_recentes.get('volume', 0):,}")
                col4.metric("Máxima 52S", f"${profile.get('range', 'N/A')}")

        # TAB 2: INDICADORES FINANCEIROS
        with tab2:
            st.subheader("📈 Indicadores Financeiros Principais")
            st.markdown("**Fonte:** Financial Modeling Prep (FMP)")
            
            indicadores = calcular_indicadores_fmp(dados_fmp)
            
            if indicadores:
                # Exibir em métricas organizadas
                st.markdown("### 📊 Margens de Rentabilidade")
                col1, col2, col3 = st.columns(3)
                col1.metric("Margem Bruta", f"{indicadores['Margem Bruta (%)']:.2f}%")
                col2.metric("Margem Operacional", f"{indicadores['Margem Operacional (%)']:.2f}%")
                col3.metric("Margem Líquida", f"{indicadores['Margem Líquida (%)']:.2f}%")
                
                st.markdown("### 🎯 Indicadores de Retorno")
                col1, col2, col3 = st.columns(3)
                col1.metric("ROE", f"{indicadores['ROE (%)']:.2f}%")
                col2.metric("ROA", f"{indicadores['ROA (%)']:.2f}%")
                col3.metric("P/L Ratio", f"{indicadores['P/L Ratio']:.2f}")
                
                st.markdown("### 💼 Indicadores de Solidez")
                col1, col2, col3 = st.columns(3)
                col1.metric("Debt-to-Equity", f"{indicadores['Debt-to-Equity']:.2f}")
                col2.metric("Current Ratio", f"{indicadores['Current Ratio']:.2f}")
                col3.metric("EPS", f"${indicadores['EPS (USD)']:.2f}")
                
                st.metric("Crescimento Receita YoY", f"{indicadores['Crescimento Receita YoY (%)']:.2f}%")
                
                # Tabela completa
                st.markdown("### 📋 Tabela Completa de Indicadores")
                df_indicadores = pd.DataFrame(list(indicadores.items()), columns=['Indicador', 'Valor'])
                st.dataframe(df_indicadores, use_container_width=True)
            else:
                st.error("Não foi possível calcular os indicadores.")

        # TAB 3: DRE
        with tab3:
            st.subheader("💰 Demonstrativo de Resultados (DRE)")
            st.markdown("**Fonte:** Financial Modeling Prep (FMP)")
            
            income = dados_fmp.get("income", [])
            if income and len(income) > 0:
                # ANÁLISES NA PARTE SUPERIOR
                st.markdown("### 📊 Análises Principais")
                
                # Dados dos últimos anos para análise
                income_atual = income[0]
                income_anterior = income[1] if len(income) > 1 else {}
                
                # Métricas principais
                receita = income_atual.get('revenue', 0)
                custo_vendas = income_atual.get('costOfRevenue', 0)
                lucro_bruto = receita - custo_vendas
                lucro_operacional = income_atual.get('operatingIncome', 0)
                ebitda = income_atual.get('ebitda', 0)
                lucro_liquido = income_atual.get('netIncome', 0)
                
                # Exibir métricas organizadas
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Receita Bruta", formatar_moeda_usd(receita))
                col2.metric("Lucro Bruto", formatar_moeda_usd(lucro_bruto))
                col3.metric("EBITDA", formatar_moeda_usd(ebitda))
                col4.metric("Lucro Líquido", formatar_moeda_usd(lucro_liquido))
                
                # Margens
                st.markdown("### 📈 Margens de Rentabilidade")
                col1, col2, col3 = st.columns(3)
                
                margem_bruta = (lucro_bruto / receita * 100) if receita > 0 else 0
                margem_operacional = (lucro_operacional / receita * 100) if receita > 0 else 0
                margem_liquida = (lucro_liquido / receita * 100) if receita > 0 else 0
                
                col1.metric("Margem Bruta", f"{margem_bruta:.1f}%")
                col2.metric("Margem Operacional", f"{margem_operacional:.1f}%")
                col3.metric("Margem Líquida", f"{margem_liquida:.1f}%")
                
                # Crescimento YoY
                if income_anterior:
                    receita_anterior = income_anterior.get('revenue', 0)
                    crescimento_receita = ((receita - receita_anterior) / receita_anterior * 100) if receita_anterior > 0 else 0
                    st.metric("Crescimento da Receita (YoY)", f"{crescimento_receita:.1f}%")
                
                # Gráfico de evolução histórica
                st.markdown("### 📊 Evolução Histórica (5 Anos)")
                df_dre = pd.DataFrame(income[:5])
                
                fig_evolucao = go.Figure()
                fig_evolucao.add_trace(go.Scatter(
                    x=df_dre['date'], 
                    y=df_dre['revenue'], 
                    name='Receita', 
                    line=dict(color='blue', width=3)
                ))
                fig_evolucao.add_trace(go.Scatter(
                    x=df_dre['date'], 
                    y=df_dre['grossProfit'], 
                    name='Lucro Bruto', 
                    line=dict(color='green', width=2)
                ))
                fig_evolucao.add_trace(go.Scatter(
                    x=df_dre['date'], 
                    y=df_dre['netIncome'], 
                    name='Lucro Líquido', 
                    line=dict(color='red', width=2)
                ))
                
                fig_evolucao.update_layout(
                    title='Evolução da Receita e Lucros',
                    height=400,
                    xaxis_title='Ano',
                    yaxis_title='Valor (USD)'
                )
                st.plotly_chart(fig_evolucao, use_container_width=True)
                
                # DRE ESTRUTURADA (DE CIMA PARA BAIXO)
                st.markdown("### 📋 Demonstrativo Estruturado (Último Ano)")
                
                # Criar DRE estruturada
                dre_estruturada = {
                    "Receita Bruta": formatar_moeda_usd(receita),
                    "(-) Custo das Vendas": formatar_moeda_usd(-custo_vendas),
                    "= Lucro Bruto": formatar_moeda_usd(lucro_bruto),
                    "(-) Despesas Operacionais": formatar_moeda_usd(-income_atual.get('operatingExpenses', 0)),
                    "= Lucro Operacional": formatar_moeda_usd(lucro_operacional),
                    "EBITDA": formatar_moeda_usd(ebitda),
                    "(-) Despesas Financeiras": formatar_moeda_usd(-income_atual.get('interestExpense', 0)),
                    "= Lucro Antes do IR": formatar_moeda_usd(income_atual.get('incomeBeforeTax', 0)),
                    "(-) Imposto de Renda": formatar_moeda_usd(-income_atual.get('incomeTaxExpense', 0)),
                    "= Lucro Líquido": formatar_moeda_usd(lucro_liquido),
                    "EPS (por ação)": f"$ {income_atual.get('eps', 0):.2f}"
                }
                
                df_dre_estruturada = pd.DataFrame(list(dre_estruturada.items()), columns=['Item', 'Valor'])
                st.dataframe(df_dre_estruturada, use_container_width=True, hide_index=True)
                
            else:
                st.info("DRE não disponível para esta empresa.")

        # TAB 4: BALANÇO
        with tab4:
            st.subheader("🏦 Balanço Patrimonial")
            st.markdown("**Fonte:** Financial Modeling Prep (FMP)")
            
            balance = dados_fmp.get("balance", [])
            if balance and len(balance) > 0:
                balance_atual = balance[0]
                
                # ANÁLISES SUPERIORES
                st.markdown("### 📊 Principais Indicadores")
                
                # Dados principais
                ativo_total = balance_atual.get('totalAssets', 0)
                ativo_circulante = balance_atual.get('totalCurrentAssets', 0)
                passivo_total = balance_atual.get('totalLiabilities', 0)
                passivo_circulante = balance_atual.get('totalCurrentLiabilities', 0)
                patrimonio_liquido = balance_atual.get('totalStockholdersEquity', 0)
                divida_total = balance_atual.get('totalDebt', 0)
                
                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ativo Total", formatar_moeda_usd(ativo_total))
                col2.metric("Patrimônio Líquido", formatar_moeda_usd(patrimonio_liquido))
                col3.metric("Dívida Total", formatar_moeda_usd(divida_total))
                col4.metric("Ativo Circulante", formatar_moeda_usd(ativo_circulante))
                
                # Índices de análise
                st.markdown("### 📈 Indicadores de Solidez")
                col1, col2, col3 = st.columns(3)
                
                current_ratio = (ativo_circulante / passivo_circulante) if passivo_circulante > 0 else 0
                debt_to_equity = (divida_total / patrimonio_liquido) if patrimonio_liquido > 0 else 0
                endividamento = (passivo_total / ativo_total * 100) if ativo_total > 0 else 0
                
                col1.metric("Current Ratio", f"{current_ratio:.2f}")
                col2.metric("Debt-to-Equity", f"{debt_to_equity:.2f}")
                col3.metric("Índice de Endividamento", f"{endividamento:.1f}%")
                
                # Gráfico de composição
                st.markdown("### 📊 Composição do Balanço")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de Ativos
                    fig_ativos = go.Figure(data=[go.Pie(
                        labels=['Ativo Circulante', 'Ativo Não Circulante'],
                        values=[ativo_circulante, ativo_total - ativo_circulante],
                        title="Composição dos Ativos"
                    )])
                    st.plotly_chart(fig_ativos, use_container_width=True)
                
                with col2:
                    # Gráfico de Passivos + PL
                    fig_passivos = go.Figure(data=[go.Pie(
                        labels=['Passivo Circulante', 'Passivo Não Circulante', 'Patrimônio Líquido'],
                        values=[passivo_circulante, passivo_total - passivo_circulante, patrimonio_liquido],
                        title="Passivo + Patrimônio Líquido"
                    )])
                    st.plotly_chart(fig_passivos, use_container_width=True)
                
                # BALANÇO ESTRUTURADO (ATIVO | PASSIVO + PL)
                st.markdown("### ⚖️ Balanço Estruturado")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 ATIVO")
                    ativo_estruturado = {
                        "ATIVO CIRCULANTE": "",
                        "Caixa e Equivalentes": formatar_moeda_usd(balance_atual.get('cashAndCashEquivalents', 0)),
                        "Aplicações Financeiras": formatar_moeda_usd(balance_atual.get('shortTermInvestments', 0)),
                        "Contas a Receber": formatar_moeda_usd(balance_atual.get('netReceivables', 0)),
                        "Estoques": formatar_moeda_usd(balance_atual.get('inventory', 0)),
                        "Outros Ativos Circulantes": formatar_moeda_usd(balance_atual.get('otherCurrentAssets', 0)),
                        "TOTAL ATIVO CIRCULANTE": formatar_moeda_usd(ativo_circulante),
                        "": "",
                        "ATIVO NÃO CIRCULANTE": "",
                        "Investimentos": formatar_moeda_usd(balance_atual.get('longTermInvestments', 0)),
                        "Imobilizado": formatar_moeda_usd(balance_atual.get('propertyPlantEquipmentNet', 0)),
                        "Intangível": formatar_moeda_usd(balance_atual.get('intangibleAssets', 0)),
                        "TOTAL ATIVO": formatar_moeda_usd(ativo_total)
                    }
                    
                    df_ativo = pd.DataFrame(list(ativo_estruturado.items()), columns=['Item', 'Valor'])
                    st.dataframe(df_ativo, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### 📉 PASSIVO + PATRIMÔNIO LÍQUIDO")
                    passivo_estruturado = {
                        "PASSIVO CIRCULANTE": "",
                        "Contas a Pagar": formatar_moeda_usd(balance_atual.get('accountPayables', 0)),
                        "Dívidas de Curto Prazo": formatar_moeda_usd(balance_atual.get('shortTermDebt', 0)),
                        "Outros Passivos Circulantes": formatar_moeda_usd(balance_atual.get('otherCurrentLiabilities', 0)),
                        "TOTAL PASSIVO CIRCULANTE": formatar_moeda_usd(passivo_circulante),
                        "": "",
                        "PASSIVO NÃO CIRCULANTE": "",
                        "Dívidas de Longo Prazo": formatar_moeda_usd(balance_atual.get('longTermDebt', 0)),
                        "Outros Passivos": formatar_moeda_usd(balance_atual.get('otherLiabilities', 0)),
                        "TOTAL PASSIVO": formatar_moeda_usd(passivo_total),
                        " ": "",
                        "PATRIMÔNIO LÍQUIDO": "",
                        "Capital Social": formatar_moeda_usd(balance_atual.get('commonStock', 0)),
                        "Lucros Acumulados": formatar_moeda_usd(balance_atual.get('retainedEarnings', 0)),
                        "TOTAL PATRIMÔNIO LÍQUIDO": formatar_moeda_usd(patrimonio_liquido),
                        "  ": "",
                        "TOTAL PASSIVO + PL": formatar_moeda_usd(passivo_total + patrimonio_liquido)
                    }
                    
                    df_passivo = pd.DataFrame(list(passivo_estruturado.items()), columns=['Item', 'Valor'])
                    st.dataframe(df_passivo, use_container_width=True, hide_index=True)
            
            else:
                st.info("Balanço patrimonial não disponível para esta empresa.")

        # TAB 5: FLUXO DE CAIXA - CORREÇÃO COMPLETA
        with tab5:
            st.subheader("💸 Demonstrativo de Fluxo de Caixa (DFC)")
            st.markdown("**Fonte:** Financial Modeling Prep (FMP)")
            
            cashflow = dados_fmp.get("cashflow", [])

            if cashflow and len(cashflow) > 0:
                cf_atual = cashflow[0]
                
                # ANÁLISES SUPERIORES - FOCO NO OPERACIONAL
                st.markdown("### 💰 Análise do Fluxo de Caixa Operacional")
                
                # Dados principais
                fc_operacional = cf_atual.get('operatingCashFlow', 0) or cf_atual.get('netCashProvidedByOperatingActivities', 0)
                fc_investimento = cf_atual.get('investingCashFlow', 0) or cf_atual.get('netCashUsedForInvestingActivites', 0) or cf_atual.get('netCashUsedProvidedByInvestingActivities', 0)
                fc_financiamento = cf_atual.get('financingCashFlow', 0) or cf_atual.get('netCashUsedProvidedByFinancingActivities', 0)
                fc_livre = fc_operacional + fc_investimento
                
                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Fluxo Operacional", formatar_moeda_usd(fc_operacional))
                col2.metric("Fluxo de Investimento", formatar_moeda_usd(fc_investimento))
                col3.metric("Fluxo de Financiamento", formatar_moeda_usd(fc_financiamento))
                col4.metric("Fluxo de Caixa Livre", formatar_moeda_usd(fc_livre))
                
                # Análises e proporções
                st.markdown("### 📊 Indicadores de Qualidade")
                
                # Obter lucro líquido para comparação
                income = dados_fmp.get("income", [])
                lucro_liquido = income[0].get('netIncome', 0) if income else 0
                
                col1, col2, col3 = st.columns(3)
                
                # Proporção FC Operacional vs Lucro Líquido
                conversao_caixa = (fc_operacional / lucro_liquido * 100) if lucro_liquido > 0 else 0
                col1.metric("Conversão Caixa/Lucro", f"{conversao_caixa:.1f}%")
                
                # Capacidade de autofinanciamento
                autofinanciamento = "Positiva" if fc_livre > 0 else "Negativa"
                col2.metric("Autofinanciamento", autofinanciamento)
                
                # Dependência de financiamento externo
                dependencia = "Alta" if abs(fc_financiamento) > fc_operacional else "Baixa"
                col3.metric("Dependência Externa", dependencia)
                
                # Gráfico histórico dos fluxos
                st.markdown("### 📈 Evolução dos Fluxos (5 Anos)")
                
                df_dfc = pd.DataFrame(cashflow[:5])
                
                fig_fluxos = go.Figure()
                
                # Fluxo Operacional
                if 'operatingCashFlow' in df_dfc.columns:
                    fig_fluxos.add_trace(go.Bar(
                        x=df_dfc['date'],
                        y=df_dfc['operatingCashFlow'],
                        name='Fluxo Operacional',
                        marker_color='blue'
                    ))
                elif 'netCashProvidedByOperatingActivities' in df_dfc.columns:
                    fig_fluxos.add_trace(go.Bar(
                        x=df_dfc['date'],
                        y=df_dfc['netCashProvidedByOperatingActivities'],
                        name='Fluxo Operacional',
                        marker_color='blue'
                    ))
                
                # Fluxo de Investimento
                if 'investingCashFlow' in df_dfc.columns:
                    fig_fluxos.add_trace(go.Bar(
                        x=df_dfc['date'],
                        y=df_dfc['investingCashFlow'],
                        name='Fluxo de Investimento',
                        marker_color='red'
                    ))
                elif 'netCashUsedForInvestingActivites' in df_dfc.columns:
                    fig_fluxos.add_trace(go.Bar(
                        x=df_dfc['date'],
                        y=df_dfc['netCashUsedForInvestingActivites'],
                        name='Fluxo de Investimento',
                        marker_color='red'
                    ))
                elif 'netCashUsedProvidedByInvestingActivities' in df_dfc.columns:
                    fig_fluxos.add_trace(go.Bar(
                        x=df_dfc['date'],
                        y=df_dfc['netCashUsedProvidedByInvestingActivities'],
                        name='Fluxo de Investimento',
                        marker_color='red'
                    ))
                
                # Fluxo de Financiamento
                if 'financingCashFlow' in df_dfc.columns:
                    fig_fluxos.add_trace(go.Bar(
                        x=df_dfc['date'],
                        y=df_dfc['financingCashFlow'],
                        name='Fluxo de Financiamento',
                        marker_color='green'
                    ))
                elif 'netCashUsedProvidedByFinancingActivities' in df_dfc.columns:
                    fig_fluxos.add_trace(go.Bar(
                        x=df_dfc['date'],
                        y=df_dfc['netCashUsedProvidedByFinancingActivities'],
                        name='Fluxo de Financiamento',
                        marker_color='green'
                    ))
                
                fig_fluxos.update_layout(
                    title='Evolução dos Fluxos de Caixa',
                    height=400,
                    xaxis_title='Ano',
                    yaxis_title='Valor (USD)'
                )
                st.plotly_chart(fig_fluxos, use_container_width=True)
                
                # DFC ESTRUTURADA (DE CIMA PARA BAIXO)
                st.markdown("### 📋 Fluxo de Caixa Estruturado (Último Ano)")
                
                dfc_estruturada = {
                    "FLUXO DE CAIXA OPERACIONAL": "",
                    "Lucro Líquido": formatar_moeda_usd(lucro_liquido),
                    "(+) Depreciação e Amortização": formatar_moeda_usd(cf_atual.get('depreciationAndAmortization', 0)),
                    "(+/-) Variação Capital de Giro": formatar_moeda_usd(cf_atual.get('changeInWorkingCapital', 0)),
                    "(+/-) Outros Ajustes": formatar_moeda_usd(cf_atual.get('otherWorkingCapital', 0)),
                    "= CAIXA GERADO OPERAÇÕES": formatar_moeda_usd(fc_operacional),
                    "": "",
                    "FLUXO DE CAIXA INVESTIMENTO": "",
                    "(-) Investimentos em Ativo Fixo": formatar_moeda_usd(cf_atual.get('capitalExpenditure', 0)),
                    "(+/-) Aquisições e Vendas": formatar_moeda_usd(cf_atual.get('acquisitionsNet', 0)),
                    "(+/-) Outros Investimentos": formatar_moeda_usd(cf_atual.get('purchasesOfInvestments', 0)),
                    "= CAIXA USADO INVESTIMENTOS": formatar_moeda_usd(fc_investimento),
                    " ": "",
                    "FLUXO DE CAIXA FINANCIAMENTO": "",
                    "(+/-) Emissão/Recompra Ações": formatar_moeda_usd(cf_atual.get('commonStockIssued', 0)),
                    "(+/-) Pagamento Dividendos": formatar_moeda_usd(cf_atual.get('dividendsPaid', 0)),
                    "(+/-) Variação Dívidas": formatar_moeda_usd(cf_atual.get('debtRepayment', 0)),
                    "= CAIXA DE FINANCIAMENTOS": formatar_moeda_usd(fc_financiamento),
                    "  ": "",
                    "VARIAÇÃO LÍQUIDA DE CAIXA": formatar_moeda_usd(cf_atual.get('netChangeInCash', 0)),
                    "FLUXO DE CAIXA LIVRE": formatar_moeda_usd(fc_livre)
                }
                
                df_dfc_estruturado = pd.DataFrame(list(dfc_estruturada.items()), columns=['Item', 'Valor'])
                st.dataframe(df_dfc_estruturada, use_container_width=True, hide_index=True)
                
            else:
                st.info("Fluxo de caixa não disponível para esta empresa.")

        # TAB 6: PERFIL DA EMPRESA
        with tab6:
            st.subheader("ℹ️ Perfil da Empresa")
            st.markdown("**Fonte:** Financial Modeling Prep (FMP)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Nome:** {profile.get('companyName', 'N/A')}")
                st.markdown(f"**Ticker:** {profile.get('symbol', 'N/A')}")
                st.markdown(f"**Setor:** {profile.get('sector', 'N/A')}")
                st.markdown(f"**Indústria:** {profile.get('industry', 'N/A')}")
                st.markdown(f"**País:** {profile.get('country', 'N/A')}")
                st.markdown(f"**Website:** {profile.get('website', 'N/A')}")
            
            with col2:
                st.markdown(f"**Cap. de Mercado:** {formatar_numero(profile.get('mktCap', 0), '${:,}')}")
                st.markdown(f"**Preço:** {formatar_numero(profile.get('price', 0), '${:.2f}')}")
                st.markdown(f"**P/L:** {profile.get('pe', 'N/A')}")
                st.markdown(f"**Beta:** {profile.get('beta', 'N/A')}")
                st.markdown(f"**Dividend Yield:** {profile.get('lastDiv', 'N/A')}")
                st.markdown(f"**Funcionários:** {formatar_numero(profile.get('fullTimeEmployees', 0))}")
            
            st.markdown("**Descrição:**")
            st.markdown(profile.get('description', 'Descrição não disponível.'))

            # TAB 7: NOTÍCIAS & ANÁLISE
            with tab7:
                st.subheader("📰 Notícias e Análise de Mercado")
                st.markdown("**Fonte:** GNews")
                company_name = profile.get('companyName', ticker_input)
                sector = profile.get('sector', '')
                noticias_empresa = obter_noticias_empresa(ticker_input, company_name)
                noticias_setor = obter_noticias_setor(sector) if sector else []
                if noticias_empresa:
                    st.markdown("### 🏢 Notícias da Empresa")
                    for noticia in noticias_empresa[:5]:
                        with st.expander(f"📰 {noticia['title'][:100]}..."):
                            st.markdown(f"**Fonte:** {noticia['source']['name'] if noticia.get('source') else ''}")
                            st.markdown(f"**Data:** {noticia.get('publishedAt', '')[:10]}")
                            st.markdown(f"**Descrição:** {noticia.get('description', '')}")
                            if noticia.get('url'):
                                st.markdown(f"[Ler notícia completa]({noticia['url']})")
                if noticias_setor:
                    st.markdown(f"### 🏭 Notícias do Setor: {sector}")
                    for noticia in noticias_setor[:3]:
                        with st.expander(f"📰 {noticia['title'][:100]}..."):
                            st.markdown(f"**Fonte:** {noticia['source']['name'] if noticia.get('source') else ''}")
                            st.markdown(f"**Data:** {noticia.get('publishedAt', '')[:10]}")
                            st.markdown(f"**Descrição:** {noticia.get('description', '')}")
                if not noticias_empresa and not noticias_setor:
                    st.info("📰 Configure sua GNews API Key para ver análises de notícias relevantes.")
                st.session_state.noticias_contexto = noticias_empresa + noticias_setor

            # TAB 8: CONSULTOR DE ANÁLISE DE AÇÕES
            with tab8:
                st.subheader("💬 Consultor Especializado - Análise de Ações")
                st.markdown("**Powered by:** Google Gemini AI + Análise de Notícias")
                
                inicializar_perfil_investidor()
                
                st.info("💡 Consultoria especializada para investidores. Pergunte sobre dividendos, estabilidade, riscos e adequação da empresa ao seu perfil.")
                
                with st.expander("👤 Seu Perfil de Investidor"):
                    perfil = st.session_state.perfil_investidor
                    st.markdown(f"**Tipo:** {perfil['tipo'].title()}")
                    st.markdown(f"**Critérios Importantes:** {', '.join(perfil['criterios_importantes'])}")
                    if perfil['setores_interesse']:
                        st.markdown(f"**Setores de Interesse:** {', '.join(perfil['setores_interesse'])}")
                
                if 'ultima_pergunta' not in st.session_state:
                    st.session_state.ultima_pergunta = ""
                if 'ultima_resposta' not in st.session_state:
                    st.session_state.ultima_resposta = ""
                
                if st.session_state.ultima_pergunta:
                    with st.chat_message("user"):
                        st.markdown(st.session_state.ultima_pergunta)
                    with st.chat_message("assistant"):
                        st.markdown(st.session_state.ultima_resposta)
                
                # CORREÇÃO: Verificar chave do Google corretamente
                if not get_google_api_key():
                    st.error("🔑 Google API Key não configurada nos secrets do Streamlit Cloud.")
                    st.info("""
                    **Configure sua Google API Key:**
                    
                    1. Vá em **Settings → Secrets** no painel do Streamlit Cloud
                    2. Adicione/verifique:
                    ```
                    GOOGLE_API_KEY = "sua_chave_google_real"
                    ```
                    3. Salve e o app reiniciará automaticamente
                    """)
                else:
                    # Chat funcional
                    if prompt := st.chat_input("Digite sua pergunta sobre análise de ações..."):
                        st.session_state.ultima_pergunta = prompt
                        
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        with st.chat_message("assistant"):
                            with st.spinner("Analisando..."):
                                try:
                                    dados_contexto = st.session_state.get('dados_analise', {})
                                    noticias = st.session_state.get('noticias_contexto', [])
                                    resposta = asyncio.run(chat_consultor_acoes(prompt, dados_contexto, noticias, session_service))
                                    st.markdown(resposta)
                                    st.session_state.ultima_resposta = resposta
                                except Exception as e:
                                    st.error(f"Erro no chat: {str(e)}")
                                    st.session_state.ultima_resposta = f"Erro ao processar sua pergunta: {str(e)}"
                
                if st.button("🔄 Nova Consulta"):
                    st.session_state.ultima_pergunta = ""
                    st.session_state.ultima_resposta = ""
                    st.rerun()

    # BOTÃO PARA ANALISAR (fora do bloco das abas)
    else:
        if st.button(f"🔮 Analisar {ticker_input}", type="primary", use_container_width=True):
            with st.spinner(f"Coletando dados completos de {ticker_input}..."):
                try:
                    # Obter dados FMP completos
                    dados_fmp = obter_dados_fmp_completos(ticker_input)
                    
                    profile = dados_fmp.get("profile", {})
                    
                    # Verificar se profile tem dados válidos
                    if not profile:
                        st.error("❌ Profile vazio - verifique o ticker ou tente novamente.")
                        return
                        
                    if not (profile.get("companyName") or profile.get("name") or profile.get("symbol")):
                        st.error("❌ Ticker não encontrado. Verifique se está correto.")
                        return
                    
                    # Armazenar dados no session_state
                    st.session_state.dados_analise = {
                        'ticker': ticker_input,
                        'profile': profile,
                        'dados_fmp': dados_fmp
                    }
                    st.session_state.analise_realizada = True
                    st.session_state.ticker_analise = ticker_input

                    company_name = profile.get('companyName') or profile.get('name') or ticker_input
                    st.success(f"✅ Dados coletados para {company_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro durante a análise: {str(e)}")

    # Rodapé
    st.markdown("---")
    st.markdown("### ⚠️ Avisos Importantes")
    st.warning("""
    - Esta ferramenta é apenas para fins educacionais
    - Não constitui recomendação de investimento
    - Sempre consulte um consultor financeiro qualificado
    - Faça sua própria pesquisa antes de investir
    """)

# Execute a função principal
if __name__ == "__main__":
    main()