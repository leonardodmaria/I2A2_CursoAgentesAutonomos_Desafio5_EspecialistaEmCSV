import openai
import streamlit as st
from agent import CSVAIAgent

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Agente CSV Inteligente", layout="wide")
st.title("🤖 Agente Inteligente de Análise de CSV")

# Inicializar o agente e histórico no session_state
if "agent" not in st.session_state:
    st.session_state.agent = CSVAIAgent()
    st.session_state.last_uploaded_file = None
    st.session_state.messages = []  # Histórico do chat
    st.session_state.chat_disabled = True  # Chat desabilitado até ter CSV

# Upload de CSV
uploaded_file = st.file_uploader("📂 Faça upload do seu arquivo CSV", type=["csv"])

# Verificar se um novo arquivo foi carregado
if uploaded_file:
    # Verificar se é um arquivo diferente do último carregado
    if uploaded_file != st.session_state.last_uploaded_file:
        # Criar nova instância do agente para o novo arquivo
        st.session_state.agent = CSVAIAgent()
        st.session_state.last_uploaded_file = uploaded_file
        st.session_state.chat_disabled = False  # Habilitar chat

        # Carregar o CSV no agente
        with st.spinner("Carregando arquivo CSV..."):
            df = st.session_state.agent.load_csv(uploaded_file)
            st.session_state.df = df
        st.success("✅ Arquivo carregado com sucesso!")

        # Apagar memória e histórico
        st.session_state.agent.memory = []
        st.session_state.messages = []

    # Exibir informações do arquivo
    df = st.session_state.df

    # Expander para visualização rápida dos dados
    with st.expander("📊 Visualizar Dados (Primeiras 10 linhas)"):
        st.dataframe(df.head(10))

    # Container principal do chat
    st.markdown("---")
    st.subheader("💬 Chat")

    # Exibir histórico do chat
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Mostrar gráficos se existirem na mensagem
                if "plot" in message and message["plot"] is not None:
                    # Reduzir o tamanho da imagem
                    st.image(message["plot"], use_container_width=False, width=400)

    # Input do usuário
    if not st.session_state.chat_disabled:
        # Campo de entrada de mensagem
        user_input = st.chat_input("Digite sua pergunta sobre os dados...")

        if user_input:
            # Adicionar mensagem do usuário ao histórico
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Exibir mensagem do usuário imediatamente
            with st.chat_message("user"):
                st.markdown(user_input)

            # Gerar resposta do agente
            with st.chat_message("assistant"):
                with st.spinner("Analisando dados..."):
                    # Verificar se é um comando de gráfico
                    if any(keyword in user_input.lower() for keyword in
                           ["gráfico", "grafico", "plot", "histograma", "dispersão", "correlação"]):
                        # Tentar detectar tipo de gráfico e colunas
                        response, plot_buffer = st.session_state.agent.generate_plot_from_text(user_input)
                    else:
                        # Pergunta normal
                        response = st.session_state.agent.ask_chatgpt(user_input)
                        plot_buffer = None

                # Exibir resposta
                st.markdown(response)

                # Exibir gráfico se gerado (com tamanho reduzido)
                if plot_buffer:
                    st.image(plot_buffer, use_container_width=False, width=400)  # Tamanho reduzido

            # Adicionar resposta ao histórico
            new_message = {"role": "assistant", "content": response, "plot": plot_buffer}
            st.session_state.messages.append(new_message)

            # Rolar automaticamente para a última mensagem
            st.rerun()

    # Sidebar com informações adicionais
    with st.sidebar:
        st.header("⚙️ Controles")

        # Botão para limpar chat
        if st.button("🗑️ Limpar Chat", use_container_width=True):

            # Apagar memória e histórico
            st.session_state.agent.memory = []
            st.session_state.messages = []

            st.rerun()

        st.markdown("---")
        st.header("📈 Gerar Gráfico")

        plot_type = st.selectbox("Tipo de gráfico",
                                 ["histograma", "dispersao", "correlacao"])  # Removido boxplot

        if plot_type == "histograma":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                column = st.selectbox("Selecione a coluna", numeric_cols)
                if st.button("Gerar", use_container_width=True):  # Texto simplificado
                    with st.spinner("Gerando histograma..."):
                        buf = st.session_state.agent.generate_plot("histograma", column)

                    # Adicionar ao chat
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Gerar histograma da coluna {column}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Aqui está o histograma da coluna **{column}**:",
                        "plot": buf
                    })
                    st.rerun()
            else:
                st.warning("Nenhuma coluna numérica encontrada.")

        elif plot_type == "dispersao":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Eixo X", numeric_cols)
                col2 = st.selectbox("Eixo Y", numeric_cols)
                if st.button("Gerar", use_container_width=True):  # Texto simplificado
                    with st.spinner("Gerando gráfico de dispersão..."):
                        buf = st.session_state.agent.generate_plot("dispersao", [col1, col2])

                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Gerar gráfico de dispersão: {col1} x {col2}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Aqui está o gráfico de dispersão **{col1} x {col2}**:",
                        "plot": buf
                    })
                    st.rerun()
            else:
                st.warning("É necessário ter pelo menos 2 colunas numéricas.")

        elif plot_type == "correlacao":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(numeric_cols) >= 2:
                if st.button("Gerar", use_container_width=True):  # Texto simplificado
                    with st.spinner("Gerando matriz de correlação..."):
                        buf = st.session_state.agent.generate_plot("correlacao")

                    st.session_state.messages.append({
                        "role": "user",
                        "content": "Gerar matriz de correlação"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Aqui está a matriz de correlação das colunas numéricas:",
                        "plot": buf
                    })
                    st.rerun()
            else:
                st.warning("É necessário ter pelo menos 2 colunas numéricas.")
