import io
import csv
import re
import statistics
import sys
import openai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional

"""
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
"""

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------------------
# Utilitários de CSV / decoding
# ---------------------------
def _try_decode(sample_bytes: bytes):
    """Tenta decodificar bytes com encodings comuns; retorna (texto, encoding)."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return sample_bytes.decode(enc), enc
        except Exception:
            pass
    return sample_bytes.decode("utf-8", errors="replace"), "utf-8-replaced"


def _is_number(s: Any) -> bool:
    """Detecta se uma string representa um número (tratando vírgulas etc.)."""
    try:
        s = str(s).strip()
    except Exception:
        return False
    if s == "":
        return False
    s = s.replace(" ", "").replace(",", ".")
    try:
        float(s)
        return True
    except Exception:
        return False


def _detect_outer_quoted(sample_text: str, min_lines: int = 3) -> bool:
    """
    Detecta se a maioria das primeiras linhas está envolta por aspas duplas externas
    (caso em que o parser verá a linha inteira como um único campo).
    """
    if not sample_text:
        return False
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return False
    sample = lines[:max(min_lines, min(10, len(lines)))]
    starts_ends = [ln.startswith('"') and ln.endswith('"') for ln in sample]
    has_inner_double = ['""' in ln for ln in sample]
    ratio_se = sum(starts_ends) / len(sample)
    ratio_inner = sum(has_inner_double) / len(sample)
    return (ratio_se >= 0.8) and (ratio_inner >= 0.5)


def _unwrap_outer_quotes(text: str) -> str:
    lines = text.splitlines()
    new_lines = []
    for ln in lines:
        if ln.startswith('"') and ln.endswith('"') and len(ln) >= 2:
            inner = ln[1:-1]
            inner = inner.replace('""', '"')
            new_lines.append(inner)
        else:
            new_lines.append(ln)
    return "\n".join(new_lines)


def _detect_delimiter_from_text(sample_text: str):
    """
    Detecta delimitador e se há header a partir de texto (str).
    Retorna (delim, has_header_guess).
    """
    if not sample_text:
        return ",", False
    sniffer = csv.Sniffer()
    common_delims = [",", ";", "\t", "|", ":"]
    try:
        dialect = sniffer.sniff(sample_text, delimiters=common_delims)
        has_header = sniffer.has_header(sample_text)
        return dialect.delimiter, has_header
    except Exception:
        lines = [ln for ln in sample_text.splitlines() if ln.strip()]
        if not lines:
            return ",", False
        scores = {}
        for d in common_delims:
            counts = [len(re.split(re.escape(d), ln)) for ln in lines]
            med = statistics.median(counts)
            stdev = statistics.pstdev(counts) if len(counts) > 1 else 0
            score = med - stdev * 0.5
            scores[d] = (score, med, stdev)
        best = max(scores.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        delim = best[0]
        first_fields = [f.strip() for f in re.split(re.escape(delim), lines[0])]
        header_like = any(not _is_number(x) for x in first_fields)
        return delim, header_like


# ---------------------------
# Classe do Agente
# ---------------------------
class CSVAIAgent:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.memory: List[Dict[str, Any]] = []  # Apenas memória RAM

    def add_memory(self, question: str, answer: str):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
        }
        self.memory.append(entry)
        # Manter só as últimas 20 interações na RAM
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]

    """

    def clear_memory(self):
        self.memory = []
        
    """

    def get_recent_history(self, n: int = 5) -> str:
        recent = self.memory[-n:]
        return "\n".join([f"Q: {m['question']}\nA: {m['answer']}" for m in recent])


    # ---------------------------
    # Carregamento de CSV robusto
    # ---------------------------
    def load_csv(self, file, sample_size: int = 8192, force_header: bool = True, **pd_read_kwargs) -> pd.DataFrame:
        """
        Carrega um CSV detectando encoding/delimiter/cabeçalho de forma robusta.
        - file: path (str) ou file-like (Streamlit uploaded file)
        - force_header: se True, assume header na primeira linha (header=0).
        - pd_read_kwargs: kwargs extras para pandas.read_csv
        """

        # ler todo conteúdo em bytes
        raw_bytes = None
        if isinstance(file, str):
            with open(file, "rb") as f:
                raw_bytes = f.read()
        else:
            try:
                file.seek(0)
            except Exception:
                pass
            raw = file.read()
            if isinstance(raw, str):
                raw_bytes = raw.encode("utf-8")
            else:
                raw_bytes = raw

        # detectar encoding e decodificar
        text, encoding = _try_decode(raw_bytes)
        self.last_detected_encoding = encoding

        # detectar e consertar linhas envoltas em aspas externas (caso problemático)
        if _detect_outer_quoted(text):
            text = _unwrap_outer_quotes(text)

        # detectar delimitador e header a partir do texto limpo (amostra)
        text_sample = "\n".join(text.splitlines()[:max(20, sample_size // 100)])
        delim, has_header_guess = _detect_delimiter_from_text(text_sample)
        self.last_detected_delimiter = delim

        header_arg = 0 if force_header else (0 if has_header_guess else None)
        self.last_detected_header = (header_arg == 0)

        # criar StringIO para garantir leitura correta pelo pandas (engine 'c' espera texto)
        text_stream = io.StringIO(text)

        read_kwargs = {"sep": delim, "header": header_arg}
        # aplicar overrides do usuário (exceto encoding)
        for k, v in pd_read_kwargs.items():
            if k == "encoding":
                continue
            read_kwargs[k] = v

        if "engine" not in read_kwargs:
            read_kwargs["engine"] = "c"
        if read_kwargs.get("engine") == "python" and "low_memory" in read_kwargs:
            read_kwargs.pop("low_memory", None)

        # tentativa principal
        try:
            text_stream.seek(0)
            df = pd.read_csv(text_stream, **read_kwargs)
        except Exception:
            # fallback: autodetect sep com engine python
            text_stream.seek(0)
            fallback_kwargs = {"sep": None, "engine": "python", "header": header_arg}
            user_overrides = {k: v for k, v in pd_read_kwargs.items() if k not in ("sep", "engine", "header", "encoding")}
            fallback_kwargs.update(user_overrides)
            fallback_kwargs.pop("low_memory", None)
            text_stream.seek(0)
            df = pd.read_csv(text_stream, **fallback_kwargs)

        self.df = df
        self.df.index = self.df.index + 1
        #self.clear_memory()

        return df

    # LLM
    def is_question_relevant(self, question: str) -> bool:
        """
        Usa o LLM para classificar a relevância da pergunta em uma escala de 1 a 10.
        """
        if self.df is None:
            return False

        df_cols = ", ".join(self.df.columns.tolist())
        df_shape = f"{len(self.df)} linhas, {len(self.df.columns)} colunas."

        prompt_relevance = f"""
        Você é um avaliador numérico de relevância.
        
        Tarefa: Avalie a relevância da pergunta do usuário para o arquivo CSV em uma escala de 1 (totalmente irrelevante) a 10 (altamente relevante).

        Contexto do CSV: O arquivo tem dados com as seguintes colunas:
        "{df_cols}"
        
        Tamanho do DataFrame:
        "{df_shape}"
        
        Perguntas consideradas altamente relevantes:
        "Quais são os tipos de dados (numéricos, categóricos) do arquivo/planilha?"
        "Qual a distribuição de cada variável (histogramas, distribuições)?"
        "Qual o intervalo de cada variável (mínimo, máximo)?"
        "Quais são as medidas de tendência central (média, mediana)?"
        "Qual a variabilidade dos dados (desvio padrão, variância)?"
        "Existem padrões ou tendências temporais?"
        "Quais os valores mais frequentes ou menos frequentes?"
        "Existem agrupamentos (clusters) nos dados?"
        "Existem valores atípicos nos dados?"
        "Como esses outliers afetam a análise?"
        "Podem ser removidos, transformados ou investigados?"
        "Como as variáveis estão relacionadas umas com as outras? (Gráficos de dispersão, tabelas cruzadas)"
        "Existe correlação entre as variáveis?"
        "Quais variáveis parecem ter maior ou menor influência sobre outras?"
        "Qual conclusão você pode tirar deste arquivo?"
        "Quantas linhas tem o arquivo/planilha?"
        "Quantas colunas tem o arquivo/planilha?"
        "Qual o tamanho do arquivo/planilha?"
        "Qual sua conclusão sobre o arquivo/planilha?"
        "Com base no histórico de perguntas, qual sua conclusão?"

        Instrução: Responda APENAS com um número inteiro entre 1 e 10 (exemplo de resposta: 5).

        Pergunta do usuário:
        "{question}"
        """

        try:
            # Manter temperatura baixa para obter um número estável
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um avaliador numérico de relevância."},
                    {"role": "user", "content": prompt_relevance},
                ],
                temperature=0.0,
                max_tokens=3  # Suficiente para um número (ex: 10)
            )
            answer = response.choices[0].message["content"].strip()

            # Tentar extrair o número
            match = re.search(r'\d+', answer)
            if match:
                relevance_score = int(match.group(0))
                print("Relevância da pergunta (de 0 a 10): " + str(relevance_score))
                # Defina o limiar de corte
                return relevance_score >= 3

            return False  # Se não conseguir extrair o número

        except Exception:
            # Em caso de falha, é mais seguro não responder
            return False


    def execute_code(self, code_string: str) -> str:
        """Executa código Python no contexto do DataFrame de forma segura."""
        if self.df is None:
            return "Erro: DataFrame não carregado."
        
        # O dataframe deve estar disponível no ambiente de execução
        local_vars = {'df': self.df}
        
        # Captura stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            # Garante que o código é executado e o resultado final é capturado
            exec(code_string, {}, local_vars)
            
            # Tenta pegar a última variável/resultado, se não houver print explícito
            result = local_vars.get('result', None) or redirected_output.getvalue().strip()
            
            if result:
                # Limita a saída para evitar sobrecarga de tokens (ex: print de DF completo)
                result_str = str(result)
                if len(result_str) > 2000:
                    return f"Resultado (Parcial):\n{result_str[:2000]}..."
                return f"Resultado:\n{result_str}"
            
            return "Execução bem-sucedida, sem saída explícita (pode ser atribuição de variável)."

        except Exception as e:
            return f"ERRO DE EXECUÇÃO: {type(e).__name__}: {str(e)}"
            
        finally:
            sys.stdout = old_stdout


    def ask_chatgpt(self, question: str, context: str = "", memory_top_k: int = 5) -> str:

        if not isinstance(question, str):
            question = str(question)

        # 1) Verificar se há CSV carregado
        if self.df is None:
            return "Por favor, carregue um arquivo CSV antes de fazer perguntas."

        # 2) Usar LLM para verificar a relevância da pergunta
        if not self.is_question_relevant(question):
            return "Desculpe, só posso responder a perguntas relacionadas ao arquivo CSV carregado..."

        # 3) Preparar contexto do DataFrame
        try:
            df_info = f"""
            O DataFrame tem {len(self.df)} linhas e {len(self.df.columns)} colunas.
            
            O nome das colunas são: {', '.join(self.df.columns.tolist())}

            Estatísticas básicas:
            {self.df.describe(include='all').to_string()}
            """
            # Limitar tamanho
            df_info = df_info[:2000] + ("..." if len(df_info) > 2000 else "")
        except Exception:
            df_info = f"DataFrame com {len(self.df)} linhas e colunas: {', '.join(self.df.columns.tolist())}"

        # 4) Buscar histórico relevante (últimas interações)
        history_context = self.get_recent_history(memory_top_k)

        # 5) Construir prompt

        prompt = f"""Você é um especialista em análise de dados CSV. Responda à pergunta do usuário de forma clara e objetiva.

        Informações do DataFrame:
        "{df_info}"
        
        Histórico de perguntas e respostas:
        "{history_context}"
        
        Pergunta do usuário:
        "{question}"

        """

        # Gerar código python
        code_generation_prompt = f"""
            Você é um especialista em Python/Pandas. Sua única função é analisar dados.
            Use a variável 'df' (o DataFrame).

            Instrução: Gere APENAS o código Python (dentro de um bloco markdown 'python')
            que responde à pergunta do usuário. O resultado final deve ser atribuído a
            uma variável chamada 'result' ou impresso no console.

            Informações do DataFrame:
            "{df_info}"
            
            Histórico de perguntas e respostas:
            "{history_context}"
            
            Pergunta do usuário:
            "{question}"
        """

        code_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Gere APENAS o código Python/Pandas."},
                {"role": "user", "content": code_generation_prompt},
            ],
            temperature=0.0,
            max_tokens=800
        )

        generated_code = code_response.choices[0].message["content"]

        # Extrair código do bloco
        match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
        code_to_execute = match.group(1).strip() if match else generated_code.strip()

        # Executar código
        execution_result = self.execute_code(code_to_execute)
        is_error = execution_result.startswith("ERRO DE EXECUÇÃO")

        # Verificar se resposta python será usada
        if is_error == False:
            # Incluir resposta da execução do python
            print("Respondeu usando python")
            prompt = prompt + "A resposta deve ser baseada EXCLUSIVAMENTE no resultado da execução do código Python/Pandas: " + execution_result
        else:
            # Não incluir
            print("Respondeu usando linguística")

        print("Prompt final:\n\n" + prompt)


        # 6) Usar a LLM - Resposta final
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise de dados CSV."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Baixa temperatura para respostas consistentes
                max_tokens=800  # Limitar tamanho da resposta
            )
            answer = response.choices[0].message["content"].strip()
        except Exception as e:
            answer = f"Erro ao processar a pergunta: {str(e)}"

        # 7) Salvar na memória (apenas se a resposta for válida)
        if not answer.startswith("Erro ao processar"):
            self.add_memory(question, answer)

        return answer


    # ---------------------------
    # Análises básicas / gráficos
    # ---------------------------
    def basic_analysis(self):
        """Retorna tipos, nulos e estatísticas resumidas do DataFrame."""
        if self.df is None:
            return {"error": "DataFrame não carregado."}
        return {
            "tipos_de_dados": self.df.dtypes.apply(lambda dt: str(dt)).to_dict(),
            "valores_nulos": self.df.isnull().sum().to_dict(),
            "estatisticas": self.df.describe(include="all").to_dict(),
        }

    def generate_plot(self, plot_type: str = "histograma", column=None):
        """Gera gráficos e retorna um buffer PNG (io.BytesIO)."""
        if self.df is None:
            raise ValueError("DataFrame não carregado.")

        fig, ax = plt.subplots(figsize=(6, 4))

        if plot_type == "histograma" and column:
            sns.histplot(self.df[column].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histograma de {column}")

        elif plot_type == "dispersao" and column and len(column) == 2:
            sns.scatterplot(x=self.df[column[0]], y=self.df[column[1]], ax=ax)
            ax.set_title(f"Dispersão: {column[0]} x {column[1]}")

        elif plot_type == "correlacao":
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, ax=ax)
            ax.set_title("Matriz de Correlação")

        else:
            ax.text(0.5, 0.5, "Tipo de gráfico não reconhecido", ha="center", va="center")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    # Adicionar este método à classe CSVAIAgent no agent.py
    def generate_plot_from_text(self, user_input: str):
        """
        Tenta gerar um gráfico baseado no texto do usuário.
        Retorna (resposta_texto, buffer_imagem)
        """
        if self.df is None:
            return "Por favor, carregue um arquivo CSV primeiro.", None

        user_input_lower = user_input.lower()

        try:
            # Detectar tipo de gráfico
            if "histograma" in user_input_lower:
                # Encontrar coluna mencionada
                for col in self.df.columns:
                    if col.lower() in user_input_lower:
                        buf = self.generate_plot("histograma", col)
                        return f"Gerando histograma para a coluna **{col}**", buf

            elif "dispersão" in user_input_lower or "dispersao" in user_input_lower:
                # Tentar encontrar duas colunas mencionadas
                mentioned_cols = []
                for col in self.df.columns:
                    if col.lower() in user_input_lower:
                        mentioned_cols.append(col)
                        if len(mentioned_cols) == 2:
                            buf = self.generate_plot("dispersao", mentioned_cols)
                            return f"Gerando gráfico de dispersão: **{mentioned_cols[0]} x {mentioned_cols[1]}**", buf

                if len(mentioned_cols) < 2:
                    return "Por favor, especifique duas colunas para o gráfico de dispersão.", None

            elif "correlação" in user_input_lower or "correlacao" in user_input_lower:
                buf = self.generate_plot("correlacao")
                return "Gerando matriz de correlação", buf

            return "Comando de gráfico não reconhecido. Tente especificar o tipo e as colunas.", None

        except Exception as e:
            return f"Erro ao gerar gráfico: {str(e)}", None
