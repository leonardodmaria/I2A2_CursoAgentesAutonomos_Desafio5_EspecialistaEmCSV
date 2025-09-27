# agent.py
import os
import io
import csv
import re
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

    def get_recent_history(self, n: int = 5) -> str:
        recent = self.memory[-n:]
        return "\n".join([f"Q: {m['question']}\nA: {m['answer']}" for m in recent])
        
    """

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

    # ---------------------------
    # Integração com ChatGPT (memória + prompt)
    # ---------------------------
    def ask_chatgpt(self, question: str, context: str = "", memory_top_k: int = 5) -> str:
        """
        Responde perguntas sobre o arquivo CSV com memória do histórico.
        Versão simplificada e eficiente.
        """
        if not isinstance(question, str):
            question = str(question)

        # 1) Verificar se há CSV carregado
        if self.df is None:
            return "Por favor, carregue um arquivo CSV antes de fazer perguntas."

        # 2) Verificar se a pergunta é relacionada ao CSV
        q_lower = question.lower().strip()

        # Palavras-chave básicas
        keywords = [
            "coluna", "linha",
            "media", "média", "mediana", "minimo", "mínimo", "maximo", "máximo",
            "estatistica", "estatística",
            "nulo", "valor",
            "distribuicao", "distribuição",
            "histograma", "grafico", "gráfico",
            "correlacao", "correlação",
            "cluster", "pca", "variancia", "variância",
            "padrao", "padrão",
            "outlier", "arquivo", "csv", "planilha", "dataset",
            "dado", "dataframe",
            "informacao", "informação",
            "conteudo", "conteúdo", "resumo", "amostra",
            "intervalo", "medida",
            "variavel", "variável",
            "atipico", "atípico", "frequente", "tendencia", "tendência",
            "conclusão", "conclusao", "historico", "histórico",
            "pergunta", "resposta", "palavra"
        ]

        # Verificar palavras-chave ou nomes de colunas
        related = any(kw in q_lower for kw in keywords)
        if not related:
            for col in self.df.columns:
                if str(col).lower() in q_lower:
                    related = True
                    break

        if not related:
            return "Pergunta inválida. Por favor, faça uma pergunta relacionada ao arquivo CSV carregado."

        # 3) Preparar contexto do DataFrame
        try:
            df_info = f"""
            DataFrame com {len(self.df)} linhas e {len(self.df.columns)} colunas.
            Colunas: {', '.join(self.df.columns.tolist())}

            Estatísticas básicas:
            {self.df.describe(include='all').to_string()}
            """
            # Limitar tamanho
            df_info = df_info[:2000] + ("..." if len(df_info) > 2000 else "")
        except Exception:
            df_info = f"DataFrame com {len(self.df)} linhas e colunas: {', '.join(self.df.columns.tolist())}"

        # 4) Buscar histórico relevante (últimas interações)
        if self.memory:
            # Usar as últimas N interações como contexto
            recent_memory = self.memory[-memory_top_k:]
            memory_text = "\n".join([
                f"P: {m['question']}\nR: {m['answer'][:300]}..." if len(
                    m['answer']) > 300 else f"P: {m['question']}\nR: {m['answer']}"
                for m in recent_memory
            ])
            history_context = f"\n\nHistórico recente:\n{memory_text}"
        else:
            history_context = ""

        # 5) Construir prompt eficiente
        prompt = f"""Você é um assistente especializado em análise de dados CSV. Responda em português de forma clara.

    Informações do DataFrame:
    {df_info}
    {history_context}

    Pergunta atual: {question}

    Responda de forma direta e útil, baseando-se nos dados disponíveis."""

        # 6) Chamada à API
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
