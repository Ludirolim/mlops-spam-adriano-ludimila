"""API de classificação de spam.

O contrato da API está no README.md.
Para rodar:  uvicorn app.main:app --reload
"""
import json
import logging
import os
import time
import uuid
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
ENV_PATH = RAIZ / ".env"
STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("spam_api")


def log_evento(nivel: str, mensagem: str, **campos) -> None:
    """Emite log estruturado em JSON na saída padrão (conforme Aula 06)."""
    registro = {
        "hora": time.strftime("%H:%M:%S"),
        "nivel": nivel,
        "mensagem": mensagem,
        **campos,
    }
    logger.info(json.dumps(registro, ensure_ascii=False))


def _carregar_dotenv(caminho: Path) -> None:
    """Carrega variáveis do arquivo .env local caso ainda não estejam definidas no ambiente."""
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip())


_carregar_dotenv(ENV_PATH)

# --- Configuração (vem de variáveis de ambiente, nunca escrita no código) ---
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
APP_ENV = os.getenv("APP_ENV", "dev")
API_KEY = os.getenv("API_KEY", "")
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(RAIZ / "model" / "modelo.joblib"))
)

# --- Modelo: carregado uma vez, quando a API inicia ---
modelo = joblib.load(MODEL_PATH)

# --- Métricas operacionais em memória (Etapa 4 - Observabilidade) ---
metricas = {
    "requisicoes": 0,
    "erros_4xx": 0,
    "erros_5xx": 0,
    "soma_latencia_ms": 0.0,
    "spam": 0,
    "nao_spam": 0,
}

app = FastAPI(title="API de classificação de spam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def middleware_observabilidade(request: Request, call_next):
    """Mede a latência, contabiliza requisições/erros e registra logs estruturados em JSON."""
    inicio = time.perf_counter()
    id_req = str(uuid.uuid4())[:8]
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
        if request.url.path == "/predict" or status_code >= 500:
            metricas["requisicoes"] += 1
            metricas["soma_latencia_ms"] += latencia_ms
            nivel = "INFO"
            if 400 <= status_code < 500:
                metricas["erros_4xx"] += 1
                nivel = "WARNING"
            elif status_code >= 500:
                metricas["erros_5xx"] += 1
                nivel = "ERROR"
            log_evento(
                nivel,
                "requisicao processada",
                id=id_req,
                metodo=request.method,
                rota=request.url.path,
                status=status_code,
                ms=latencia_ms,
                versao=MODEL_VERSION,
                ambiente=APP_ENV,
            )


class Mensagem(BaseModel):
    texto: str = Field(min_length=1, max_length=2000)


@app.get("/", include_in_schema=False)
def pagina_web():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "versao": MODEL_VERSION,
        "ambiente": APP_ENV,
    }


@app.get("/metrics")
def obter_metricas():
    total = metricas["requisicoes"]
    latencia_media = round(metricas["soma_latencia_ms"] / total, 2) if total > 0 else 0.0
    return {
        "requisicoes": total,
        "erros_4xx": metricas["erros_4xx"],
        "erros_5xx": metricas["erros_5xx"],
        "latencia_media_ms": latencia_media,
        "spam": metricas["spam"],
        "nao_spam": metricas["nao_spam"],
        "versao": MODEL_VERSION,
        "ambiente": APP_ENV,
    }


@app.post("/predict")
def predict(
    msg: Mensagem,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida ou ausente")

    texto_limpo = msg.texto.strip()
    if not texto_limpo:
        raise HTTPException(status_code=422, detail="O texto da mensagem não pode ser vazio")

    probabilidades = modelo.predict_proba([texto_limpo])[0]
    indice = int(probabilidades.argmax())
    rotulo = str(modelo.classes_[indice])
    probabilidade = round(float(probabilidades[indice]), 2)

    if rotulo in ("spam", "nao_spam"):
        metricas[rotulo] += 1

    return {
        "rotulo": rotulo,
        "probabilidade": probabilidade,
        "versao": MODEL_VERSION,
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
