"""API de classificação de spam.

O contrato da API está no README.md.
Para rodar:  uvicorn app.main:app --reload
"""
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
ENV_PATH = RAIZ / ".env"
STATIC_DIR = Path(__file__).resolve().parent / "static"


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

app = FastAPI(title="API de classificação de spam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    return {
        "rotulo": rotulo,
        "probabilidade": probabilidade,
        "versao": MODEL_VERSION,
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
