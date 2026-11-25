"""Testes automatizados da API de classificação de spam (Etapa 3 e 4 - CI/CD e Observabilidade)."""
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

# Garante uma chave de API conhecida durante a execução dos testes automatizados
main_module.API_KEY = "chave-ficticia-123"

client = TestClient(app)
HEADERS = {"X-API-Key": "chave-ficticia-123"}


def test_health_retorna_200_e_formato_esperado():
    """Verifica se GET /health retorna HTTP 200 e os campos do contrato."""
    resposta = client.get("/health")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status"] == "ok"
    assert "versao" in dados
    assert "ambiente" in dados


def test_predict_classifica_mensagem_spam():
    """Verifica se POST /predict classifica corretamente uma mensagem típica de spam."""
    payload = {
        "texto": "PARABÉNS! Você ganhou um prêmio de R$ 5.000. Clique no link para resgatar agora"
    }
    resposta = client.post("/predict", json=payload, headers=HEADERS)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["rotulo"] == "spam"
    assert isinstance(dados["probabilidade"], float)
    assert 0.0 <= dados["probabilidade"] <= 1.0
    assert "versao" in dados


def test_predict_classifica_mensagem_comum_nao_spam():
    """Verifica se POST /predict classifica corretamente uma mensagem comum como nao_spam."""
    payload = {"texto": "A reunião de alinhamento de amanhã foi adiada para quinta-feira"}
    resposta = client.post("/predict", json=payload, headers=HEADERS)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["rotulo"] == "nao_spam"
    assert isinstance(dados["probabilidade"], float)
    assert 0.0 <= dados["probabilidade"] <= 1.0
    assert "versao" in dados


def test_predict_rejeita_requisicao_sem_api_key_com_401():
    """Verifica se POST /predict retorna HTTP 401 quando a chave X-API-Key está ausente ou incorreta."""
    payload = {"texto": "Mensagem de teste para validar autenticação"}
    resposta_sem_chave = client.post("/predict", json=payload)
    assert resposta_sem_chave.status_code == 401

    resposta_chave_errada = client.post(
        "/predict", json=payload, headers={"X-API-Key": "chave-incorreta"}
    )
    assert resposta_chave_errada.status_code == 401


def test_predict_rejeita_texto_vazio_com_erro_4xx():
    """Verifica se POST /predict rejeita texto vazio ou apenas espaços com erro 4xx (422)."""
    resposta_vazia = client.post("/predict", json={"texto": ""}, headers=HEADERS)
    assert 400 <= resposta_vazia.status_code < 500

    resposta_espacos = client.post("/predict", json={"texto": "   "}, headers=HEADERS)
    assert 400 <= resposta_espacos.status_code < 500


def test_metrics_retorna_200_e_contabiliza_operacao():
    """Verifica se GET /metrics retorna as métricas operacionais exigidas pelo painel."""
    resposta = client.get("/metrics")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert "requisicoes" in dados
    assert "erros_5xx" in dados
    assert "latencia_media_ms" in dados
    assert dados["requisicoes"] >= 1
    assert dados["erros_5xx"] == 0
