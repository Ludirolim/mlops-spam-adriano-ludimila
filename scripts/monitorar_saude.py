"""Script de verificação de saúde, alertas operacionais e reinício controlado (Etapa 4 - Aulas 06 e 07).

Uso:
    python scripts/monitorar_saude.py              # monitora continuamente a cada 10s
    python scripts/monitorar_saude.py --once       # executa uma verificação única e encerra
"""
import argparse
import json
import subprocess
import time
import urllib.request

LIMITE_FALHAS_SEGUIDAS = 3
LIMITE_ERROS_5XX = 5
LIMITE_LATENCIA_MS = 500.0


def consultar_json(url: str, timeout: int = 3) -> tuple[bool, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                dados = json.loads(resp.read().decode("utf-8"))
                return True, dados
            return False, {}
    except Exception:
        return False, {}


def verificar_ciclo(base_url: str, falhas_atuais: int, auto_restart: bool) -> int:
    horario = time.strftime("%H:%M:%S")
    ok_health, dados_health = consultar_json(f"{base_url}/health")

    if not ok_health:
        falhas_atuais += 1
        print(f"[{horario}] ALERTA: serviço fora do ar (/health falhou {falhas_atuais}/{LIMITE_FALHAS_SEGUIDAS})")
        if falhas_atuais >= LIMITE_FALHAS_SEGUIDAS:
            print(f"[{horario}] AÇÃO AUTOMÁTICA: limite de {LIMITE_FALHAS_SEGUIDAS} falhas atingido. Acionando reinício controlado...")
            if auto_restart:
                subprocess.run(
                    ["kubectl", "rollout", "restart", "deployment/spam-api-deployment"],
                    check=False,
                )
            falhas_atuais = 0
        return falhas_atuais

    falhas_atuais = 0
    versao = dados_health.get("versao", "--")
    ambiente = dados_health.get("ambiente", "--")

    ok_metrics, metricas = consultar_json(f"{base_url}/metrics")
    if ok_metrics:
        req = metricas.get("requisicoes", 0)
        erros_5xx = metricas.get("erros_5xx", 0)
        latencia = float(metricas.get("latencia_media_ms", 0.0))

        print(
            f"[{horario}] OK | status=saudavel versao={versao} amb={ambiente} "
            f"requisicoes={req} erros_5xx={erros_5xx} latencia_media_ms={latencia}"
        )

        if erros_5xx > LIMITE_ERROS_5XX:
            print(f"[{horario}] ALERTA: muitos erros 5xx detectados ({erros_5xx} > {LIMITE_ERROS_5XX})")
        if latencia > LIMITE_LATENCIA_MS:
            print(f"[{horario}] ALERTA: latência média alta ({latencia} ms > {LIMITE_LATENCIA_MS} ms)")
    else:
        print(f"[{horario}] OK | /health respondeu 200 (versao={versao}, amb={ambiente})")

    return falhas_atuais


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitora saúde e métricas da API de Spam.")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base do serviço")
    parser.add_argument("--intervalo", type=int, default=10, help="Intervalo em segundos")
    parser.add_argument("--once", action="store_true", help="Executa apenas uma verificação")
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Desativa a execução automática do kubectl rollout restart",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    falhas = 0

    while True:
        falhas = verificar_ciclo(base_url, falhas, auto_restart=not args.no_restart)
        if args.once:
            break
        time.sleep(args.intervalo)


if __name__ == "__main__":
    main()
