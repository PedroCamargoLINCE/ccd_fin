from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT
TAXAS = RAW / "Taxas"

CACHE = Path("C:/temp/ccd_cache")
PROCESSED = CACHE / "processed"
CHECKPOINTS = CACHE / "checkpoints"
LIGHTNING_LOGS = CACHE / "lightning_logs"

for p in (PROCESSED, CHECKPOINTS, LIGHTNING_LOGS):
    p.mkdir(parents=True, exist_ok=True)

DISEASES = ["hanseniase", "hepatite", "hiv_aids", "sifilis", "tuberculose"]

MUNICIPIOS_ALVO = {
    "3505708": "Barueri",
    "3506003": "Bauru",
    "3509502": "Campinas",
    "3510609": "Carapicuíba",
    "3513801": "Diadema",
    "3518701": "Guarujá",
    "3518800": "Guarulhos",
    "3522208": "Itapevi",
    "3525904": "Jundiaí",
    "3529401": "Mauá",
    "3534401": "Osasco",
    "3536505": "Paulínia",
    "3541000": "Praia Grande",
    "3543402": "Ribeirão Preto",
    "3547809": "Santo André",
    "3548500": "Santos",
    "3548708": "São Bernardo do Campo",
    "3549805": "São José do Rio Preto",
    "3549904": "São José dos Campos",
    "3550308": "São Paulo",
    "3551009": "São Vicente",
    "3552205": "Sorocaba",
    "3554102": "Taboão da Serra",
}
