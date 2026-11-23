"""Application locale : tableau de bord, mise à jour hebdomadaire, résultats officiels et vérifications.

Lancement : double-cliquer sur `lancer_application.bat`, ou `python app/server.py` depuis la racine du projet.
Le serveur n'écoute que sur 127.0.0.1 (votre machine) et n'exécute qu'une liste fermée d'actions. Chaque action
tourne dans un processus Python séparé, dont la sortie est affichée en direct dans l'application.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
PY = sys.executable

# Actions disponibles : nom -> étapes (libellé, commande)
JOBS: dict[str, list[tuple[str, list[str]]]] = {
    "update": [
        ("Télécharger les prix jusqu'à la dernière semaine terminée", [PY, "scripts/01_download_data.py", "--update"]),
        ("Faire décider l'agent et calculer les résultats", [PY, "dashboard/build_dashboard.py"]),
    ],
    "rebuild": [("Recalculer le tableau de bord avec les données actuelles", [PY, "dashboard/build_dashboard.py"])],
    "tests": [("Lancer les tests automatiques", [PY, "-m", "pytest", "-p", "no:cacheprovider"])],
}

FIGURE_CAPTIONS = {
    "fig_test_valeur_drawdown": "Valeur cumulée et drawdown au test (2022-2026)",
    "fig_test_ic_sharpe": "Sharpe au test et intervalles de confiance à 95 %",
    "fig_test_rendement_risque": "Rendement et volatilité au test",
    "fig_ecart_etf": "D'où vient l'écart avec l'ETF",
    "fig_regularite": "Régularité : Sharpe par période et dans la pire période",
    "fig_validation_blocs": "Sharpe par bloc de validation (2014-2021)",
    "fig_reglage": "Réglage des hyperparamètres : agent entraîné et référence",
    "fig_poids_agent": "Poids de l'agent au test",
    "fig_turnover": "Rotation hebdomadaire du portefeuille",
    "fig_entrainement": "Dynamique d'entraînement des 5 modèles finaux",
    "fig_univers": "Titres investissables et découpage temporel",
    "fig_versions": "Alternatives étudiées",
}


# --------------------------------------------------------------------------- exécution des actions
class JobRunner:
    """Exécute une action à la fois, étape par étape, en gardant la sortie en mémoire."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job: dict | None = None

    def start(self, name: str) -> tuple[bool, str]:
        with self.lock:
            if self.job and self.job["state"] == "running":
                return False, "Une action est déjà en cours : attendez qu'elle se termine."
            self.job = {"name": name, "state": "running", "step": 0, "steps": [s for s, _ in JOBS[name]],
                        "log": [], "started": datetime.now().strftime("%H:%M:%S"), "ended": None}
        threading.Thread(target=self._run, args=(name,), daemon=True).start()
        return True, "démarré"

    def _log(self, line: str) -> None:
        with self.lock:
            self.job["log"].append(line.rstrip("\n"))
            if len(self.job["log"]) > 4000:
                self.job["log"] = self.job["log"][-4000:]

    def _run(self, name: str) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        for i, (label, cmd) in enumerate(JOBS[name]):
            with self.lock:
                self.job["step"] = i
            self._log(f"▶ {label}")
            try:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", env=env,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                for line in proc.stdout:
                    self._log(line)
                code = proc.wait()
            except OSError as exc:
                self._log(f"Impossible de lancer la commande : {exc}")
                code = -1
            if code != 0:
                self._log(f"✖ Échec de l'étape « {label} » (code {code}).")
                with self.lock:
                    self.job.update(state="failed", ended=datetime.now().strftime("%H:%M:%S"))
                return
            self._log(f"✔ {label} : terminé.")
        with self.lock:
            self.job.update(step=len(JOBS[name]), state="succeeded", ended=datetime.now().strftime("%H:%M:%S"))

    def snapshot(self, offset: int) -> dict:
        with self.lock:
            if not self.job:
                return {"state": "idle"}
            j = dict(self.job)
            j["log"] = self.job["log"][offset:]
            j["next_offset"] = len(self.job["log"])
            return j


RUNNER = JobRunner()


# --------------------------------------------------------------------------- données affichées
def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def last_csv_date(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4096))
        tail = f.read().decode("utf-8", errors="replace").strip().splitlines()
    return tail[-1].split(",")[0][:10] if tail else None


def status() -> dict:
    data = read_json(ROOT / "dashboard" / "data.json") or {}
    weeks = data.get("weeks", [])
    done = [w for w in weeks if w.get("ret") is not None]
    last = None
    if done:
        i = len(done) - 1
        etf = data["bench"]["ETF Euro Stoxx 50"][i]
        last = {"date": done[-1]["date"], "end": done[-1]["end"], "agent": done[-1]["ret"], "etf": etf}
    return {
        "prices_until": last_csv_date(ROOT / "data" / "processed" / "prices.csv"),
        "decision_date": weeks[-1]["date"] if weeks else None,
        "dashboard_generated": data.get("generated"),
        "n_weeks": len(weeks),
        "last_completed_week": last,
        "python": sys.version.split()[0],
    }


def read_table(name: str) -> list[dict]:
    path = ROOT / "results" / "tables" / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["strategy"] = r.pop("") if "" in r else r.get("strategy")
    return rows


def results() -> dict:
    figs = sorted(p.stem for p in (ROOT / "report" / "figures").glob("*.png"))
    order = list(FIGURE_CAPTIONS)
    figs.sort(key=lambda s: order.index(s) if s in order else 99)
    return {"test": read_table("test_comparison.csv"), "tuning": read_table("tuning_summary.csv"),
            "best_config": read_json(ROOT / "results" / "tables" / "best_config.json"),
            "lock": (ROOT / "results" / ".final_test_done").read_text(encoding="utf-8") if (ROOT / "results" / ".final_test_done").exists() else None,
            "figures": [{"name": f, "caption": FIGURE_CAPTIONS.get(f, f)} for f in figs]}


def dashboard_html() -> bytes:
    data = (ROOT / "dashboard" / "data.json").read_text(encoding="utf-8")
    template = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    return template.replace("/*__DATA__*/null", data).encode("utf-8")


# --------------------------------------------------------------------------- serveur HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # pas de journal HTTP dans la console
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: object, code: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", code)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                self._send((APP / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/dashboard":
                self._send(dashboard_html(), "text/html; charset=utf-8")
            elif url.path == "/api/status":
                self._json(status())
            elif url.path == "/api/results":
                self._json(results())
            elif url.path == "/api/job":
                offset = int(parse_qs(url.query).get("offset", ["0"])[0])
                self._json(RUNNER.snapshot(offset))
            elif url.path.startswith("/figures/"):
                name = Path(url.path).name
                path = ROOT / "report" / "figures" / name
                if path.suffix != ".png" or not path.is_file():
                    return self._send(b"introuvable", "text/plain; charset=utf-8", 404)
                self._send(path.read_bytes(), "image/png")
            else:
                self._send(b"introuvable", "text/plain; charset=utf-8", 404)
        except FileNotFoundError as exc:
            self._json({"error": f"fichier manquant : {exc.filename}"}, 500)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith("/api/run/"):
            name = url.path.rsplit("/", 1)[-1]
            if name not in JOBS:
                return self._json({"ok": False, "message": "action inconnue"}, 404)
            ok, msg = RUNNER.start(name)
            return self._json({"ok": ok, "message": msg}, 200 if ok else HTTPStatus.CONFLICT)
        self._json({"ok": False, "message": "introuvable"}, 404)


def main() -> None:
    port = int(os.environ.get("APP_PORT", "8765"))
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("Aucun port libre entre 8765 et 8784.")
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Application lancée : {url}")
    print("Laissez cette fenêtre ouverte pendant l'utilisation ; fermez-la pour arrêter l'application.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
