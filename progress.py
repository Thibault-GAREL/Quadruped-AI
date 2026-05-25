"""Affiche le progres du dernier run MLflow neuro-GA.

Usage:
    python progress.py
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB = Path(__file__).resolve().parent / "mlflow.db"

if not DB.exists():
    print(f"DB introuvable : {DB}")
    raise SystemExit(1)

conn = sqlite3.connect(str(DB))
cur = conn.cursor()

row = cur.execute(
    "SELECT run_uuid, name, start_time, end_time, status "
    "FROM runs ORDER BY start_time DESC LIMIT 1"
).fetchone()

if not row:
    print("Aucun run en base.")
    raise SystemExit()

run_uuid, name, start_ms, end_ms, status = row
start_dt = datetime.fromtimestamp(start_ms / 1000)

if end_ms and status == "FINISHED":
    end_dt = datetime.fromtimestamp(end_ms / 1000)
    duration = end_dt - start_dt
    live = False
else:
    end_dt = None
    duration = datetime.now() - start_dt
    live = True

step, best = cur.execute(
    "SELECT MAX(step), MAX(value) FROM metrics "
    "WHERE run_uuid=? AND key='fitness_best'",
    (run_uuid,),
).fetchone()
n_steps = cur.execute(
    "SELECT COUNT(DISTINCT step) FROM metrics WHERE run_uuid=?",
    (run_uuid,),
).fetchone()[0]

print(f"Run             : {name}")
print(f"Status          : {'EN COURS' if live else status}")
print(f"Demarrage       : {start_dt:%Y-%m-%d %H:%M:%S}")
print(f"Duree           : {duration}")
print(f"Generations     : {n_steps}")
print(f"Step max        : {step}")
if best is not None:
    print(f"Best fitness    : {best:.2f}")
else:
    print("Best fitness    : (pas encore loggue)")

if live and n_steps:
    rate = n_steps / max(duration.total_seconds() / 60, 1)
    print(f"Cadence         : {rate:.1f} gens/min")
