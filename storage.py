import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("trading_history.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT,
            capital INTEGER,
            risk TEXT,
            horizon TEXT,
            verdict TEXT,
            summary TEXT,
            shares INTEGER,
            amount INTEGER
        )
        """)

def save_decision(symbol, capital, risk, horizon, result):
    plan = result.get("plan", {})
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        INSERT INTO decisions(symbol, capital, risk, horizon, verdict, summary, shares, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, int(capital), risk, horizon, result.get("verdict", ""),
              result.get("summary", ""), int(plan.get("shares", 0)), int(plan.get("amount", 0))))

def load_history(limit=10):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT created_at AS 日時, symbol AS 銘柄, verdict AS 判断, shares AS 仮想株数, amount AS 仮想金額 FROM decisions ORDER BY id DESC LIMIT ?",
            conn, params=(int(limit),)
        )
