import os
import io
import json
import uuid
import time
import sqlite3
import tempfile
import hashlib
import math
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


APP_TITLE = "SME Cash-Flow Forecasting and Liquidity Intelligence"
SCHEMA_VERSION = "1.1.1"
DB_PATH = Path(os.getenv("LIQUIDITY_DB", "liquidity_demo.sqlite"))
UPLOAD_DIR = Path(os.getenv("LIQUIDITY_UPLOAD_DIR", "data/uploads"))
OUTPUT_DIR = Path(os.getenv("LIQUIDITY_OUTPUT_DIR", "outputs"))
ALLOWED_CURRENCIES = {"USD", "EUR", "GBP"}

DEFAULT_SEED = 20260611
MAX_SEED = 2_147_483_647

ROLES = ["administrator", "analyst", "reviewer", "read_only"]

PERMISSIONS = {
    "administrator": {
        "view",
        "export",
        "import",
        "delete",
        "configure",
        "create_forecast",
        "validate",
        "approve",
        "override",
        "archive",
        "simulate",
        "run_tests",
        "inject_shock",
        "save_simulation",
    },
    "analyst": {
        "view",
        "export",
        "import",
        "configure",
        "create_forecast",
        "validate",
        "simulate",
        "run_tests",
        "inject_shock",
        "save_simulation",
    },
    "reviewer": {
        "view",
        "export",
        "validate",
        "approve",
        "override",
        "archive",
        "simulate",
        "run_tests",
    },
    "read_only": {
        "view",
        "export",
    },
}

VALID_TRANSITIONS = {
    "draft": {"validated", "rejected"},
    "validated": {"approved", "rejected", "draft"},
    "approved": {"archived", "corrected"},
    "rejected": {"draft"},
    "corrected": {"validated"},
    "archived": set(),
}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT,
    currency TEXT,
    timezone TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    schema_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    org_id TEXT,
    username TEXT NOT NULL,
    role TEXT NOT NULL,
    password_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    FOREIGN KEY(org_id) REFERENCES organizations(id)
);

CREATE TABLE IF NOT EXISTS source_systems (
    id TEXT PRIMARY KEY,
    org_id TEXT,
    name TEXT NOT NULL,
    system_type TEXT,
    created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    FOREIGN KEY(org_id) REFERENCES organizations(id)
);

CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    name TEXT NOT NULL,
    currency TEXT NOT NULL,
    account_type TEXT,
    opening_balance REAL NOT NULL,
    as_of_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    FOREIGN KEY(org_id) REFERENCES organizations(id)
);

CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    account_id TEXT,
    txn_date TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    external_ref TEXT,
    source_system_id TEXT,
    provenance TEXT,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(org_id) REFERENCES organizations(id),
    FOREIGN KEY(account_id) REFERENCES accounts(id),
    FOREIGN KEY(source_system_id) REFERENCES source_systems(id)
);

CREATE TABLE IF NOT EXISTS invoices (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    customer TEXT,
    issue_date TEXT,
    due_date TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    collection_probability REAL DEFAULT 1.0,
    delay_days INTEGER DEFAULT 0,
    source_system_id TEXT,
    provenance TEXT,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(org_id, invoice_number)
);

CREATE TABLE IF NOT EXISTS payroll_runs (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    payroll_date TEXT NOT NULL,
    gross_amount REAL NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    delay_days INTEGER DEFAULT 0,
    source_system_id TEXT,
    provenance TEXT,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS supplier_obligations (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    supplier TEXT,
    invoice_reference TEXT,
    due_date TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    delay_days INTEGER DEFAULT 0,
    terms TEXT,
    status TEXT NOT NULL,
    source_system_id TEXT,
    provenance TEXT,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS financing_facilities (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    lender TEXT,
    facility_type TEXT,
    limit_amount REAL NOT NULL,
    available_amount REAL NOT NULL,
    outstanding_amount REAL NOT NULL DEFAULT 0,
    interest_rate REAL NOT NULL,
    repayment_date TEXT,
    repayment_amount REAL DEFAULT 0,
    status TEXT NOT NULL,
    provenance TEXT,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecast_runs (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    as_of_date TEXT NOT NULL,
    opening_balance REAL NOT NULL,
    scenario_json TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    approved_by TEXT,
    approved_at TEXT,
    override_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS forecast_lines (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    day_index INTEGER NOT NULL,
    date TEXT NOT NULL,
    opening_balance REAL NOT NULL,
    receipts REAL NOT NULL,
    payroll REAL NOT NULL,
    supplier_payments REAL NOT NULL,
    financing_in REAL NOT NULL,
    financing_out REAL NOT NULL,
    net_change REAL NOT NULL,
    closing_balance REAL NOT NULL,
    minimum_cash REAL NOT NULL,
    financing_need REAL NOT NULL,
    confidence_lower REAL NOT NULL,
    confidence_upper REAL NOT NULL,
    explanation_json TEXT NOT NULL,
    data_quality_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES forecast_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    seed INTEGER NOT NULL,
    as_of_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    opening_balance REAL NOT NULL,
    ending_balance REAL NOT NULL,
    minimum_balance REAL NOT NULL,
    days_below_minimum INTEGER NOT NULL,
    total_financing_draws REAL NOT NULL,
    scenario_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS simulation_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    day_index INTEGER NOT NULL,
    date TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT,
    amount REAL NOT NULL,
    balance_after REAL NOT NULL,
    severity TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    actor_role TEXT NOT NULL,
    actor_user TEXT,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    before_json TEXT,
    after_json TEXT,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_batches (
    id TEXT PRIMARY KEY,
    org_id TEXT,
    filename TEXT NOT NULL,
    file_checksum TEXT NOT NULL,
    import_type TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    file_path TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    schema_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_errors (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    row_number INTEGER,
    field TEXT,
    issue TEXT,
    value TEXT,
    severity TEXT,
    raw_row_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(batch_id) REFERENCES import_batches(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_data (
    key TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class Scenario:
    horizon_days: int = 90
    receipt_delay_days: int = 3
    receipt_confidence: float = 0.92
    payroll_delay_days: int = 0
    supplier_delay_days: int = 5
    minimum_cash: float = 50000.0
    financing_buffer: float = 10000.0
    financing_interest_rate: float = 0.085
    seasonality_factor: float = 1.0

    def validate(self) -> None:
        if not isinstance(self.horizon_days, int) or self.horizon_days < 1 or self.horizon_days > 365:
            raise ValueError("horizon_days must be an integer between 1 and 365.")
        if self.receipt_delay_days < 0 or self.receipt_delay_days > 90:
            raise ValueError("receipt_delay_days must be between 0 and 90.")
        if self.payroll_delay_days < 0 or self.payroll_delay_days > 30:
            raise ValueError("payroll_delay_days must be between 0 and 30.")
        if self.supplier_delay_days < 0 or self.supplier_delay_days > 90:
            raise ValueError("supplier_delay_days must be between 0 and 90.")
        if not 0.0 <= self.receipt_confidence <= 1.0:
            raise ValueError("receipt_confidence must be between 0.0 and 1.0.")
        if self.minimum_cash < 0:
            raise ValueError("minimum_cash cannot be negative.")
        if self.financing_buffer < 0:
            raise ValueError("financing_buffer cannot be negative.")
        if not 0.0 <= self.financing_interest_rate <= 1.0:
            raise ValueError("financing_interest_rate must be between 0.0 and 1.0.")
        if not 0.5 <= self.seasonality_factor <= 1.5:
            raise ValueError("seasonality_factor must be between 0.5 and 1.5.")

    def signature(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
        ("schema_version", SCHEMA_VERSION, utc_now()),
    )
    conn.commit()


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def can(action: str) -> bool:
    role = st.session_state.get("role", "read_only")
    return action in PERMISSIONS.get(role, set())


def add_missing_columns(df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, val in mapping.items():
        if col not in out.columns:
            out[col] = val
    return out


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_text(value: Any, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def normalize_seed(value: Any) -> int:
    try:
        seed = int(value)
    except Exception:
        seed = DEFAULT_SEED
    return max(0, min(seed, MAX_SEED))


def facility_metrics(facilities: pd.DataFrame, fallback_rate: float = 0.0) -> Tuple[float, float, float]:
    if facilities.empty:
        return 0.0, 0.0, float(fallback_rate) / 365.0

    fac = add_missing_columns(
        facilities,
        {
            "available_amount": 0.0,
            "limit_amount": 0.0,
            "outstanding_amount": 0.0,
            "interest_rate": float(fallback_rate),
        },
    )

    available = float(
        np.nansum(pd.to_numeric(fac["available_amount"], errors="coerce").fillna(0.0))
    )
    outstanding = float(
        np.nansum(pd.to_numeric(fac["outstanding_amount"], errors="coerce").fillna(0.0))
    )

    rates = pd.to_numeric(fac["interest_rate"], errors="coerce")
    limits = pd.to_numeric(fac["limit_amount"], errors="coerce")

    valid = rates.notna() & limits.notna() & (limits > 0)
    if bool(valid.any()):
        weighted = float(
            np.nansum(rates[valid] * limits[valid]) / np.nansum(limits[valid])
        )
    else:
        finite_rates = rates[np.isfinite(rates)]
        weighted = float(finite_rates.mean()) if not finite_rates.empty else float(fallback_rate)

    weighted = clamp_float(weighted, 0.0, 1.0)
    return available, outstanding, weighted / 365.0


def ensure_demo_data() -> None:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) AS c FROM organizations").fetchone()["c"]
    if count > 0:
        conn.close()
        return

    rng = np.random.default_rng(DEFAULT_SEED)
    now = utc_now()
    today = datetime.now(timezone.utc).date()

    org_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO organizations
        (id, name, country, currency, timezone, created_at, updated_at, schema_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            org_id,
            "Northwind Retail Ltd",
            "GB",
            "GBP",
            "Europe/London",
            now,
            now,
            SCHEMA_VERSION,
        ),
    )

    demo_users = [
        ("admin", "administrator"),
        ("analyst", "analyst"),
        ("reviewer", "reviewer"),
        ("viewer", "read_only"),
    ]
    for username, role in demo_users:
        conn.execute(
            """
            INSERT INTO users
            (id, org_id, username, role, password_hash, created_at, updated_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                org_id,
                username,
                role,
                None,
                now,
                now,
                SCHEMA_VERSION,
            ),
        )

    source_names = [
        ("Bank feed", "bank"),
        ("AR ledger", "ar"),
        ("AP ledger", "ap"),
        ("Payroll system", "payroll"),
        ("Treasury workbook", "treasury"),
        ("Manual upload", "upload"),
    ]
    sources: Dict[str, str] = {}
    for name, typ in source_names:
        sid = str(uuid.uuid4())
        sources[name] = sid
        conn.execute(
            """
            INSERT INTO source_systems
            (id, org_id, name, system_type, created_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sid, org_id, name, typ, now, SCHEMA_VERSION),
        )

    account_id = str(uuid.uuid4())
    opening_balance = 185000.0
    conn.execute(
        """
        INSERT INTO accounts
        (id, org_id, name, currency, account_type, opening_balance, as_of_date,
         created_at, updated_at, schema_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            account_id,
            org_id,
            "Main operating account",
            "GBP",
            "operating",
            opening_balance,
            today.isoformat(),
            now,
            now,
            SCHEMA_VERSION,
        ),
    )

    txn_rows = []
    start = today - timedelta(days=180)
    for i in range(181):
        d = start + timedelta(days=i)
        dow = d.weekday()

        sales = 4200 + 1800 * math.sin(i / 5.0) + rng.normal(0, 450)
        if dow >= 5:
            sales *= 0.45
        sales = max(0.0, round(float(sales), 2))

        txn_rows.append(
            (
                str(uuid.uuid4()),
                org_id,
                account_id,
                d.isoformat(),
                sales,
                "GBP",
                "sales_receipt",
                "Daily sales receipt",
                f"HIST-S-{i:04d}",
                sources["Bank feed"],
                json.dumps({"generator": "synthetic_demo", "seed": DEFAULT_SEED}),
                SCHEMA_VERSION,
                now,
                now,
            )
        )

        if i % 14 == 0:
            payroll_amt = -round(float(11500 + rng.normal(0, 350)), 2)
            txn_rows.append(
                (
                    str(uuid.uuid4()),
                    org_id,
                    account_id,
                    d.isoformat(),
                    payroll_amt,
                    "GBP",
                    "payroll",
                    "Biweekly payroll run",
                    f"HIST-P-{i:04d}",
                    sources["Payroll system"],
                    json.dumps({"generator": "synthetic_demo"}),
                    SCHEMA_VERSION,
                    now,
                    now,
                )
            )

        if i % 7 == 0:
            supplier_amt = -round(float(5200 + rng.normal(0, 600)), 2)
            txn_rows.append(
                (
                    str(uuid.uuid4()),
                    org_id,
                    account_id,
                    d.isoformat(),
                    supplier_amt,
                    "GBP",
                    "supplier",
                    "Supplier payment",
                    f"HIST-AP-{i:04d}",
                    sources["AP ledger"],
                    json.dumps({"generator": "synthetic_demo"}),
                    SCHEMA_VERSION,
                    now,
                    now,
                )
            )

        if d.day == 1:
            txn_rows.append(
                (
                    str(uuid.uuid4()),
                    org_id,
                    account_id,
                    d.isoformat(),
                    -4500.0,
                    "GBP",
                    "rent",
                    "Monthly rent",
                    f"HIST-RENT-{i:04d}",
                    sources["AP ledger"],
                    json.dumps({"generator": "synthetic_demo"}),
                    SCHEMA_VERSION,
                    now,
                    now,
                )
            )

        if i % 30 == 0:
            txn_rows.append(
                (
                    str(uuid.uuid4()),
                    org_id,
                    account_id,
                    d.isoformat(),
                    -260.0,
                    "GBP",
                    "interest",
                    "Facility interest",
                    f"HIST-INT-{i:04d}",
                    sources["Treasury workbook"],
                    json.dumps({"generator": "synthetic_demo"}),
                    SCHEMA_VERSION,
                    now,
                    now,
                )
            )

    conn.executemany(
        """
        INSERT INTO transactions
        (id, org_id, account_id, txn_date, amount, currency, category, description,
         external_ref, source_system_id, provenance, schema_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        txn_rows,
    )

    customers = [
        "Acme Retail",
        "Blue Harbor Foods",
        "Cedar Logistics",
        "Delta Manufacturing",
        "Everline Services",
        "Foxglove Health",
        "Granite Construction",
        "Harbor Point Cafes",
    ]

    invoice_rows = []
    for i in range(95):
        issue = today - timedelta(days=int(rng.integers(0, 35)))
        due = today + timedelta(days=int(rng.integers(1, 95)))
        amount = round(float(rng.uniform(1800, 28000)), 2)
        prob = round(float(rng.uniform(0.72, 0.99)), 3)
        delay = int(rng.integers(0, 8))
        customer = str(rng.choice(customers))
        invoice_rows.append(
            (
                str(uuid.uuid4()),
                org_id,
                f"INV-2026-{i:04d}",
                customer,
                issue.isoformat(),
                due.isoformat(),
                amount,
                "GBP",
                "scheduled",
                prob,
                delay,
                sources["AR ledger"],
                json.dumps({"generator": "synthetic_demo"}),
                SCHEMA_VERSION,
                now,
                now,
            )
        )

    conn.executemany(
        """
        INSERT INTO invoices
        (id, org_id, invoice_number, customer, issue_date, due_date, amount, currency,
         status, collection_probability, delay_days, source_system_id, provenance,
         schema_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        invoice_rows,
    )

    payroll_rows = []
    for i in range(8):
        pay_date = today + timedelta(days=14 * i + 3)
        amount = round(float(12000 + rng.normal(0, 400)), 2)
        payroll_rows.append(
            (
                str(uuid.uuid4()),
                org_id,
                pay_date.isoformat(),
                amount,
                "GBP",
                "scheduled",
                0,
                sources["Payroll system"],
                json.dumps({"generator": "synthetic_demo"}),
                SCHEMA_VERSION,
                now,
                now,
            )
        )

    conn.executemany(
        """
        INSERT INTO payroll_runs
        (id, org_id, payroll_date, gross_amount, currency, status, delay_days,
         source_system_id, provenance, schema_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payroll_rows,
    )

    supplier_rows = []
    suppliers = [
        "Alpine Packaging",
        "Brightline Utilities",
        "CargoNorth",
        "Digital Ops Vendor",
        "EverFresh Supplies",
        "Factory Direct",
    ]
    for i in range(16):
        due = today + timedelta(days=7 * i + 2)
        amount = round(float(4800 + rng.normal(0, 700)), 2)
        delay = int(rng.integers(0, 10))
        supplier = str(rng.choice(suppliers))
        supplier_rows.append(
            (
                str(uuid.uuid4()),
                org_id,
                supplier,
                f"SUP-{i:04d}",
                due.isoformat(),
                amount,
                "GBP",
                delay,
                "net30",
                "scheduled",
                sources["AP ledger"],
                json.dumps({"generator": "synthetic_demo"}),
                SCHEMA_VERSION,
                now,
                now,
            )
        )

    conn.executemany(
        """
        INSERT INTO supplier_obligations
        (id, org_id, supplier, invoice_reference, due_date, amount, currency,
         delay_days, terms, status, source_system_id, provenance, schema_version,
         created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        supplier_rows,
    )

    conn.execute(
        """
        INSERT INTO financing_facilities
        (id, org_id, lender, facility_type, limit_amount, available_amount,
         outstanding_amount, interest_rate, repayment_date, repayment_amount,
         status, provenance, schema_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            org_id,
            "Meridian Bank",
            "revolving",
            150000.0,
            150000.0,
            0.0,
            0.085,
            (today + timedelta(days=180)).isoformat(),
            0.0,
            "active",
            json.dumps({"generator": "synthetic_demo"}),
            SCHEMA_VERSION,
            now,
            now,
        ),
    )

    conn.execute(
        """
        INSERT INTO financing_facilities
        (id, org_id, lender, facility_type, limit_amount, available_amount,
         outstanding_amount, interest_rate, repayment_date, repayment_amount,
         status, provenance, schema_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            org_id,
            "North Capital",
            "term_loan",
            25000.0,
            0.0,
            25000.0,
            0.065,
            (today + timedelta(days=60)).isoformat(),
            25000.0,
            "active",
            json.dumps({"generator": "synthetic_demo"}),
            SCHEMA_VERSION,
            now,
            now,
        ),
    )

    reference_items = [
        ("currencies", sorted(ALLOWED_CURRENCIES)),
        ("transaction_categories", [
            "sales_receipt",
            "payroll",
            "supplier",
            "rent",
            "utilities",
            "tax",
            "loan_draw",
            "loan_repayment",
            "interest",
            "other",
        ]),
        ("roles", ROLES),
    ]
    for key, value in reference_items:
        conn.execute(
            """
            INSERT OR REPLACE INTO reference_data
            (key, version, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, SCHEMA_VERSION, json.dumps(value), now),
        )

    conn.commit()
    conn.close()


def get_default_org_id() -> Optional[str]:
    conn = get_conn()
    row = conn.execute("SELECT id FROM organizations LIMIT 1").fetchone()
    conn.close()
    return row["id"] if row else None


def load_org_name() -> str:
    conn = get_conn()
    row = conn.execute("SELECT name FROM organizations LIMIT 1").fetchone()
    conn.close()
    return row["name"] if row else "Demo Organization"


def load_opening_balance() -> float:
    conn = get_conn()
    row = conn.execute(
        "SELECT opening_balance FROM accounts ORDER BY as_of_date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return float(row["opening_balance"]) if row else 0.0


def get_default_account_id() -> Optional[str]:
    conn = get_conn()
    row = conn.execute("SELECT id FROM accounts LIMIT 1").fetchone()
    conn.close()
    return row["id"] if row else None


def get_or_create_source_system(name: str, system_type: str = "upload") -> str:
    org_id = get_default_org_id()
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM source_systems WHERE name = ? AND org_id = ?",
        (name, org_id),
    ).fetchone()
    if row:
        sid = row["id"]
    else:
        sid = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO source_systems
            (id, org_id, name, system_type, created_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sid, org_id, name, system_type, utc_now(), SCHEMA_VERSION),
        )
        conn.commit()
    conn.close()
    return sid


def load_schedules() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = get_conn()
    invoices = pd.read_sql_query(
        "SELECT * FROM invoices WHERE status NOT IN ('paid', 'cancelled')", conn
    )
    payroll = pd.read_sql_query(
        "SELECT * FROM payroll_runs WHERE status NOT IN ('paid', 'cancelled')", conn
    )
    suppliers = pd.read_sql_query(
        "SELECT * FROM supplier_obligations WHERE status NOT IN ('paid', 'cancelled')", conn
    )
    facilities = pd.read_sql_query(
        "SELECT * FROM financing_facilities WHERE status = 'active'", conn
    )
    transactions = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    return invoices, payroll, suppliers, facilities, transactions


def load_table(table_name: str, limit: int = 500) -> pd.DataFrame:
    allowed = {
        "transactions",
        "invoices",
        "payroll_runs",
        "supplier_obligations",
        "financing_facilities",
        "forecast_runs",
        "forecast_lines",
        "simulation_runs",
        "simulation_events",
        "audit_events",
        "import_batches",
        "import_errors",
        "reference_data",
    }
    if table_name not in allowed:
        raise ValueError("Unsupported table.")
    conn = get_conn()
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df


def record_audit(
    action: str,
    entity_type: str,
    entity_id: Optional[str],
    before: Optional[Any] = None,
    after: Optional[Any] = None,
    reason: str = "",
    actor_role: Optional[str] = None,
) -> None:
    if actor_role is None:
        actor_role = st.session_state.get("role", "system")
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO audit_events
        (id, actor_role, actor_user, action, entity_type, entity_id,
         before_json, after_json, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            actor_role,
            actor_role,
            action,
            entity_type,
            entity_id,
            json.dumps(before, default=str) if before is not None else None,
            json.dumps(after, default=str) if after is not None else None,
            reason,
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()


IMPORT_SPECS = {
    "transactions": {
        "required": ["txn_date", "amount", "currency", "category", "description"],
        "optional": ["external_ref", "account_id"],
        "date_cols": ["txn_date"],
        "numeric_cols": ["amount"],
        "unique_cols": ["external_ref"],
        "category_allowed": {
            "sales_receipt",
            "payroll",
            "supplier",
            "rent",
            "utilities",
            "tax",
            "loan_draw",
            "loan_repayment",
            "interest",
            "other",
        },
    },
    "invoices": {
        "required": ["invoice_number", "due_date", "amount", "currency"],
        "optional": ["customer", "issue_date", "collection_probability", "delay_days"],
        "date_cols": ["due_date", "issue_date"],
        "numeric_cols": ["amount", "collection_probability"],
        "unique_cols": ["invoice_number"],
    },
}


def _issues_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["row_number", "field", "issue", "value", "severity"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records, columns=cols)


def validate_import(
    df: pd.DataFrame,
    import_type: str,
    check_database: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if import_type not in IMPORT_SPECS:
        raise ValueError("Unsupported import type.")

    spec = IMPORT_SPECS[import_type]
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    error_rows: set[int] = set()

    work = df.copy()
    work.columns = [str(c).strip().lower().replace(" ", "_") for c in work.columns]

    if work.empty:
        errors.append(
            {
                "row_number": 0,
                "field": "file",
                "issue": "File contains no data rows.",
                "value": "",
                "severity": "error",
            }
        )
        return pd.DataFrame(), _issues_df(errors), _issues_df(warnings)

    missing = [c for c in spec["required"] if c not in work.columns]
    if missing:
        errors.append(
            {
                "row_number": 0,
                "field": "schema",
                "issue": f"Missing required columns: {', '.join(missing)}",
                "value": "",
                "severity": "error",
            }
        )
        return pd.DataFrame(), _issues_df(errors), _issues_df(warnings)

    for col in spec.get("unique_cols", []):
        if col not in work.columns:
            continue
        dup_mask = work[col].notna() & work.duplicated(subset=[col], keep=False)
        for idx in work.index[dup_mask]:
            error_rows.add(idx)
            errors.append(
                {
                    "row_number": int(idx) + 2,
                    "field": col,
                    "issue": "Duplicate key within uploaded file.",
                    "value": safe_text(work.at[idx, col]),
                    "severity": "error",
                }
            )

    if check_database:
        org_id = get_default_org_id()
        conn = get_conn()
        if import_type == "invoices" and "invoice_number" in work.columns:
            existing = {
                row["invoice_number"]
                for row in conn.execute(
                    "SELECT invoice_number FROM invoices WHERE org_id = ?", (org_id,)
                ).fetchall()
            }
            for idx, value in work["invoice_number"].items():
                text = safe_text(value)
                if text and text in existing:
                    error_rows.add(idx)
                    errors.append(
                        {
                            "row_number": int(idx) + 2,
                            "field": "invoice_number",
                            "issue": "Invoice number already exists in database.",
                            "value": text,
                            "severity": "error",
                        }
                    )
        if import_type == "transactions" and "external_ref" in work.columns:
            existing = {
                row["external_ref"]
                for row in conn.execute(
                    "SELECT external_ref FROM transactions WHERE org_id = ? AND external_ref IS NOT NULL",
                    (org_id,),
                ).fetchall()
                if row["external_ref"]
            }
            for idx, value in work["external_ref"].items():
                text = safe_text(value)
                if text and text in existing:
                    error_rows.add(idx)
                    errors.append(
                        {
                            "row_number": int(idx) + 2,
                            "field": "external_ref",
                            "issue": "External reference already exists in database.",
                            "value": text,
                            "severity": "error",
                        }
                    )
        conn.close()

    today = datetime.now(timezone.utc).date()

    for idx, row in work.iterrows():
        for col in spec["date_cols"]:
            value = row.get(col)
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                error_rows.add(idx)
                errors.append(
                    {
                        "row_number": int(idx) + 2,
                        "field": col,
                        "issue": "Invalid or missing date.",
                        "value": safe_text(value),
                        "severity": "error",
                    }
                )
            else:
                parsed_date = parsed.date()
                if parsed_date > today + timedelta(days=365):
                    warnings.append(
                        {
                            "row_number": int(idx) + 2,
                            "field": col,
                            "issue": "Date is more than 365 days in the future.",
                            "value": parsed_date.isoformat(),
                            "severity": "warning",
                        }
                    )

        for col in spec["numeric_cols"]:
            value = row.get(col)
            try:
                num = float(value)
            except Exception:
                error_rows.add(idx)
                errors.append(
                    {
                        "row_number": int(idx) + 2,
                        "field": col,
                        "issue": "Non-numeric value.",
                        "value": safe_text(value),
                        "severity": "error",
                    }
                )
            else:
                if not np.isfinite(num) or abs(num) > 1_000_000_000:
                    error_rows.add(idx)
                    errors.append(
                        {
                            "row_number": int(idx) + 2,
                            "field": col,
                            "issue": "Numeric value outside allowed range.",
                            "value": safe_text(value),
                            "severity": "error",
                        }
                    )

        currency = row.get("currency")
        if pd.isna(currency) or safe_text(currency).upper() not in ALLOWED_CURRENCIES:
            error_rows.add(idx)
            errors.append(
                {
                    "row_number": int(idx) + 2,
                    "field": "currency",
                    "issue": f"Currency must be one of {sorted(ALLOWED_CURRENCIES)}.",
                    "value": safe_text(currency),
                    "severity": "error",
                }
            )

        if import_type == "transactions":
            category = safe_text(row.get("category")).lower()
            if category not in spec["category_allowed"]:
                error_rows.add(idx)
                errors.append(
                    {
                        "row_number": int(idx) + 2,
                        "field": "category",
                        "issue": "Category is not in the allowed reference catalog.",
                        "value": category,
                        "severity": "error",
                    }
                )
            amount = safe_float(row.get("amount"))
            if amount == 0:
                warnings.append(
                    {
                        "row_number": int(idx) + 2,
                        "field": "amount",
                        "issue": "Zero-value transaction.",
                        "value": "0",
                        "severity": "warning",
                    }
                )

        if import_type == "invoices":
            amount = safe_float(row.get("amount"))
            if amount <= 0:
                error_rows.add(idx)
                errors.append(
                    {
                        "row_number": int(idx) + 2,
                        "field": "amount",
                        "issue": "Invoice amount must be greater than zero.",
                        "value": safe_text(amount),
                        "severity": "error",
                    }
                )

            issue = pd.to_datetime(row.get("issue_date"), errors="coerce")
            due = pd.to_datetime(row.get("due_date"), errors="coerce")
            if pd.notna(issue) and pd.notna(due) and due < issue:
                error_rows.add(idx)
                errors.append(
                    {
                        "row_number": int(idx) + 2,
                        "field": "due_date",
                        "issue": "Due date cannot be earlier than issue date.",
                        "value": f"{issue.date()} < {due.date()}",
                        "severity": "error",
                    }
                )

            prob_value = row.get("collection_probability")
            if pd.notna(prob_value):
                try:
                    prob = float(prob_value)
                    if not 0.0 <= prob <= 1.0:
                        error_rows.add(idx)
                        errors.append(
                            {
                                "row_number": int(idx) + 2,
                                "field": "collection_probability",
                                "issue": "Collection probability must be between 0 and 1.",
                                "value": safe_text(prob_value),
                                "severity": "error",
                            }
                        )
                    elif prob < 0.80:
                        warnings.append(
                            {
                                "row_number": int(idx) + 2,
                                "field": "collection_probability",
                                "issue": "Low collection probability may affect forecast confidence.",
                                "value": safe_text(prob_value),
                                "severity": "warning",
                            }
                        )
                except Exception:
                    error_rows.add(idx)
                    errors.append(
                        {
                            "row_number": int(idx) + 2,
                            "field": "collection_probability",
                            "issue": "Collection probability must be numeric.",
                            "value": safe_text(prob_value),
                            "severity": "error",
                        }
                    )

    valid_indices = [i for i in work.index if i not in error_rows]
    valid_df = work.loc[valid_indices].copy()
    return valid_df, _issues_df(errors), _issues_df(warnings)


def persist_import_errors(batch_id: str, errors_df: pd.DataFrame) -> None:
    if errors_df.empty:
        return
    conn = get_conn()
    rows = []
    for _, r in errors_df.iterrows():
        rows.append(
            (
                str(uuid.uuid4()),
                batch_id,
                safe_int(r.get("row_number"), 0),
                safe_text(r.get("field")),
                safe_text(r.get("issue")),
                safe_text(r.get("value")),
                safe_text(r.get("severity"), "error"),
                json.dumps(r.to_dict(), default=str),
                utc_now(),
            )
        )
    conn.executemany(
        """
        INSERT INTO import_errors
        (id, batch_id, row_number, field, issue, value, severity, raw_row_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def commit_import(
    import_type: str,
    valid_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    filename: str,
    file_bytes: bytes,
    actor_role: str,
) -> Tuple[str, int]:
    if valid_df.empty:
        raise ValueError("No valid rows to commit.")

    org_id = get_default_org_id()
    account_id = get_default_account_id()
    source_id = get_or_create_source_system("Manual upload", "upload")
    checksum = hashlib.sha256(file_bytes).hexdigest()
    batch_id = str(uuid.uuid4())
    now = utc_now()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name.replace("/", "_").replace("\\", "_")
    file_path = UPLOAD_DIR / f"{batch_id}_{safe_name}"
    file_path.write_bytes(file_bytes)

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO import_batches
        (id, org_id, filename, file_checksum, import_type, row_count, status,
         file_path, created_at, created_by, schema_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id,
            org_id,
            filename,
            checksum,
            import_type,
            len(valid_df),
            "committed",
            str(file_path),
            now,
            actor_role,
            SCHEMA_VERSION,
        ),
    )

    rows = []
    if import_type == "transactions":
        for idx, r in valid_df.iterrows():
            rows.append(
                (
                    str(uuid.uuid4()),
                    org_id,
                    account_id,
                    pd.to_datetime(r["txn_date"]).date().isoformat(),
                    float(r["amount"]),
                    safe_text(r["currency"]).upper(),
                    safe_text(r["category"]).lower(),
                    safe_text(r.get("description")),
                    safe_text(r.get("external_ref")) or None,
                    source_id,
                    json.dumps(
                        {
                            "batch_id": batch_id,
                            "filename": filename,
                            "row_number": int(idx) + 2,
                            "checksum": checksum,
                        }
                    ),
                    SCHEMA_VERSION,
                    now,
                    now,
                )
            )
        conn.executemany(
            """
            INSERT INTO transactions
            (id, org_id, account_id, txn_date, amount, currency, category, description,
             external_ref, source_system_id, provenance, schema_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    elif import_type == "invoices":
        for idx, r in valid_df.iterrows():
            issue_parsed = pd.to_datetime(r.get("issue_date"), errors="coerce")
            issue_date = issue_parsed.date().isoformat() if pd.notna(issue_parsed) else None
            rows.append(
                (
                    str(uuid.uuid4()),
                    org_id,
                    safe_text(r["invoice_number"]),
                    safe_text(r.get("customer")),
                    issue_date,
                    pd.to_datetime(r["due_date"]).date().isoformat(),
                    float(r["amount"]),
                    safe_text(r["currency"]).upper(),
                    "scheduled",
                    safe_float(r.get("collection_probability"), 1.0),
                    safe_int(r.get("delay_days"), 0),
                    source_id,
                    json.dumps(
                        {
                            "batch_id": batch_id,
                            "filename": filename,
                            "row_number": int(idx) + 2,
                            "checksum": checksum,
                        }
                    ),
                    SCHEMA_VERSION,
                    now,
                    now,
                )
            )
        conn.executemany(
            """
            INSERT INTO invoices
            (id, org_id, invoice_number, customer, issue_date, due_date, amount, currency,
             status, collection_probability, delay_days, source_system_id, provenance,
             schema_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    conn.commit()
    conn.close()

    persist_import_errors(batch_id, errors_df)
    record_audit(
        action="import_commit",
        entity_type="import_batch",
        entity_id=batch_id,
        after={"import_type": import_type, "row_count": len(valid_df), "filename": filename},
        reason="Committed validated import rows.",
        actor_role=actor_role,
    )
    return batch_id, len(rows)


def project_cashflow(
    opening_balance: float,
    invoices: pd.DataFrame,
    payroll: pd.DataFrame,
    suppliers: pd.DataFrame,
    facilities: pd.DataFrame,
    scenario: Scenario,
    as_of: Optional[date] = None,
) -> pd.DataFrame:
    scenario.validate()

    as_of_date = datetime.now(timezone.utc).date() if as_of is None else as_of
    horizon = scenario.horizon_days

    receipts = np.zeros(horizon + 1, dtype=float)
    payroll_out = np.zeros(horizon + 1, dtype=float)
    supplier_out = np.zeros(horizon + 1, dtype=float)
    repayments = np.zeros(horizon + 1, dtype=float)
    uncertainty = np.zeros(horizon + 1, dtype=float)
    warnings: List[str] = []

    if not invoices.empty:
        inv = add_missing_columns(
            invoices,
            {
                "due_date": pd.NaT,
                "amount": 0.0,
                "collection_probability": 1.0,
                "delay_days": 0,
                "invoice_number": "",
            },
        )
        inv["due_dt"] = pd.to_datetime(inv["due_date"], errors="coerce")
        for _, r in inv.iterrows():
            if pd.isna(r["due_dt"]):
                warnings.append(f"Invoice {safe_text(r.get('invoice_number'))} has invalid due date.")
                continue
            amount = safe_float(r.get("amount"), 0.0)
            if amount <= 0:
                warnings.append(f"Invoice {safe_text(r.get('invoice_number'))} has non-positive amount.")
                continue
            prob = clamp_float(safe_float(r.get("collection_probability"), 1.0), 0.0, 1.0)
            delay = scenario.receipt_delay_days + safe_int(r.get("delay_days"), 0)
            due_date = r["due_dt"].date() + timedelta(days=delay)
            idx = (due_date - as_of_date).days
            if idx < 0:
                idx = 0
                warnings.append(f"Invoice {safe_text(r.get('invoice_number'))} is overdue and pulled to day 0.")
            if idx <= horizon:
                effective = amount * prob * scenario.receipt_confidence * scenario.seasonality_factor
                receipts[idx] += effective
                uncertainty[idx] += amount * (1.0 - prob)

    if not payroll.empty:
        pay = add_missing_columns(
            payroll,
            {
                "payroll_date": pd.NaT,
                "gross_amount": 0.0,
                "delay_days": 0,
            },
        )
        pay["pay_dt"] = pd.to_datetime(pay["payroll_date"], errors="coerce")
        for _, r in pay.iterrows():
            if pd.isna(r["pay_dt"]):
                warnings.append("Payroll run has invalid date.")
                continue
            amount = abs(safe_float(r.get("gross_amount"), 0.0))
            if amount <= 0:
                continue
            delay = scenario.payroll_delay_days + safe_int(r.get("delay_days"), 0)
            pay_date = r["pay_dt"].date() + timedelta(days=delay)
            idx = (pay_date - as_of_date).days
            if idx < 0:
                idx = 0
                warnings.append("A payroll run was overdue and pulled to day 0.")
            if idx <= horizon:
                payroll_out[idx] += amount

    if not suppliers.empty:
        sup = add_missing_columns(
            suppliers,
            {
                "due_date": pd.NaT,
                "amount": 0.0,
                "delay_days": 0,
                "supplier": "",
            },
        )
        sup["due_dt"] = pd.to_datetime(sup["due_date"], errors="coerce")
        for _, r in sup.iterrows():
            if pd.isna(r["due_dt"]):
                warnings.append(f"Supplier obligation for {safe_text(r.get('supplier'))} has invalid due date.")
                continue
            amount = abs(safe_float(r.get("amount"), 0.0))
            if amount <= 0:
                continue
            delay = scenario.supplier_delay_days + safe_int(r.get("delay_days"), 0)
            due_date = r["due_dt"].date() + timedelta(days=delay)
            idx = (due_date - as_of_date).days
            if idx < 0:
                idx = 0
                warnings.append(f"Supplier obligation for {safe_text(r.get('supplier'))} is overdue and pulled to day 0.")
            if idx <= horizon:
                supplier_out[idx] += amount

    available, outstanding, daily_rate = facility_metrics(
        facilities,
        fallback_rate=scenario.financing_interest_rate,
    )

    if not facilities.empty:
        fac = add_missing_columns(
            facilities,
            {
                "repayment_date": pd.NaT,
                "repayment_amount": 0.0,
            },
        )
        fac["repay_dt"] = pd.to_datetime(fac["repayment_date"], errors="coerce")
        for _, r in fac.iterrows():
            if pd.isna(r["repay_dt"]):
                continue
            amount = safe_float(r.get("repayment_amount"), 0.0)
            if amount <= 0:
                continue
            idx = (r["repay_dt"].date() - as_of_date).days
            if idx < 0:
                idx = 0
                warnings.append("A scheduled financing repayment was overdue and pulled to day 0.")
            if idx <= horizon:
                repayments[idx] += amount

    lines = []
    balance = float(opening_balance)

    for idx in range(horizon + 1):
        day = as_of_date + timedelta(days=idx)
        opening = balance
        rec = float(receipts[idx])
        pay = float(payroll_out[idx])
        sup = float(supplier_out[idx])

        pre_financing = opening + rec - pay - sup

        draw = 0.0
        threshold = scenario.minimum_cash - scenario.financing_buffer
        if pre_financing < threshold and available > 0:
            draw = min(available, threshold - pre_financing)
            available -= draw
            outstanding += draw

        repay = min(float(repayments[idx]), outstanding)
        interest = outstanding * daily_rate
        financing_out = repay + interest
        outstanding -= repay

        closing = pre_financing + draw - financing_out
        financing_need = max(0.0, scenario.minimum_cash - closing)

        width = max(
            250.0,
            abs(closing) * 0.025 + float(uncertainty[idx]) * 0.35 + (pay + sup) * 0.01,
        )
        lower = closing - width
        upper = closing + width

        explanation = {
            "formula": "closing = opening + receipts - payroll - supplier_payments + financing_in - financing_out",
            "engine_version": "deterministic_liquidity_engine_v1",
            "schema_version": SCHEMA_VERSION,
            "opening_balance": opening,
            "receipts": rec,
            "payroll": pay,
            "supplier_payments": sup,
            "financing_in": draw,
            "financing_out": financing_out,
            "interest_component": interest,
            "repayment_component": repay,
            "minimum_cash": scenario.minimum_cash,
            "financing_buffer": scenario.financing_buffer,
            "receipt_confidence": scenario.receipt_confidence,
            "receipt_delay_days": scenario.receipt_delay_days,
            "supplier_delay_days": scenario.supplier_delay_days,
            "payroll_delay_days": scenario.payroll_delay_days,
            "reviewer_status": "pending",
        }

        data_quality = []
        if rec == 0:
            data_quality.append("No scheduled receipts on this date.")
        if financing_need > 0:
            data_quality.append("Projected balance is below minimum cash target.")
        if draw > 0:
            data_quality.append("Synthetic financing draw applied to maintain buffer.")
        if idx == 0 and warnings:
            data_quality.extend(warnings[:5])

        lines.append(
            {
                "day_index": idx,
                "date": day.isoformat(),
                "opening_balance": opening,
                "receipts": rec,
                "payroll": pay,
                "supplier_payments": sup,
                "financing_in": draw,
                "financing_out": financing_out,
                "net_change": closing - opening,
                "closing_balance": closing,
                "minimum_cash": scenario.minimum_cash,
                "financing_need": financing_need,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "explanation_json": json.dumps(explanation),
                "data_quality_json": json.dumps(data_quality),
            }
        )

        balance = closing

    return pd.DataFrame(lines)


def apply_overrides(lines: pd.DataFrame, overrides: Dict[str, Dict[str, Any]], scenario: Scenario) -> pd.DataFrame:
    if not overrides:
        return lines
    out = lines.copy()
    for key, ov in overrides.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if idx < 0 or idx >= len(out):
            continue
        delta = safe_float(ov.get("delta"), 0.0)
        if delta == 0:
            continue
        out.loc[out["day_index"] >= idx, "closing_balance"] += delta
        out.loc[out["day_index"] >= idx, "financing_need"] = out.loc[
            out["day_index"] >= idx, "closing_balance"
        ].apply(lambda x: max(0.0, scenario.minimum_cash - float(x)))

        for i in range(idx, len(out)):
            try:
                expl = json.loads(out.at[i, "explanation_json"])
            except Exception:
                expl = {}
            expl["override"] = {
                "override_day_index": idx,
                "delta": delta,
                "reason": safe_text(ov.get("reason"), "No reason supplied."),
            }
            out.at[i, "explanation_json"] = json.dumps(expl)
    return out


def compute_data_quality(
    invoices: pd.DataFrame,
    payroll: pd.DataFrame,
    suppliers: pd.DataFrame,
    facilities: pd.DataFrame,
) -> Tuple[float, List[str]]:
    total = 0
    good = 0
    issues: List[str] = []

    def evaluate_df(
        df: pd.DataFrame,
        name: str,
        date_col: str,
        amount_col: str,
        extra_mask=None,
    ) -> None:
        nonlocal total, good
        if df.empty:
            issues.append(f"No {name} records loaded.")
            return
        total += len(df)
        dates = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series([pd.NaT] * len(df))
        amounts = pd.to_numeric(df[amount_col], errors="coerce") if amount_col in df.columns else pd.Series([np.nan] * len(df))
        mask = dates.notna() & amounts.notna() & (amounts != 0)
        if extra_mask is not None:
            mask &= extra_mask(df)
        good += int(mask.sum())
        bad = int((~mask).sum())
        if bad:
            issues.append(f"{bad} {name} record(s) have missing or invalid date/amount values.")

    def invoice_extra(df: pd.DataFrame) -> pd.Series:
        if "collection_probability" not in df.columns:
            return pd.Series(True, index=df.index)
        return pd.to_numeric(df["collection_probability"], errors="coerce").between(0.0, 1.0)

    evaluate_df(invoices, "invoice", "due_date", "amount", invoice_extra)
    evaluate_df(payroll, "payroll", "payroll_date", "gross_amount")
    evaluate_df(suppliers, "supplier obligation", "due_date", "amount")

    if facilities.empty:
        issues.append("No active financing facilities loaded.")
    else:
        fac = add_missing_columns(
            facilities,
            {
                "available_amount": 0.0,
                "limit_amount": 0.0,
                "interest_rate": 0.0,
            },
        )
        total += len(fac)
        mask = (
            pd.to_numeric(fac["available_amount"], errors="coerce").notna()
            & pd.to_numeric(fac["limit_amount"], errors="coerce").notna()
            & pd.to_numeric(fac["interest_rate"], errors="coerce").between(0.0, 1.0)
        )
        good += int(mask.sum())
        bad = int((~mask).sum())
        if bad:
            issues.append(f"{bad} financing facility record(s) have invalid limits or rates.")

    score = good / total if total else 0.0
    return score, issues


def save_forecast_run(
    scenario: Scenario,
    lines: pd.DataFrame,
    opening_balance: float,
    actor_role: str,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    org_id = get_default_org_id()
    run_id = str(uuid.uuid4())
    now = utc_now()
    as_of = lines["date"].iloc[0] if not lines.empty else datetime.now(timezone.utc).date().isoformat()

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO forecast_runs
        (id, org_id, created_by, created_at, horizon_days, as_of_date, opening_balance,
         scenario_json, status, version, approved_by, approved_at, override_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            org_id,
            actor_role,
            now,
            scenario.horizon_days,
            as_of,
            float(opening_balance),
            json.dumps(asdict(scenario)),
            "draft",
            SCHEMA_VERSION,
            None,
            None,
            json.dumps(overrides or {}, default=str),
            "Saved from forecast lab.",
        ),
    )

    rows = []
    for _, r in lines.iterrows():
        rows.append(
            (
                str(uuid.uuid4()),
                run_id,
                int(r["day_index"]),
                safe_text(r["date"]),
                safe_float(r["opening_balance"]),
                safe_float(r["receipts"]),
                safe_float(r["payroll"]),
                safe_float(r["supplier_payments"]),
                safe_float(r["financing_in"]),
                safe_float(r["financing_out"]),
                safe_float(r["net_change"]),
                safe_float(r["closing_balance"]),
                safe_float(r["minimum_cash"]),
                safe_float(r["financing_need"]),
                safe_float(r["confidence_lower"]),
                safe_float(r["confidence_upper"]),
                safe_text(r["explanation_json"], "{}"),
                safe_text(r["data_quality_json"], "[]"),
            )
        )

    conn.executemany(
        """
        INSERT INTO forecast_lines
        (id, run_id, day_index, date, opening_balance, receipts, payroll, supplier_payments,
         financing_in, financing_out, net_change, closing_balance, minimum_cash, financing_need,
         confidence_lower, confidence_upper, explanation_json, data_quality_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()

    record_audit(
        action="forecast_save",
        entity_type="forecast_run",
        entity_id=run_id,
        after={"status": "draft", "horizon_days": scenario.horizon_days},
        reason="Forecast run saved.",
        actor_role=actor_role,
    )
    return run_id


def change_forecast_status(run_id: str, new_status: str, actor_role: str, reason: str = "") -> None:
    conn = get_conn()
    row = conn.execute("SELECT status, scenario_json FROM forecast_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Forecast run not found.")

    current = row["status"]
    allowed = VALID_TRANSITIONS.get(current, set())
    if new_status not in allowed:
        conn.close()
        raise ValueError(f"Invalid transition: {current} -> {new_status}.")

    now = utc_now()
    if new_status == "approved":
        conn.execute(
            """
            UPDATE forecast_runs
            SET status = ?, approved_by = ?, approved_at = ?, notes = ?
            WHERE id = ?
            """,
            (new_status, actor_role, now, reason, run_id),
        )
    else:
        conn.execute(
            "UPDATE forecast_runs SET status = ?, notes = ? WHERE id = ?",
            (new_status, reason, run_id),
        )
    conn.commit()
    conn.close()

    record_audit(
        action=f"forecast_{new_status}",
        entity_type="forecast_run",
        entity_id=run_id,
        before={"status": current},
        after={"status": new_status},
        reason=reason,
        actor_role=actor_role,
    )


def run_ml_evaluation(transactions_df: pd.DataFrame, holdout_days: int = 30) -> Dict[str, Any]:
    if transactions_df.empty:
        return {"error": "No transaction history available for evaluation."}

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["txn_date"], errors="coerce").dt.normalize()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])

    if df.empty:
        return {"error": "Transaction history contains no valid dated amounts."}

    daily = df.groupby("date", as_index=False)["amount"].sum().sort_values("date").reset_index(drop=True)

    if len(daily) < holdout_days + 30:
        return {"error": "Not enough daily history for a time-based holdout."}

    daily["lag_1"] = daily["amount"].shift(1)
    daily["lag_7"] = daily["amount"].shift(7)
    daily["roll_7"] = daily["amount"].rolling(7).mean()
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["sin_doy"] = np.sin(2 * np.pi * daily["date"].dt.dayofyear / 365.0)
    daily["cos_doy"] = np.cos(2 * np.pi * daily["date"].dt.dayofyear / 365.0)

    features = ["lag_1", "lag_7", "roll_7", "dow", "month", "sin_doy", "cos_doy"]
    model_data = daily.dropna().reset_index(drop=True)

    if len(model_data) < holdout_days + 10:
        return {"error": "Insufficient rows after feature engineering."}

    split = max(10, len(model_data) - holdout_days)
    train = model_data.iloc[:split]
    test = model_data.iloc[split:]

    if train.empty or test.empty:
        return {"error": "Train/test split is empty."}

    reg = LinearRegression()
    reg.fit(train[features], train["amount"])

    pred = reg.predict(test[features])
    naive = test["lag_1"].to_numpy()

    model_mae = float(mean_absolute_error(test["amount"], pred))
    model_rmse = float(np.sqrt(mean_squared_error(test["amount"], pred)))
    naive_mae = float(mean_absolute_error(test["amount"], naive))
    naive_rmse = float(np.sqrt(mean_squared_error(test["amount"], naive)))

    metrics = pd.DataFrame(
        [
            {"Model": "Linear regression residual baseline", "MAE": model_mae, "RMSE": model_rmse},
            {"Model": "Naive lag-1 baseline", "MAE": naive_mae, "RMSE": naive_rmse},
        ]
    )

    plot_df = test[["date", "amount"]].copy()
    plot_df["Model prediction"] = pred
    plot_df["Naive lag-1"] = naive

    coefficients = pd.DataFrame(
        {
            "feature": features,
            "coefficient": [float(c) for c in reg.coef_],
        }
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

    return {
        "metrics": metrics,
        "plot_df": plot_df,
        "coefficients": coefficients,
        "info": {
            "model": "sklearn LinearRegression",
            "target": "next-day net cash flow",
            "unit": "currency per day",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split_method": "chronological holdout; no future leakage",
            "schema_version": SCHEMA_VERSION,
            "limitation": "Trained on synthetic demo data only. Not evidence of real-world predictive performance.",
        },
    }


def render_demo_video_bytes(
    forecast_df: pd.DataFrame,
    scenario: Scenario,
    org_name: str,
) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        import cv2
    except Exception as exc:
        return None, (
            "Video dependencies are unavailable. Install the packages in requirements.txt. "
            f"Import error: {exc}"
        )

    if forecast_df.empty:
        return None, "No forecast data available to render video."

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return None, "Forecast dates are invalid."

    fps = 12
    duration_seconds = 8
    total_frames = fps * duration_seconds
    width, height = 1280, 720
    frames = []

    for fidx in range(total_frames):
        progress = fidx / max(1, total_frames - 1)
        day_idx = int(progress * (len(df) - 1))
        row = df.iloc[day_idx]

        fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)

        ax.fill_between(
            df["date"],
            df["confidence_lower"],
            df["confidence_upper"],
            color="#4C78A8",
            alpha=0.16,
            label="Uncertainty band",
        )
        ax.plot(
            df["date"],
            df["closing_balance"],
            color="#4C78A8",
            linewidth=2.6,
            label="Projected closing balance",
        )
        ax.axhline(
            scenario.minimum_cash,
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label="Minimum cash",
        )
        ax.axvline(row["date"], color="#54A24B", linestyle=":", linewidth=2.2)
        ax.scatter(
            [row["date"]],
            [row["closing_balance"]],
            color="#54A24B",
            s=70,
            zorder=5,
        )

        info = (
            f"Day {int(row['day_index']):03d} | {row['date'].date()}\n"
            f"Closing: {float(row['closing_balance']):,.0f}\n"
            f"Receipts: {float(row['receipts']):,.0f}\n"
            f"Payroll: {float(row['payroll']):,.0f}\n"
            f"Suppliers: {float(row['supplier_payments']):,.0f}\n"
            f"Financing need: {float(row['financing_need']):,.0f}\n"
            f"Receipt delay: {scenario.receipt_delay_days}d\n"
            f"Supplier delay: {scenario.supplier_delay_days}d"
        )

        ax.text(
            0.02,
            0.97,
            info,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.88, edgecolor="#CCCCCC"),
        )

        ax.set_title(f"{org_name} — live liquidity simulation", fontsize=15, pad=12)
        ax.set_ylabel("Cash balance")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.18)
        fig.subplots_adjust(left=0.09, right=0.96, top=0.88, bottom=0.14)
        fig.autofmt_xdate()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        plt.close(fig)
        buf.seek(0)

        img = np.array(Image.open(buf).convert("RGB"))
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    tmp_path = Path(tempfile.gettempdir()) / f"liquidity_demo_{uuid.uuid4().hex}.mp4"

    try:
        writer = None
        for codec in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
            if writer.isOpened():
                break
        if writer is None or not writer.isOpened():
            return None, "OpenCV could not open an MP4 writer on this system."
        for frame in frames:
            writer.write(frame)
        writer.release()
        data = tmp_path.read_bytes()
        return data, None
    except Exception as exc:
        return None, f"Video encoding failed: {exc}"
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def build_management_summary(
    lines: pd.DataFrame,
    scenario: Scenario,
    org_name: str,
    opening_balance: float,
    data_quality_score: float,
) -> str:
    if lines.empty:
        return "# Liquidity projection\n\nNo forecast lines available.\n"

    min_row = lines.loc[lines["closing_balance"].idxmin()]
    end_row = lines.iloc[-1]
    gap_rows = lines[lines["financing_need"] > 0]

    gap_text = "No projected financing need under the current scenario."
    if not gap_rows.empty:
        first_gap = gap_rows.iloc[0]
        total_need = float(gap_rows["financing_need"].sum())
        gap_text = (
            f"First projected financing need occurs on {first_gap['date']} "
            f"with estimated need {float(first_gap['financing_need']):,.0f}. "
            f"Aggregate projected shortfall across horizon: {total_need:,.0f}."
        )

    return f"""# Liquidity projection summary

Organization: {org_name}  
Generated: {utc_now()}  
Data classification: synthetic demo data  
Decision status: decision support only; not regulated financial advice

## Headline metrics

- Opening balance: {opening_balance:,.0f}
- Horizon: {scenario.horizon_days} days
- Minimum cash target: {scenario.minimum_cash:,.0f}
- Financing buffer: {scenario.financing_buffer:,.0f}
- Projected ending balance: {float(end_row['closing_balance']):,.0f}
- Lowest projected balance: {float(min_row['closing_balance']):,.0f} on {min_row['date']}
- Data quality score: {data_quality_score:.0%}

## Scenario controls

- Receipt delay: {scenario.receipt_delay_days} days
- Receipt confidence: {scenario.receipt_confidence:.0%}
- Payroll delay: {scenario.payroll_delay_days} days
- Supplier delay: {scenario.supplier_delay_days} days
- Financing APR assumption: {scenario.financing_interest_rate:.2%}

## Financing interpretation

{gap_text}

## Explainability

Each projected day includes:
- opening balance,
- scheduled receipts,
- payroll outflows,
- supplier outflows,
- financing inflows/outflows,
- uncertainty band,
- data-quality warnings,
- reviewer status,
- engine version and formula.

## Governance note

This report is intended for authorized human review. It does not make autonomous lending decisions and does not promise returns.
"""


def _test_scenario_validation() -> None:
    try:
        Scenario(horizon_days=-5).validate()
    except ValueError:
        return
    raise AssertionError("Invalid scenario was accepted.")


def _test_deterministic_forecast() -> None:
    invoices = pd.DataFrame(
        [
            {
                "invoice_number": "T-1",
                "due_date": "2026-01-02",
                "amount": 500.0,
                "collection_probability": 1.0,
                "delay_days": 0,
            }
        ]
    )
    payroll = pd.DataFrame(
        [
            {
                "payroll_date": "2026-01-03",
                "gross_amount": 300.0,
                "delay_days": 0,
            }
        ]
    )
    suppliers = pd.DataFrame(
        [
            {
                "supplier": "Test Supplier",
                "due_date": "2026-01-03",
                "amount": 200.0,
                "delay_days": 0,
            }
        ]
    )
    facilities = pd.DataFrame()
    scenario = Scenario(
        horizon_days=3,
        receipt_delay_days=0,
        receipt_confidence=1.0,
        payroll_delay_days=0,
        supplier_delay_days=0,
        minimum_cash=0.0,
        financing_buffer=0.0,
        financing_interest_rate=0.0,
        seasonality_factor=1.0,
    )
    lines = project_cashflow(
        opening_balance=1000.0,
        invoices=invoices,
        payroll=payroll,
        suppliers=suppliers,
        facilities=facilities,
        scenario=scenario,
        as_of=date(2026, 1, 1),
    )
    if not np.isclose(lines.loc[1, "closing_balance"], 1500.0):
        raise AssertionError("Day 1 closing balance mismatch.")
    if not np.isclose(lines.loc[2, "closing_balance"], 1000.0):
        raise AssertionError("Day 2 closing balance mismatch.")


def _test_import_validation() -> None:
    bad = pd.DataFrame(
        [
            {
                "txn_date": "not-a-date",
                "amount": "abc",
                "currency": "ZZZ",
                "category": "unknown_category",
                "description": "bad row",
            }
        ]
    )
    valid_df, errors_df, _ = validate_import(bad, "transactions", check_database=False)
    if not valid_df.empty:
        raise AssertionError("Malformed row was incorrectly marked valid.")
    if errors_df.empty:
        raise AssertionError("Malformed row did not produce an error.")


def _test_reproducibility() -> None:
    invoices = pd.DataFrame(
        [
            {
                "invoice_number": "R-1",
                "due_date": "2026-02-03",
                "amount": 1200.0,
                "collection_probability": 0.9,
                "delay_days": 1,
            }
        ]
    )
    payroll = pd.DataFrame()
    suppliers = pd.DataFrame()
    facilities = pd.DataFrame()
    scenario = Scenario(horizon_days=10)
    a = project_cashflow(5000.0, invoices, payroll, suppliers, facilities, scenario, as_of=date(2026, 2, 1))
    b = project_cashflow(5000.0, invoices, payroll, suppliers, facilities, scenario, as_of=date(2026, 2, 1))
    if not np.allclose(a["closing_balance"].to_numpy(), b["closing_balance"].to_numpy()):
        raise AssertionError("Forecast calculation was not reproducible.")


def _test_audit_persistence() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    conn.execute(
        """
        INSERT INTO audit_events
        (id, actor_role, actor_user, action, entity_type, entity_id,
         before_json, after_json, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            "tester",
            "tester",
            "unit_test",
            "test_entity",
            "test-id",
            None,
            None,
            "audit persistence check",
            utc_now(),
        ),
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) AS c FROM audit_events").fetchone()["c"]
    conn.close()
    if count != 1:
        raise AssertionError("Audit event was not persisted.")


def run_self_tests() -> pd.DataFrame:
    results = []

    def add(name: str, fn) -> None:
        try:
            fn()
            results.append({"Test": name, "Result": "PASS", "Details": ""})
        except AssertionError as exc:
            results.append({"Test": name, "Result": "FAIL", "Details": str(exc)})
        except Exception as exc:
            results.append({"Test": name, "Result": "ERROR", "Details": repr(exc)})

    add("Scenario validation rejects invalid inputs", _test_scenario_validation)
    add("Deterministic cash-flow engine matches hand-checkable example", _test_deterministic_forecast)
    add("Import validator rejects malformed rows with actionable errors", _test_import_validation)
    add("Forecast engine is reproducible for identical inputs", _test_reproducibility)
    add("Audit event persistence works", _test_audit_persistence)

    return pd.DataFrame(results)


def default_sim_config() -> Dict[str, Any]:
    return {
        "speed_ms": 250,
        "stochastic": True,
        "auto_finance": True,
    }


def build_simulation_schedule(
    invoices: pd.DataFrame,
    payroll: pd.DataFrame,
    suppliers: pd.DataFrame,
    facilities: pd.DataFrame,
    scenario: Scenario,
    seed: int,
    as_of_date: date,
    stochastic: bool = True,
) -> Dict[str, Any]:
    horizon = scenario.horizon_days
    receipts = np.zeros(horizon + 1, dtype=float)
    payroll_arr = np.zeros(horizon + 1, dtype=float)
    supplier_arr = np.zeros(horizon + 1, dtype=float)
    repayment_arr = np.zeros(horizon + 1, dtype=float)
    events: List[List[Dict[str, Any]]] = [[] for _ in range(horizon + 1)]
    rng = np.random.default_rng(normalize_seed(seed))

    def clamp_idx(idx: int) -> int:
        return max(0, min(horizon, int(idx)))

    if not invoices.empty:
        inv = add_missing_columns(
            invoices,
            {
                "invoice_number": "",
                "customer": "",
                "due_date": pd.NaT,
                "amount": 0.0,
                "collection_probability": 1.0,
                "delay_days": 0,
            },
        )
        inv["due_dt"] = pd.to_datetime(inv["due_date"], errors="coerce")
        for _, r in inv.iterrows():
            due = r["due_dt"]
            if pd.isna(due):
                continue
            amount = safe_float(r.get("amount"), 0.0)
            if amount <= 0:
                continue
            prob = clamp_float(
                safe_float(r.get("collection_probability"), 1.0) * scenario.receipt_confidence,
                0.0,
                1.0,
            )
            base_idx = (
                due.date()
                + timedelta(days=scenario.receipt_delay_days + safe_int(r.get("delay_days"), 0))
                - as_of_date
            ).days
            base_idx = clamp_idx(base_idx)

            u = rng.random()
            if u < prob:
                day = clamp_idx(base_idx + (int(rng.integers(0, 3)) if stochastic else 0))
                amt = float(amount)
                etype = "receipt"
                sev = "info"
            elif u < min(1.0, prob + 0.18):
                day = clamp_idx(base_idx + (int(rng.integers(7, 15)) if stochastic else 7))
                amt = float(amount * rng.uniform(0.65, 0.90))
                etype = "late_receipt"
                sev = "warning"
            else:
                day = base_idx
                amt = 0.0
                etype = "receipt_default"
                sev = "critical"

            receipts[day] += amt
            events[day].append(
                {
                    "event_type": etype,
                    "description": f"Invoice {safe_text(r.get('invoice_number'))} — {safe_text(r.get('customer'))}",
                    "amount": float(amt),
                    "severity": sev,
                    "payload": {
                        "invoice_number": safe_text(r.get("invoice_number")),
                        "customer": safe_text(r.get("customer")),
                        "base_due_date": due.date().isoformat(),
                        "probability": float(prob),
                    },
                }
            )

    if not payroll.empty:
        pay = add_missing_columns(
            payroll,
            {
                "payroll_date": pd.NaT,
                "gross_amount": 0.0,
                "delay_days": 0,
            },
        )
        pay["pay_dt"] = pd.to_datetime(pay["payroll_date"], errors="coerce")
        for _, r in pay.iterrows():
            pay_date = r["pay_dt"]
            if pd.isna(pay_date):
                continue
            amount = abs(safe_float(r.get("gross_amount"), 0.0))
            if amount <= 0:
                continue
            base_idx = (
                pay_date.date()
                + timedelta(days=scenario.payroll_delay_days + safe_int(r.get("delay_days"), 0))
                - as_of_date
            ).days
            base_idx = clamp_idx(base_idx)
            day = clamp_idx(base_idx + (int(rng.integers(0, 2)) if stochastic else 0))
            noise = float(rng.normal(0, 0.01)) if stochastic else 0.0
            amt = float(amount * (1.0 + noise))
            payroll_arr[day] += amt
            events[day].append(
                {
                    "event_type": "payroll",
                    "description": "Payroll run",
                    "amount": float(amt),
                    "severity": "info",
                    "payload": {
                        "base_payroll_date": pay_date.date().isoformat(),
                        "noise": noise,
                    },
                }
            )

    if not suppliers.empty:
        sup = add_missing_columns(
            suppliers,
            {
                "supplier": "",
                "invoice_reference": "",
                "due_date": pd.NaT,
                "amount": 0.0,
                "delay_days": 0,
            },
        )
        sup["due_dt"] = pd.to_datetime(sup["due_date"], errors="coerce")
        for _, r in sup.iterrows():
            due = r["due_dt"]
            if pd.isna(due):
                continue
            amount = abs(safe_float(r.get("amount"), 0.0))
            if amount <= 0:
                continue
            base_idx = (
                due.date()
                + timedelta(days=scenario.supplier_delay_days + safe_int(r.get("delay_days"), 0))
                - as_of_date
            ).days
            base_idx = clamp_idx(base_idx)
            day = clamp_idx(base_idx + (int(rng.integers(0, 6)) if stochastic else 0))
            noise = float(rng.normal(0, 0.02)) if stochastic else 0.0
            amt = float(amount * (1.0 + noise))
            supplier_arr[day] += amt
            events[day].append(
                {
                    "event_type": "supplier_payment",
                    "description": f"Supplier {safe_text(r.get('supplier'))} — {safe_text(r.get('invoice_reference'))}",
                    "amount": float(amt),
                    "severity": "info",
                    "payload": {
                        "supplier": safe_text(r.get("supplier")),
                        "base_due_date": due.date().isoformat(),
                        "noise": noise,
                    },
                }
            )

    if not facilities.empty:
        fac = add_missing_columns(
            facilities,
            {
                "lender": "",
                "repayment_date": pd.NaT,
                "repayment_amount": 0.0,
            },
        )
        fac["repay_dt"] = pd.to_datetime(fac["repayment_date"], errors="coerce")
        for _, r in fac.iterrows():
            repay_date = r["repay_dt"]
            amount = safe_float(r.get("repayment_amount"), 0.0)
            if pd.isna(repay_date) or amount <= 0:
                continue
            idx = clamp_idx((repay_date.date() - as_of_date).days)
            repayment_arr[idx] += amount
            events[idx].append(
                {
                    "event_type": "financing_repayment",
                    "description": f"Scheduled repayment — {safe_text(r.get('lender'))}",
                    "amount": float(amount),
                    "severity": "info",
                    "payload": {
                        "lender": safe_text(r.get("lender")),
                        "repayment_date": repay_date.date().isoformat(),
                    },
                }
            )

    return {
        "receipts": receipts,
        "payroll": payroll_arr,
        "suppliers": supplier_arr,
        "repayments": repayment_arr,
        "events": events,
    }


def compute_simulation_metrics(state: Dict[str, Any], scenario: Scenario) -> Dict[str, Any]:
    hist = pd.DataFrame(state.get("history", []))
    if hist.empty:
        return {}

    finite_var = pd.to_numeric(hist.get("variance", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "days_simulated": int(len(hist)),
        "ending_balance": float(hist["closing_balance"].iloc[-1]),
        "minimum_balance": float(hist["closing_balance"].min()),
        "days_below_minimum": int((hist["closing_balance"] < scenario.minimum_cash).sum()),
        "total_financing_draws": float(hist["financing_in"].sum()),
        "total_financing_need": float(hist["financing_need"].sum()),
        "forecast_mae": float(np.nanmean(np.abs(finite_var))) if not finite_var.empty else 0.0,
        "forecast_rmse": float(np.sqrt(np.nanmean(np.square(finite_var)))) if not finite_var.empty else 0.0,
        "alert_count": int(len(state.get("alerts", []))),
        "event_count": int(len(state.get("events", []))),
    }


def step_simulation(state: Dict[str, Any], scenario: Scenario, config: Dict[str, Any]) -> None:
    horizon = scenario.horizon_days
    day = int(state["current_day"])
    if day > horizon:
        state["playing"] = False
        return

    current_date = state["as_of_date"] + timedelta(days=day)
    sched = state["schedule"]

    receipts = float(sched["receipts"][day]) if day < len(sched["receipts"]) else 0.0
    payroll = float(sched["payroll"][day]) if day < len(sched["payroll"]) else 0.0
    suppliers = float(sched["suppliers"][day]) if day < len(sched["suppliers"]) else 0.0
    repayments = float(sched["repayments"][day]) if day < len(sched["repayments"]) else 0.0
    events = [dict(e) for e in sched["events"][day]] if day < len(sched["events"]) else []

    financing_in = 0.0

    for shock in state.get("shocks", []):
        start = int(shock.get("start_day", 0))
        duration = max(1, int(shock.get("duration", 1)))
        magnitude = float(shock.get("magnitude", 0.0))
        typ = safe_text(shock.get("type"))
        label = safe_text(shock.get("label"), typ)

        if start <= day < start + duration:
            if typ == "receipt_shortfall":
                reduction = receipts * clamp_float(magnitude, 0.0, 1.0)
                receipts -= reduction
                if reduction > 0:
                    events.append(
                        {
                            "event_type": "shock_receipt_shortfall",
                            "description": label,
                            "amount": float(-reduction),
                            "severity": "warning",
                            "payload": dict(shock),
                        }
                    )
            elif typ == "payroll_surge":
                payroll += magnitude
                events.append(
                    {
                        "event_type": "shock_payroll_surge",
                        "description": label,
                        "amount": float(magnitude),
                        "severity": "warning",
                        "payload": dict(shock),
                    }
                )
            elif typ == "supplier_acceleration":
                suppliers += magnitude
                events.append(
                    {
                        "event_type": "shock_supplier_acceleration",
                        "description": label,
                        "amount": float(magnitude),
                        "severity": "warning",
                        "payload": dict(shock),
                    }
                )
            elif typ == "unexpected_tax":
                suppliers += magnitude
                events.append(
                    {
                        "event_type": "shock_unexpected_tax",
                        "description": label,
                        "amount": float(magnitude),
                        "severity": "critical",
                        "payload": dict(shock),
                    }
                )
            elif typ == "financing_injection":
                if day == start:
                    financing_in += magnitude
                    events.append(
                        {
                            "event_type": "shock_financing_injection",
                            "description": label,
                            "amount": float(magnitude),
                            "severity": "info",
                            "payload": dict(shock),
                        }
                    )

    opening = float(state["balance"])
    pre_financing = opening + receipts - payroll - suppliers

    threshold = scenario.minimum_cash - scenario.financing_buffer
    if config.get("auto_finance", True) and pre_financing < threshold and state["available"] > 0:
        draw = min(float(state["available"]), float(threshold - pre_financing))
        if draw > 0:
            financing_in += draw
            state["available"] -= draw
            state["outstanding"] += draw
            events.append(
                {
                    "event_type": "financing_draw",
                    "description": "Automatic financing draw to maintain buffer",
                    "amount": float(draw),
                    "severity": "warning",
                    "payload": {
                        "available_after": float(state["available"]),
                        "outstanding_after": float(state["outstanding"]),
                    },
                }
            )

    interest = float(state["outstanding"]) * float(state.get("daily_rate", 0.0))
    repay = min(float(repayments), float(state["outstanding"]))
    financing_out = repay + interest
    state["outstanding"] -= repay

    closing = pre_financing + financing_in - financing_out
    net_change = closing - opening

    fc = state.get("forecast_lines")
    forecast_close = np.nan
    if fc is not None and not fc.empty and day < len(fc):
        forecast_close = safe_float(fc.iloc[day]["closing_balance"], np.nan)

    variance = closing - forecast_close if np.isfinite(forecast_close) else np.nan
    financing_need = max(0.0, scenario.minimum_cash - closing)

    row = {
        "day_index": day,
        "date": current_date.isoformat(),
        "opening_balance": float(opening),
        "receipts": float(receipts),
        "payroll": float(payroll),
        "supplier_payments": float(suppliers),
        "financing_in": float(financing_in),
        "financing_out": float(financing_out),
        "net_change": float(net_change),
        "closing_balance": float(closing),
        "minimum_cash": float(scenario.minimum_cash),
        "financing_need": float(financing_need),
        "forecast_closing_balance": None if not np.isfinite(forecast_close) else float(forecast_close),
        "variance": None if not np.isfinite(variance) else float(variance),
        "available_facility": float(state["available"]),
        "outstanding_facility": float(state["outstanding"]),
    }

    state["history"].append(row)
    state["balance"] = float(closing)

    for e in events:
        e["day_index"] = day
        e["date"] = current_date.isoformat()
        e["balance_after"] = float(closing)
        e["created_at"] = utc_now()
        state["events"].append(e)

    if closing < scenario.minimum_cash:
        state["alerts"].append(
            {
                "day_index": day,
                "date": current_date.isoformat(),
                "severity": "critical" if closing < 0 else "warning",
                "message": f"Balance {closing:,.0f} below minimum cash {scenario.minimum_cash:,.0f}.",
            }
        )

    if np.isfinite(variance) and variance < -max(10000.0, abs(opening) * 0.03):
        state["alerts"].append(
            {
                "day_index": day,
                "date": current_date.isoformat(),
                "severity": "warning",
                "message": f"Simulation missed forecast by {variance:,.0f}.",
            }
        )

    if receipts == 0 and (payroll + suppliers) > 0:
        state["alerts"].append(
            {
                "day_index": day,
                "date": current_date.isoformat(),
                "severity": "warning",
                "message": "No receipts while payroll/supplier outflows occurred.",
            }
        )

    state["current_day"] = day + 1
    if state["current_day"] > horizon:
        state["playing"] = False

    state["metrics"] = compute_simulation_metrics(state, scenario)


def init_simulation_state(
    forecast_lines: pd.DataFrame,
    opening_balance: float,
    scenario: Scenario,
    invoices: pd.DataFrame,
    payroll: pd.DataFrame,
    suppliers: pd.DataFrame,
    facilities: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    config: Optional[Dict[str, Any]] = None,
    shocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if config is None:
        config = default_sim_config()
    if shocks is None:
        shocks = []

    seed = normalize_seed(seed)
    stochastic = bool(config.get("stochastic", True))

    as_of_date = datetime.now(timezone.utc).date()
    if forecast_lines is not None and not forecast_lines.empty:
        first_date = pd.to_datetime(forecast_lines.iloc[0]["date"], errors="coerce")
        if pd.notna(first_date):
            as_of_date = first_date.date()

    schedule = build_simulation_schedule(
        invoices=invoices,
        payroll=payroll,
        suppliers=suppliers,
        facilities=facilities,
        scenario=scenario,
        seed=seed,
        as_of_date=as_of_date,
        stochastic=stochastic,
    )

    available, outstanding, daily_rate = facility_metrics(
        facilities,
        fallback_rate=scenario.financing_interest_rate,
    )

    state = {
        "run_id": str(uuid.uuid4()),
        "as_of_date": as_of_date,
        "current_day": 0,
        "playing": True,
        "seed": int(seed),
        "pending_seed": int(seed),
        "balance": float(opening_balance),
        "opening_balance": float(opening_balance),
        "available": float(available),
        "outstanding": float(outstanding),
        "daily_rate": float(daily_rate),
        "history": [],
        "events": [],
        "alerts": [],
        "shocks": list(shocks),
        "schedule": schedule,
        "forecast_lines": forecast_lines.copy() if forecast_lines is not None else pd.DataFrame(),
        "scenario_snapshot": asdict(scenario),
        "scenario_signature": scenario.signature(),
        "config": dict(config),
        "metrics": {},
        "created_at": utc_now(),
        "schedule_stochastic_applied": stochastic,
    }

    step_simulation(state, scenario, state["config"])
    return state


def rebuild_simulation_from_context(
    ctx: Dict[str, Any],
    scenario: Scenario,
    seed: int = DEFAULT_SEED,
    config: Optional[Dict[str, Any]] = None,
    shocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if config is None:
        config = default_sim_config()
    if shocks is None:
        shocks = []

    lines = project_cashflow(
        opening_balance=ctx["opening_balance"],
        invoices=ctx["invoices"],
        payroll=ctx["payroll"],
        suppliers=ctx["suppliers"],
        facilities=ctx["facilities"],
        scenario=scenario,
    )
    return init_simulation_state(
        forecast_lines=lines,
        opening_balance=ctx["opening_balance"],
        scenario=scenario,
        invoices=ctx["invoices"],
        payroll=ctx["payroll"],
        suppliers=ctx["suppliers"],
        facilities=ctx["facilities"],
        seed=normalize_seed(seed),
        config=config,
        shocks=shocks,
    )


def save_simulation_run(state: Dict[str, Any], scenario: Scenario, actor_role: str, notes: str = "") -> str:
    org_id = get_default_org_id()
    run_id = state["run_id"]
    now = utc_now()
    metrics = state.get("metrics") or compute_simulation_metrics(state, scenario)
    hist = pd.DataFrame(state.get("history", []))

    ending_balance = float(hist["closing_balance"].iloc[-1]) if not hist.empty else float(state["balance"])
    minimum_balance = float(hist["closing_balance"].min()) if not hist.empty else float(state["balance"])
    days_below = int(metrics.get("days_below_minimum", 0))
    total_draws = float(metrics.get("total_financing_draws", 0.0))
    status = "completed" if state["current_day"] > scenario.horizon_days else "in_progress"

    conn = get_conn()
    conn.execute("DELETE FROM simulation_events WHERE run_id = ?", (run_id,))
    conn.execute(
        """
        INSERT OR REPLACE INTO simulation_runs
        (id, org_id, created_by, created_at, seed, as_of_date, horizon_days, opening_balance,
         ending_balance, minimum_balance, days_below_minimum, total_financing_draws,
         scenario_json, config_json, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            org_id,
            actor_role,
            now,
            int(state["seed"]),
            state["as_of_date"].isoformat(),
            int(scenario.horizon_days),
            float(state["opening_balance"]),
            ending_balance,
            minimum_balance,
            days_below,
            total_draws,
            json.dumps(state.get("scenario_snapshot", {}), default=str),
            json.dumps(state.get("config", {}), default=str),
            status,
            notes,
        ),
    )

    rows = []
    for e in state.get("events", []):
        rows.append(
            (
                str(uuid.uuid4()),
                run_id,
                int(e.get("day_index", 0)),
                safe_text(e.get("date")),
                safe_text(e.get("event_type"), "event"),
                safe_text(e.get("description")),
                safe_float(e.get("amount"), 0.0),
                safe_float(e.get("balance_after"), 0.0),
                safe_text(e.get("severity"), "info"),
                json.dumps(e.get("payload", {}), default=str),
                utc_now(),
            )
        )

    conn.executemany(
        """
        INSERT INTO simulation_events
        (id, run_id, day_index, date, event_type, description, amount, balance_after,
         severity, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()

    record_audit(
        action="simulation_save",
        entity_type="simulation_run",
        entity_id=run_id,
        after={"status": status, "events": len(rows), "seed": int(state["seed"])},
        reason=notes or "Simulation run saved.",
        actor_role=actor_role,
    )
    return run_id


def combined_stress_shocks(start_day: int, horizon_days: int) -> List[Dict[str, Any]]:
    def clamp_day(d: int) -> int:
        return max(0, min(horizon_days, int(d)))

    return [
        {
            "id": str(uuid.uuid4()),
            "type": "receipt_shortfall",
            "start_day": clamp_day(start_day),
            "duration": 21,
            "magnitude": 0.45,
            "label": "Receipt shortfall 45%",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "payroll_surge",
            "start_day": clamp_day(start_day),
            "duration": 3,
            "magnitude": 7500.0,
            "label": "Payroll surge",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "supplier_acceleration",
            "start_day": clamp_day(start_day),
            "duration": 1,
            "magnitude": 35000.0,
            "label": "Supplier acceleration",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "unexpected_tax",
            "start_day": clamp_day(start_day + 10),
            "duration": 1,
            "magnitude": 18000.0,
            "label": "Unexpected tax",
        },
    ]


def severe_stress_scenario(scenario: Scenario) -> Scenario:
    data = asdict(scenario)
    data["minimum_cash"] = max(float(scenario.minimum_cash), 150000.0)
    data["receipt_confidence"] = min(float(scenario.receipt_confidence), 0.65)
    data["horizon_days"] = min(int(scenario.horizon_days), 60)
    return Scenario(**data)


def simulate_to_end(
    ctx: Dict[str, Any],
    scenario: Scenario,
    seed: int,
    config: Dict[str, Any],
    shocks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    state = rebuild_simulation_from_context(ctx, scenario, seed=seed, config=config, shocks=shocks)
    guard = 0
    while state["current_day"] <= scenario.horizon_days and guard <= scenario.horizon_days + 5:
        step_simulation(state, scenario, state["config"])
        guard += 1
    hist = pd.DataFrame(state["history"])
    return state, hist


def run_simulation_test_suite(
    ctx: Dict[str, Any],
    scenario: Scenario,
    config: Dict[str, Any],
) -> pd.DataFrame:
    results = []

    def add(name: str, fn) -> None:
        try:
            ok, details = fn()
            results.append({"Test": name, "Result": "PASS" if ok else "FAIL", "Details": details})
        except Exception as exc:
            results.append({"Test": name, "Result": "ERROR", "Details": repr(exc)})

    def test_reproducibility():
        s1, h1 = simulate_to_end(ctx, scenario, seed=999, config=config)
        s2, h2 = simulate_to_end(ctx, scenario, seed=999, config=config)
        p1 = json.dumps(h1.to_dict(orient="records"), default=str, sort_keys=True)
        p2 = json.dumps(h2.to_dict(orient="records"), default=str, sort_keys=True)
        ok = hashlib.sha256(p1.encode("utf-8")).hexdigest() == hashlib.sha256(p2.encode("utf-8")).hexdigest()
        return ok, "Same seed produced identical simulated history." if ok else "Histories differed."

    def test_row_count():
        _, h = simulate_to_end(ctx, scenario, seed=999, config=config)
        ok = len(h) == scenario.horizon_days + 1
        return ok, f"Rows={len(h)}, expected={scenario.horizon_days + 1}"

    def test_no_nan_core_fields():
        _, h = simulate_to_end(ctx, scenario, seed=999, config=config)
        cols = ["closing_balance", "receipts", "payroll", "supplier_payments", "financing_in", "financing_out"]
        bad = h[cols].isna().any().any()
        return not bool(bad), "No NaNs in core simulation fields." if not bad else "NaNs detected."

    def test_stress_triggers_alert_or_financing():
        stress_scn = severe_stress_scenario(scenario)
        shocks = combined_stress_shocks(0, stress_scn.horizon_days)
        s, h = simulate_to_end(ctx, stress_scn, seed=999, config=config, shocks=shocks)
        ok = (
            len(s["alerts"]) > 0
            or float(h["financing_in"].sum()) > 0
            or float(h["closing_balance"].min()) < stress_scn.minimum_cash
        )
        return ok, f"Alerts={len(s['alerts'])}, draws={float(h['financing_in'].sum()):,.0f}"

    def test_no_auto_finance_still_detects_shortfall():
        stress_scn = severe_stress_scenario(scenario)
        shocks = combined_stress_shocks(0, stress_scn.horizon_days)
        cfg = dict(config)
        cfg["auto_finance"] = False
        _, h = simulate_to_end(ctx, stress_scn, seed=999, config=cfg, shocks=shocks)
        ok = float(h["closing_balance"].min()) < stress_scn.minimum_cash
        return ok, f"Minimum balance={float(h['closing_balance'].min()):,.0f}"

    def test_event_log_non_empty():
        stress_scn = severe_stress_scenario(scenario)
        shocks = combined_stress_shocks(0, stress_scn.horizon_days)
        s, _ = simulate_to_end(ctx, stress_scn, seed=999, config=config, shocks=shocks)
        ok = len(s["events"]) > 0
        return ok, f"Events={len(s['events'])}"

    add("Seeded live simulation is reproducible", test_reproducibility)
    add("Simulation produces one row per day including day 0", test_row_count)
    add("Core simulation fields contain no NaNs", test_no_nan_core_fields)
    add("Combined stress triggers alert, financing draw, or shortfall", test_stress_triggers_alert_or_financing)
    add("Without auto financing, stress produces balance below minimum", test_no_auto_finance_still_detects_shortfall)
    add("Simulation event log is non-empty", test_event_log_non_empty)

    return pd.DataFrame(results)


PRODUCT_BRIEF_MD = """
# Product brief — SME Cash-Flow Forecasting and Liquidity Intelligence

## Problem

Small and mid-sized enterprises, lenders, payment companies, and treasury teams often manage short-term liquidity using fragmented bank extracts, AR/AP ledgers, payroll calendars, supplier terms, and manual spreadsheets. The current workflow is slow, error-prone, and difficult to explain. Finance operators spend time assembling data rather than interpreting it. Reviewers cannot easily see why a projected cash shortfall appears, whether it is caused by delayed receipts, payroll timing, supplier terms, or financing constraints.

## Narrow initial workflow

The first release solves one end-to-end workflow:

1. Load or import business cash-movement records.
2. Validate records with row-level errors and warnings.
3. Run a deterministic short- and medium-term liquidity projection.
4. Apply scenario controls for delayed receipts, payroll, supplier payments, and financing needs.
5. Run a live simulation test harness with seeded stochastic cash events.
6. Inject shocks and stress scenarios for testing.
7. Present explainable outputs with uncertainty and data-quality warnings.
8. Allow analyst drafting, reviewer approval, manual override with reason, and export.

## Target users

- Administrator: manages reference data, roles, retention, audit, and system configuration.
- Analyst/operator: imports data, prepares forecasts, runs scenarios, runs live simulations, and drafts reports.
- Reviewer: challenges assumptions, approves or rejects outputs, and applies governed overrides.
- Read-only stakeholder: views approved outputs and exports reports.

## Measurable success criteria

- A documented sample file can be imported successfully.
- Malformed rows are rejected with actionable row-level errors.
- Core calculations are reproducible from the same inputs and scenario version.
- Live simulation is reproducible from the same seed.
- Shock injection changes future cash events and is audited.
- Every projected output has traceable inputs, rules, uncertainty, and reviewer status.
- A report can be exported as CSV, JSON, and Markdown.
- Automated self-tests and simulation tests pass locally on Windows.

## Out of scope for MVP

- Autonomous lending decisions.
- Regulated financial advice.
- Promises of investment returns.
- Direct live bank API integrations.
- Multi-tenant cloud deployment.
"""

FUNCTIONAL_REQUIREMENTS_DF = pd.DataFrame(
    [
        {
            "ID": "FR-01",
            "Requirement": "Load synthetic demo data automatically on first run.",
            "Acceptance": "App shows non-empty transactions, invoices, payroll, suppliers, facilities, and forecast chart without user setup.",
        },
        {
            "ID": "FR-02",
            "Requirement": "Import CSV/XLSX transaction and invoice records.",
            "Acceptance": "Valid rows are previewed and committable; invalid rows are blocked with row-level errors.",
        },
        {
            "ID": "FR-03",
            "Requirement": "Validate dates, currencies, numeric ranges, duplicates, and referential keys.",
            "Acceptance": "Malformed file produces downloadable error report with row number, field, issue, value, and severity.",
        },
        {
            "ID": "FR-04",
            "Requirement": "Run deterministic liquidity projection over configurable horizon.",
            "Acceptance": "Projection uses documented formula and returns day-by-day structured output.",
        },
        {
            "ID": "FR-05",
            "Requirement": "Support scenario controls for receipt delay, payroll delay, supplier delay, minimum cash, financing buffer, and financing rate.",
            "Acceptance": "Changing controls updates projection and explanation immediately.",
        },
        {
            "ID": "FR-06",
            "Requirement": "Show uncertainty band and data-quality warnings.",
            "Acceptance": "Each projected day includes lower/upper confidence values and warning list.",
        },
        {
            "ID": "FR-07",
            "Requirement": "Provide role-based permissions for administrator, analyst, reviewer, and read-only stakeholder.",
            "Acceptance": "Restricted actions are disabled or rejected for unauthorized roles.",
        },
        {
            "ID": "FR-08",
            "Requirement": "Support forecast lifecycle: draft, validated, approved, rejected, corrected, archived.",
            "Acceptance": "Only allowed transitions are accepted and audited.",
        },
        {
            "ID": "FR-09",
            "Requirement": "Allow reviewer override with mandatory reason.",
            "Acceptance": "Override changes output, preserves original explanation, and writes audit event.",
        },
        {
            "ID": "FR-10",
            "Requirement": "Export forecast CSV, scenario JSON, and management report Markdown.",
            "Acceptance": "Downloads contain the currently displayed projection and scenario.",
        },
        {
            "ID": "FR-11",
            "Requirement": "Generate a live demo MP4 inside the app.",
            "Acceptance": "App displays and allows download of an MP4 generated from synthetic demo forecast data.",
        },
        {
            "ID": "FR-12",
            "Requirement": "Provide reproducible baseline analytics evaluation.",
            "Acceptance": "ML evaluation uses chronological holdout, reports MAE/RMSE, and compares to naive baseline.",
        },
        {
            "ID": "FR-13",
            "Requirement": "Maintain audit trail for material actions.",
            "Acceptance": "Import commits, forecast saves, approvals, overrides, simulation saves, and shocks appear in audit log.",
        },
        {
            "ID": "FR-14",
            "Requirement": "Preserve original uploaded files and checksums.",
            "Acceptance": "Uploaded file is stored outside relational rows and metadata references checksum and path.",
        },
        {
            "ID": "FR-15",
            "Requirement": "Run automated self-tests from the UI.",
            "Acceptance": "Self-test panel reports pass/fail for validation, calculation, reproducibility, import errors, and audit persistence.",
        },
        {
            "ID": "FR-16",
            "Requirement": "Provide live simulation inside the app for testing.",
            "Acceptance": "Simulation auto-starts, supports play/pause/step/reset, and advances day-by-day with actual vs forecast comparison.",
        },
        {
            "ID": "FR-17",
            "Requirement": "Support shock injection and stress presets.",
            "Acceptance": "Users can inject receipt shortfall, payroll surge, supplier acceleration, tax shock, or financing injection and observe effects.",
        },
        {
            "ID": "FR-18",
            "Requirement": "Provide automated simulation test suite.",
            "Acceptance": "Suite verifies reproducibility, row count, NaN absence, stress alerting, financing behavior, and event logging.",
        },
    ]
)

WORKFLOW_DIAGRAM = """
Actors and workflow

[Analyst / Operator]
   |
   | 1. Upload CSV/XLSX or use synthetic demo data
   v
[Import Preview]
   |
   | 2. Validate schema, types, dates, currencies, duplicates, ranges
   |---- Exception: missing / contradictory / late / out-of-range data
   |       |
   |       v
   |   [Error report download] -> correct source -> re-import
   v
[Validated records]
   |
   | 3. Draft liquidity forecast
   v
[Deterministic domain engine]
   |
   | 4. Apply scenario controls:
   |       - receipt delay
   |       - payroll delay
   |       - supplier delay
   |       - minimum cash
   |       - financing buffer
   |       - financing rate
   v
[Explainable projection]
   |
   | 5. Run live simulation test harness
   |       - seeded stochastic cash events
   |       - play / pause / step / reset
   |       - inject shocks
   |       - compare actual simulated balance vs forecast
   v
[Simulation events, alerts, variance metrics]
   |
   | 6. Reviewer examines contributions, uncertainty, data quality, simulation stress
   |---- Reject -> return to analyst for correction
   |---- Override -> mandatory reason + audit event
   |---- Approve -> status becomes approved
   v
[Approved forecast]
   |
   | 7. Export CSV / JSON / Markdown report
   v
[Read-only stakeholder]

Administrator:
   - manages reference data, roles, retention, backup/restore, audit export
"""

BACKLOG_DF = pd.DataFrame(
    [
        {"Tier": "MVP", "Priority": "P0", "Item": "Synthetic demo data loader", "Rationale": "Ensures immediate runnable demo without blank dashboard."},
        {"Tier": "MVP", "Priority": "P0", "Item": "CSV/XLSX import validation", "Rationale": "Core data governance and exception handling."},
        {"Tier": "MVP", "Priority": "P0", "Item": "Deterministic liquidity engine", "Rationale": "Explainable baseline before ML."},
        {"Tier": "MVP", "Priority": "P0", "Item": "Scenario controls", "Rationale": "Meets product mission for delayed receipts, payroll, suppliers, financing."},
        {"Tier": "MVP", "Priority": "P0", "Item": "Live simulation test harness", "Rationale": "Allows day-by-day testing of liquidity behavior and forecast variance."},
        {"Tier": "MVP", "Priority": "P0", "Item": "Shock injection and stress presets", "Rationale": "Tests delayed receipts, payroll surges, supplier acceleration, tax shocks, financing needs."},
        {"Tier": "MVP", "Priority": "P0", "Item": "Live MP4 demo generation", "Rationale": "Provides immediately shareable working demo."},
        {"Tier": "MVP", "Priority": "P1", "Item": "Role-based permissions", "Rationale": "Supports analyst/reviewer governance."},
        {"Tier": "MVP", "Priority": "P1", "Item": "Audit log", "Rationale": "Traceability for approvals, overrides, simulations, and shocks."},
        {"Tier": "MVP", "Priority": "P1", "Item": "Export CSV/JSON/Markdown", "Rationale": "Operational handoff to stakeholders."},
        {"Tier": "MVP", "Priority": "P1", "Item": "Automated self-tests", "Rationale": "Reproducible quality gate."},
        {"Tier": "MVP", "Priority": "P1", "Item": "Automated simulation test suite", "Rationale": "Verifies live simulation behavior and stress response."},
        {"Tier": "Post-MVP", "Priority": "P2", "Item": "SQLite to PostgreSQL migration", "Rationale": "Multi-user team deployment."},
        {"Tier": "Post-MVP", "Priority": "P2", "Item": "FastAPI service layer", "Rationale": "Programmatic access and integrations."},
        {"Tier": "Post-MVP", "Priority": "P2", "Item": "Constrained financing optimizer", "Rationale": "Recommend minimum-cost financing schedule."},
        {"Tier": "Post-MVP", "Priority": "P2", "Item": "Anomaly detection on cash movements", "Rationale": "Flag unusual receipts/payments before forecasting."},
        {"Tier": "Enterprise", "Priority": "P3", "Item": "SSO / directory integration", "Rationale": "Enterprise identity governance."},
        {"Tier": "Enterprise", "Priority": "P3", "Item": "Immutable audit export", "Rationale": "Regulatory review and evidence packaging."},
        {"Tier": "Enterprise", "Priority": "P3", "Item": "Tenant isolation and retention policies", "Rationale": "Multi-client financial data governance."},
        {"Tier": "Enterprise", "Priority": "P3", "Item": "On-premises deployment runbook", "Rationale": "Customer-managed infrastructure."},
    ]
)

ACCEPTANCE_DF = pd.DataFrame(
    [
        {"Criterion": "Sample import", "Pass condition": "Documented CSV/XLSX sample imports with row count confirmation.", "Fail condition": "Import crashes, silently drops rows, or lacks provenance."},
        {"Criterion": "Malformed row rejection", "Pass condition": "Bad date, bad currency, duplicate key, or out-of-range amount produces row-level error report.", "Fail condition": "Invalid row is committed or error message is not actionable."},
        {"Criterion": "Reproducible calculation", "Pass condition": "Same inputs and scenario version produce identical forecast values.", "Fail condition": "Results vary without documented version change."},
        {"Criterion": "Traceable explanation", "Pass condition": "Each day shows formula, inputs, contributions, uncertainty, data quality, and reviewer status.", "Fail condition": "Output chart exists but cannot be explained."},
        {"Criterion": "Report export", "Pass condition": "CSV, JSON, and Markdown downloads reflect current projection.", "Fail condition": "Export is empty, stale, or missing scenario metadata."},
        {"Criterion": "Live demo video", "Pass condition": "App generates and displays downloadable MP4 from synthetic demo data.", "Fail condition": "No video is produced or dashboard is blank."},
        {"Criterion": "Live simulation start", "Pass condition": "App automatically starts day-by-day simulation on launch and shows actual vs forecast.", "Fail condition": "Dashboard is static or simulation requires manual data setup."},
        {"Criterion": "Simulation reproducibility", "Pass condition": "Same seed and config produce identical simulated history.", "Fail condition": "Seeded run varies without documented change."},
        {"Criterion": "Shock injection", "Pass condition": "Injected shocks alter future receipts/payments/financing and are recorded in events/audit.", "Fail condition": "Shock has no observable effect or is not auditable."},
        {"Criterion": "Simulation test suite", "Pass condition": "Automated simulation tests pass for reproducibility, row count, NaN absence, stress alerting, and event logging.", "Fail condition": "Tests fail or are skipped without explanation."},
        {"Criterion": "Role enforcement", "Pass condition": "Read-only cannot import/approve/override/simulate-control; reviewer can approve; analyst can draft/simulate.", "Fail condition": "UI-only restriction with no workflow-side enforcement."},
        {"Criterion": "Audit trail", "Pass condition": "Material actions write actor, action, entity, before/after, reason, timestamp.", "Fail condition": "Approvals, overrides, shocks, or simulations are not traceable."},
        {"Criterion": "Automated tests", "Pass condition": "Self-test panel returns PASS for core validation, calculation, reproducibility, import, audit.", "Fail condition": "Tests fail or are skipped without explanation."},
    ]
)

RISK_DF = pd.DataFrame(
    [
        {"Risk": "Domain validity", "Likelihood": "Medium", "Impact": "High", "Mitigation": "Use deterministic rules first; require qualified finance review; avoid autonomous decisions."},
        {"Risk": "Data quality", "Likelihood": "High", "Impact": "High", "Mitigation": "Row-level validation, provenance, checksums, exception reports, retention of original files."},
        {"Risk": "Model overclaim", "Likelihood": "Medium", "Impact": "High", "Mitigation": "Synthetic-data disclaimer; chronological holdout; baseline comparison; no real-world performance claim without permissioned data."},
        {"Risk": "Simulation realism", "Likelihood": "High", "Impact": "Medium", "Mitigation": "Label synthetic stochastic events clearly; use seeds; compare to deterministic baseline; require domain review before real-data use."},
        {"Risk": "Security", "Likelihood": "Medium", "Impact": "High", "Mitigation": "Role gates, audit events, no secrets in source, upload validation, local SQLite isolation, PostgreSQL path for team deployment."},
        {"Risk": "Adoption", "Likelihood": "Medium", "Impact": "Medium", "Mitigation": "Solve one narrow workflow end-to-end; export familiar CSV/Markdown reports; show explainable contributions."},
        {"Risk": "Operational failure", "Likelihood": "Low", "Impact": "High", "Mitigation": "Self-tests, simulation tests, backup/restore instructions, reproducible demo reset, pinned dependency list."},
        {"Risk": "Regulatory misrepresentation", "Likelihood": "Low", "Impact": "High", "Mitigation": "Label as decision support; require human approval; do not promise returns or regulated advice."},
    ]
)

DATA_DICTIONARY_DF = pd.DataFrame(
    [
        {"Field": "organizations.id", "Type": "TEXT UUID", "Unit": "identifier", "Meaning": "Stable organization identifier.", "Allowed": "UUID string", "Source": "system", "Sensitivity": "internal", "Validation": "primary key"},
        {"Field": "accounts.opening_balance", "Type": "REAL", "Unit": "organization currency", "Meaning": "Starting cash balance for projection.", "Allowed": "finite number", "Source": "treasury/account system", "Sensitivity": "confidential", "Validation": "not null"},
        {"Field": "transactions.txn_date", "Type": "TEXT ISO date", "Unit": "date", "Meaning": "Business date of cash movement.", "Allowed": "YYYY-MM-DD", "Source": "bank/ledger", "Sensitivity": "confidential", "Validation": "parseable date"},
        {"Field": "transactions.amount", "Type": "REAL", "Unit": "currency", "Meaning": "Signed cash movement; positive receipt, negative payment.", "Allowed": "-1e9 to 1e9", "Source": "bank/ledger", "Sensitivity": "confidential", "Validation": "numeric finite"},
        {"Field": "transactions.currency", "Type": "TEXT", "Unit": "ISO currency", "Meaning": "Transaction currency.", "Allowed": "USD, EUR, GBP", "Source": "ledger", "Sensitivity": "internal", "Validation": "allowed catalog"},
        {"Field": "invoices.due_date", "Type": "TEXT ISO date", "Unit": "date", "Meaning": "Contractual due date before scenario delay.", "Allowed": "YYYY-MM-DD", "Source": "AR ledger", "Sensitivity": "confidential", "Validation": "parseable date"},
        {"Field": "invoices.amount", "Type": "REAL", "Unit": "currency", "Meaning": "Invoice receivable amount.", "Allowed": "> 0", "Source": "AR ledger", "Sensitivity": "confidential", "Validation": "positive numeric"},
        {"Field": "invoices.collection_probability", "Type": "REAL", "Unit": "probability", "Meaning": "Expected likelihood of collection.", "Allowed": "0.0 to 1.0", "Source": "credit policy/manual", "Sensitivity": "internal", "Validation": "bounded probability"},
        {"Field": "payroll_runs.gross_amount", "Type": "REAL", "Unit": "currency", "Meaning": "Gross payroll outflow.", "Allowed": "> 0", "Source": "payroll system", "Sensitivity": "confidential", "Validation": "positive numeric"},
        {"Field": "supplier_obligations.amount", "Type": "REAL", "Unit": "currency", "Meaning": "Supplier payment outflow.", "Allowed": "> 0", "Source": "AP ledger", "Sensitivity": "confidential", "Validation": "positive numeric"},
        {"Field": "financing_facilities.limit_amount", "Type": "REAL", "Unit": "currency", "Meaning": "Maximum facility capacity.", "Allowed": ">= 0", "Source": "treasury/loan agreement", "Sensitivity": "confidential", "Validation": "non-negative"},
        {"Field": "financing_facilities.available_amount", "Type": "REAL", "Unit": "currency", "Meaning": "Undrawn available capacity.", "Allowed": ">= 0", "Source": "treasury", "Sensitivity": "confidential", "Validation": "non-negative"},
        {"Field": "financing_facilities.outstanding_amount", "Type": "REAL", "Unit": "currency", "Meaning": "Existing drawn balance.", "Allowed": ">= 0", "Source": "treasury", "Sensitivity": "confidential", "Validation": "non-negative"},
        {"Field": "financing_facilities.interest_rate", "Type": "REAL", "Unit": "decimal APR", "Meaning": "Annual interest rate assumption.", "Allowed": "0.0 to 1.0", "Source": "facility agreement", "Sensitivity": "internal", "Validation": "bounded rate"},
        {"Field": "forecast_runs.scenario_json", "Type": "TEXT JSON", "Unit": "scenario", "Meaning": "Serialized scenario controls.", "Allowed": "valid JSON", "Source": "engine", "Sensitivity": "internal", "Validation": "deserializable"},
        {"Field": "forecast_lines.closing_balance", "Type": "REAL", "Unit": "currency", "Meaning": "Projected end-of-day cash balance.", "Allowed": "finite number", "Source": "deterministic engine", "Sensitivity": "confidential", "Validation": "derived"},
        {"Field": "forecast_lines.confidence_lower", "Type": "REAL", "Unit": "currency", "Meaning": "Lower uncertainty bound.", "Allowed": "<= upper", "Source": "engine heuristic", "Sensitivity": "internal", "Validation": "derived"},
        {"Field": "forecast_lines.confidence_upper", "Type": "REAL", "Unit": "currency", "Meaning": "Upper uncertainty bound.", "Allowed": ">= lower", "Source": "engine heuristic", "Sensitivity": "internal", "Validation": "derived"},
        {"Field": "simulation_runs.seed", "Type": "INTEGER", "Unit": "random seed", "Meaning": "Seed used for reproducible stochastic simulation.", "Allowed": "integer", "Source": "simulation harness", "Sensitivity": "internal", "Validation": "not null"},
        {"Field": "simulation_runs.config_json", "Type": "TEXT JSON", "Unit": "config", "Meaning": "Simulation speed, stochastic flag, auto-finance flag.", "Allowed": "valid JSON", "Source": "simulation harness", "Sensitivity": "internal", "Validation": "deserializable"},
        {"Field": "simulation_events.event_type", "Type": "TEXT", "Unit": "event", "Meaning": "Receipt, payroll, supplier, shock, financing draw, repayment.", "Allowed": "controlled vocabulary", "Source": "simulation harness", "Sensitivity": "internal", "Validation": "not null"},
        {"Field": "simulation_events.balance_after", "Type": "REAL", "Unit": "currency", "Meaning": "Cash balance after event.", "Allowed": "finite number", "Source": "simulation harness", "Sensitivity": "confidential", "Validation": "derived"},
        {"Field": "audit_events.action", "Type": "TEXT", "Unit": "event", "Meaning": "Material user or system action.", "Allowed": "import_commit, forecast_save, forecast_approved, override, simulation_shock, simulation_save, etc.", "Source": "application", "Sensitivity": "internal", "Validation": "not null"},
        {"Field": "import_batches.file_checksum", "Type": "TEXT SHA256", "Unit": "hash", "Meaning": "Checksum of preserved original file.", "Allowed": "64 hex chars", "Source": "upload pipeline", "Sensitivity": "internal", "Validation": "sha256"},
    ]
)

ERD_TEXT = """
Normalized entity relationships (MVP)

organizations 1 ── * users
organizations 1 ── * source_systems
organizations 1 ── * accounts
organizations 1 ── * transactions
organizations 1 ── * invoices
organizations 1 ── * payroll_runs
organizations 1 ── * supplier_obligations
organizations 1 ── * financing_facilities
organizations 1 ── * forecast_runs
organizations 1 ── * simulation_runs

source_systems 1 ── * transactions
source_systems 1 ── * invoices
source_systems 1 ── * payroll_runs
source_systems 1 ── * supplier_obligations

accounts 1 ── * transactions

forecast_runs 1 ── * forecast_lines
simulation_runs 1 ── * simulation_events

import_batches 1 ── * import_errors
import_batches 1 ── * transactions/invoices through provenance JSON

audit_events polymorphic entity_type/entity_id
reference_data keyed catalog versions for currencies, categories, roles
"""


def build_default_scenario() -> Scenario:
    return Scenario(
        horizon_days=90,
        receipt_delay_days=3,
        receipt_confidence=0.92,
        payroll_delay_days=0,
        supplier_delay_days=5,
        minimum_cash=50000.0,
        financing_buffer=10000.0,
        financing_interest_rate=0.085,
        seasonality_factor=1.0,
    )


def build_context() -> Dict[str, Any]:
    ensure_demo_data()
    org_name = load_org_name()
    opening_balance = load_opening_balance()
    invoices, payroll, suppliers, facilities, transactions = load_schedules()
    default_scenario = build_default_scenario()

    default_lines = project_cashflow(
        opening_balance=opening_balance,
        invoices=invoices,
        payroll=payroll,
        suppliers=suppliers,
        facilities=facilities,
        scenario=default_scenario,
    )

    quality_score, quality_issues = compute_data_quality(invoices, payroll, suppliers, facilities)

    return {
        "org_name": org_name,
        "opening_balance": opening_balance,
        "invoices": invoices,
        "payroll": payroll,
        "suppliers": suppliers,
        "facilities": facilities,
        "transactions": transactions,
        "default_scenario": default_scenario,
        "default_lines": default_lines,
        "data_quality_score": quality_score,
        "data_quality_issues": quality_issues,
    }


def ensure_session_state() -> None:
    st.session_state.setdefault("role", "analyst")
    st.session_state.setdefault("scenario", build_default_scenario())
    st.session_state.setdefault("overrides", {})
    st.session_state.setdefault("demo_video_bytes", None)
    st.session_state.setdefault("demo_video_error", None)
    st.session_state.setdefault("ml_result", None)
    st.session_state.setdefault("self_test_results", None)
    st.session_state.setdefault("sim_test_results", None)
    st.session_state.setdefault("sim_error", None)
    st.session_state.setdefault("current_lines", None)
    st.session_state.setdefault("current_run_id", None)
    st.session_state.setdefault("current_run_status", None)

    if "ctx" not in st.session_state or st.session_state.ctx is None:
        st.session_state.ctx = build_context()

    ctx = st.session_state.ctx

    st.session_state.setdefault("data_quality_score", ctx.get("data_quality_score", 0.0))
    st.session_state.setdefault("data_quality_issues", ctx.get("data_quality_issues", []))

    if "sim_state" not in st.session_state or st.session_state.sim_state is None:
        st.session_state.sim_state = rebuild_simulation_from_context(
            ctx=ctx,
            scenario=st.session_state.scenario,
            seed=normalize_seed(DEFAULT_SEED),
            config=default_sim_config(),
            shocks=[],
        )

    if st.session_state.current_lines is None:
        st.session_state.current_lines = ctx["default_lines"]

    if st.session_state.demo_video_bytes is None and st.session_state.demo_video_error is None:
        with st.spinner("Generating live demo MP4 from synthetic data..."):
            video_bytes, video_error = render_demo_video_bytes(
                ctx["default_lines"],
                st.session_state.scenario,
                ctx["org_name"],
            )
        st.session_state.demo_video_bytes = video_bytes
        st.session_state.demo_video_error = video_error
        if video_bytes:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "live_demo.mp4").write_bytes(video_bytes)


def render_video_block() -> None:
    if st.session_state.get("demo_video_bytes"):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.video(
                st.session_state["demo_video_bytes"],
                format="video/mp4",
                loop=True,
            )
        with c2:
            st.download_button(
                "Download live demo MP4",
                st.session_state["demo_video_bytes"],
                file_name="liquidity_live_demo.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        st.warning(st.session_state.get("demo_video_error") or "Demo video was not generated.")


def render_live_simulation(ctx: Dict[str, Any]) -> None:
    state = st.session_state.sim_state
    scenario = st.session_state.scenario
    sig = scenario.signature()

    state.setdefault("shocks", [])
    state.setdefault("config", default_sim_config())
    state.setdefault("pending_seed", normalize_seed(state.get("seed", DEFAULT_SEED)))
    state.setdefault("schedule_stochastic_applied", bool(state.get("config", {}).get("stochastic", True)))

    if not can("simulate"):
        state["playing"] = False

    st.subheader("Live simulation test harness")
    st.caption("Synthetic stochastic cash events. Decision support only; not regulated financial advice.")

    render_video_block()

    if st.session_state.get("sim_error"):
        st.error(f"Simulation error: {st.session_state.sim_error}")
        st.session_state.sim_error = None

    if state.get("scenario_signature") != sig:
        state["playing"] = False
        st.warning("Scenario has changed since simulation was built. Rebuild to apply current scenario.")
        if st.button("Rebuild simulation with current scenario", use_container_width=True):
            st.session_state.sim_state = rebuild_simulation_from_context(
                ctx=ctx,
                scenario=scenario,
                seed=int(state.get("pending_seed", state.get("seed", DEFAULT_SEED))),
                config=state.get("config", default_sim_config()),
                shocks=state.get("shocks", []),
            )
            st.rerun()
        return

    hist = pd.DataFrame(state.get("history", []))
    metrics = state.get("metrics", {})
    latest = hist.iloc[-1] if not hist.empty else None

    latest_balance = safe_float(latest.get("closing_balance"), np.nan) if latest is not None else np.nan
    latest_forecast = safe_float(latest.get("forecast_closing_balance"), np.nan) if latest is not None else np.nan
    latest_variance = safe_float(latest.get("variance"), np.nan) if latest is not None else np.nan

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Status", "Playing" if state.get("playing") else "Paused")
    m2.metric("Day", f"{state.get('current_day', 0) - 1} / {scenario.horizon_days}")
    m3.metric("Actual balance", f"{latest_balance:,.0f}" if np.isfinite(latest_balance) else "—")
    m4.metric("Forecast balance", f"{latest_forecast:,.0f}" if np.isfinite(latest_forecast) else "—")
    m5.metric("Variance", f"{latest_variance:,.0f}" if np.isfinite(latest_variance) else "—")
    m6.metric("Financing draws", f"{float(metrics.get('total_financing_draws', 0.0)):,.0f}")
    m7.metric("Alerts", f"{int(metrics.get('alert_count', 0))}")

    with st.expander("Simulation settings", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        speed_ms = s1.slider(
            "Playback speed ms",
            min_value=50,
            max_value=1000,
            step=50,
            value=int(state.get("config", {}).get("speed_ms", 250)),
            key="sim_speed_ms",
            disabled=not can("configure"),
        )
        seed_val = s2.number_input(
            "Random seed",
            min_value=0,
            max_value=MAX_SEED,
            step=1,
            value=normalize_seed(state.get("pending_seed", state.get("seed", DEFAULT_SEED))),
            key="sim_seed",
            disabled=not can("configure"),
        )
        stochastic = s3.checkbox(
            "Stochastic events",
            value=bool(state.get("config", {}).get("stochastic", True)),
            key="sim_stochastic",
            disabled=not can("configure"),
        )
        auto_finance = s4.checkbox(
            "Auto financing",
            value=bool(state.get("config", {}).get("auto_finance", True)),
            key="sim_auto_finance",
            disabled=not can("configure"),
        )

        state.setdefault("config", default_sim_config())
        state["config"]["speed_ms"] = int(speed_ms)
        state["config"]["stochastic"] = bool(stochastic)
        state["config"]["auto_finance"] = bool(auto_finance)
        state["pending_seed"] = normalize_seed(seed_val)

        if bool(stochastic) != bool(state.get("schedule_stochastic_applied", stochastic)):
            st.info("Stochastic setting applies after rebuild or reset.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    if c1.button("▶ Play", use_container_width=True, disabled=not can("simulate")):
        if state["current_day"] > scenario.horizon_days:
            st.session_state.sim_state = rebuild_simulation_from_context(
                ctx=ctx,
                scenario=scenario,
                seed=int(state.get("pending_seed", state.get("seed", DEFAULT_SEED))),
                config=state.get("config", default_sim_config()),
                shocks=state.get("shocks", []),
            )
            st.session_state.sim_state["playing"] = True
        else:
            state["playing"] = True
        st.rerun()

    if c2.button("⏸ Pause", use_container_width=True):
        state["playing"] = False
        st.rerun()

    if c3.button("Step", use_container_width=True, disabled=not can("simulate") or state.get("playing", False)):
        if state["current_day"] <= scenario.horizon_days:
            step_simulation(state, scenario, state.get("config", default_sim_config()))
        st.rerun()

    if c4.button("Reset", use_container_width=True, disabled=not can("simulate")):
        st.session_state.sim_state = rebuild_simulation_from_context(
            ctx=ctx,
            scenario=scenario,
            seed=int(state.get("pending_seed", state.get("seed", DEFAULT_SEED))),
            config=state.get("config", default_sim_config()),
            shocks=[],
        )
        st.rerun()

    if c5.button("Run 7 days", use_container_width=True, disabled=not can("simulate")):
        state["playing"] = False
        for _ in range(7):
            if state["current_day"] <= scenario.horizon_days:
                step_simulation(state, scenario, state.get("config", default_sim_config()))
        st.rerun()

    if c6.button("Run to horizon", use_container_width=True, disabled=not can("simulate")):
        state["playing"] = False
        guard = 0
        while state["current_day"] <= scenario.horizon_days and guard <= scenario.horizon_days + 5:
            step_simulation(state, scenario, state.get("config", default_sim_config()))
            guard += 1
        st.rerun()

    with st.expander("Inject test shocks", expanded=False):
        shock_type = st.selectbox(
            "Shock type",
            [
                "receipt_shortfall",
                "payroll_surge",
                "supplier_acceleration",
                "unexpected_tax",
                "financing_injection",
            ],
            key="shock_type_select",
        )

        default_start = min(max(0, int(state.get("current_day", 0))), int(scenario.horizon_days))

        with st.form("shock_form"):
            f1, f2, f3 = st.columns(3)
            start_day = f1.number_input(
                "Start day",
                min_value=0,
                max_value=int(scenario.horizon_days),
                value=int(default_start),
                step=1,
            )
            duration = f2.number_input(
                "Duration days",
                min_value=1,
                max_value=max(1, int(scenario.horizon_days)),
                value=1,
                step=1,
            )
            if shock_type == "receipt_shortfall":
                magnitude = f3.slider(
                    "Shortfall fraction",
                    min_value=0.0,
                    max_value=0.90,
                    step=0.05,
                    value=0.25,
                )
            else:
                magnitude = f3.number_input(
                    "Amount",
                    min_value=0.0,
                    max_value=1000000.0,
                    step=1000.0,
                    value=10000.0,
                    format="%.0f",
                )

            label = st.text_input("Shock label", placeholder="Optional reviewer-facing label")
            submitted = st.form_submit_button("Inject shock", disabled=not can("inject_shock"))

        if submitted:
            if start_day < state.get("current_day", 0):
                start_day = min(max(0, int(state.get("current_day", 0))), int(scenario.horizon_days))
            shock = {
                "id": str(uuid.uuid4()),
                "type": shock_type,
                "start_day": int(start_day),
                "duration": int(duration),
                "magnitude": float(magnitude),
                "label": label.strip() or shock_type,
            }
            state.setdefault("shocks", []).append(shock)
            record_audit(
                action="simulation_shock",
                entity_type="simulation_run",
                entity_id=state.get("run_id"),
                after=shock,
                reason=label.strip() or "Shock injected in live simulation.",
            )
            st.success("Shock injected. It will affect future simulated days.")

        p1, p2, p3 = st.columns(3)
        if p1.button("Apply combined stress and restart", use_container_width=True, disabled=not can("inject_shock")):
            shocks = combined_stress_shocks(0, scenario.horizon_days)
            st.session_state.sim_state = rebuild_simulation_from_context(
                ctx=ctx,
                scenario=scenario,
                seed=int(state.get("pending_seed", state.get("seed", DEFAULT_SEED))),
                config=state.get("config", default_sim_config()),
                shocks=shocks,
            )
            st.session_state.sim_state["playing"] = True
            record_audit(
                action="simulation_stress_preset",
                entity_type="simulation_run",
                entity_id=st.session_state.sim_state.get("run_id"),
                after={"preset": "combined_stress", "shock_count": len(shocks)},
                reason="Combined stress preset applied.",
            )
            st.rerun()

        if p2.button("Clear shocks", use_container_width=True, disabled=not can("inject_shock")):
            state["shocks"] = []
            st.success("Shocks cleared. Future days will use base schedule only.")

        if p3.button("Save simulation run", use_container_width=True, disabled=not can("save_simulation")):
            run_id = save_simulation_run(
                state=state,
                scenario=scenario,
                actor_role=st.session_state.role,
                notes="Saved from live simulation harness.",
            )
            st.success(f"Simulation run saved: {run_id}")

    if not hist.empty:
        fig = go.Figure()
        fc = state.get("forecast_lines")
        fc_plot = None
        if fc is not None and not fc.empty:
            fc_plot = fc.copy()
            fc_plot["date"] = pd.to_datetime(fc_plot["date"], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=fc_plot["date"],
                    y=fc_plot["closing_balance"],
                    mode="lines",
                    line=dict(color="#4C78A8", width=2, dash="dash"),
                    name="Deterministic forecast",
                )
            )

        hist_plot = hist.copy()
        hist_plot["date"] = pd.to_datetime(hist_plot["date"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=hist_plot["date"],
                y=hist_plot["closing_balance"],
                mode="lines+markers",
                line=dict(color="#54A24B", width=3),
                name="Live simulated actual",
            )
        )

        date_parts = [hist_plot["date"]]
        if fc_plot is not None:
            date_parts.append(fc_plot["date"])
        all_dates = pd.concat(date_parts).dropna()

        y_parts = [hist_plot["closing_balance"]]
        if fc_plot is not None:
            y_parts.append(fc_plot["closing_balance"])
        y_parts.append(pd.Series([scenario.minimum_cash], dtype=float))
        all_y = pd.concat(y_parts).dropna().astype(float)

        if not all_dates.empty and not all_y.empty:
            x_min = all_dates.min()
            x_max = all_dates.max()
            y_min = float(all_y.min())
            y_max = float(all_y.max())

            if math.isclose(y_min, y_max, rel_tol=1e-12, abs_tol=1.0):
                y_min -= 1000.0
                y_max += 1000.0

            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[scenario.minimum_cash, scenario.minimum_cash],
                    mode="lines",
                    line=dict(color="#E45756", dash="dash", width=2),
                    name="Minimum cash",
                    hoverinfo="y",
                )
            )

            for shock in state.get("shocks", [])[-10:]:
                sd = int(shock.get("start_day", 0))
                if sd <= state.get("current_day", 0) - 1:
                    shock_x = pd.Timestamp(state["as_of_date"]) + pd.Timedelta(days=sd)
                    label = safe_text(shock.get("label"), shock.get("type"))
                    fig.add_trace(
                        go.Scatter(
                            x=[shock_x, shock_x],
                            y=[y_min, y_max],
                            mode="lines",
                            line=dict(color="#F58518", dash="dot", width=2),
                            name=f"Shock: {label}",
                            hoverinfo="x+y",
                        )
                    )

        fig.update_layout(
            title="Live simulation: actual vs deterministic forecast",
            xaxis_title="Date",
            yaxis_title="Cash balance",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=60, b=10),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        movement = hist.melt(
            id_vars=["date"],
            value_vars=["receipts", "payroll", "supplier_payments", "financing_in", "financing_out"],
            var_name="stream",
            value_name="amount",
        )
        movement["signed_amount"] = movement.apply(
            lambda r: r["amount"] if r["stream"] in ["receipts", "financing_in"] else -r["amount"],
            axis=1,
        )
        fig2 = px.bar(
            movement,
            x="date",
            y="signed_amount",
            color="stream",
            barmode="relative",
            title="Simulated daily cash movement composition",
            labels={"signed_amount": "Signed cash impact", "date": "Date"},
            height=330,
        )
        st.plotly_chart(fig2, use_container_width=True)

        display_cols = [
            "day_index",
            "date",
            "opening_balance",
            "receipts",
            "payroll",
            "supplier_payments",
            "financing_in",
            "financing_out",
            "closing_balance",
            "forecast_closing_balance",
            "variance",
            "financing_need",
            "available_facility",
            "outstanding_facility",
        ]
        st.dataframe(hist[display_cols], use_container_width=True, height=280)
    else:
        st.info("Simulation history is empty. Press Play or Step.")

    ev = pd.DataFrame(state.get("events", []))
    if not ev.empty:
        ev = ev.sort_values(["day_index", "created_at"], ascending=[False, False])
        st.markdown("#### Event feed")
        st.dataframe(
            ev[["day_index", "date", "event_type", "description", "amount", "balance_after", "severity"]],
            use_container_width=True,
            height=260,
        )

    alerts = pd.DataFrame(state.get("alerts", []))
    if not alerts.empty:
        st.markdown("#### Alerts")
        st.dataframe(alerts, use_container_width=True, height=220)

    st.markdown("#### Simulation metrics")
    st.json(metrics)

    if can("run_tests"):
        if st.button("Run automated simulation test suite"):
            with st.spinner("Running seeded simulation tests..."):
                st.session_state.sim_test_results = run_simulation_test_suite(
                    ctx=ctx,
                    scenario=scenario,
                    config=state.get("config", default_sim_config()),
                )
        results = st.session_state.get("sim_test_results")
        if results is not None:
            st.dataframe(results, use_container_width=True)
            if (results["Result"] == "PASS").all():
                st.success("All simulation tests passed.")
            else:
                st.error("One or more simulation tests failed or errored.")

    if (
        state.get("playing", False)
        and can("simulate")
        and state.get("scenario_signature") == sig
        and state.get("current_day", 0) <= scenario.horizon_days
    ):
        try:
            step_simulation(state, scenario, state.get("config", default_sim_config()))
        except Exception as exc:
            state["playing"] = False
            st.session_state.sim_error = str(exc)
        else:
            time.sleep(max(0.05, int(state.get("config", {}).get("speed_ms", 250)) / 1000.0))
            st.rerun()
    elif state.get("playing", False) and state.get("current_day", 0) > scenario.horizon_days:
        state["playing"] = False


def render_forecast_lab(ctx: Dict[str, Any]) -> None:
    st.subheader("Deterministic forecast lab")
    st.caption("Explainable short- and medium-term liquidity projection with scenario controls.")

    scenario = st.session_state.scenario

    with st.expander("Scenario controls", expanded=True):
        r1, r2, r3, r4 = st.columns(4)
        horizon = r1.number_input(
            "Horizon days",
            min_value=30,
            max_value=180,
            step=5,
            value=int(scenario.horizon_days),
            key="fl_horizon",
            disabled=not can("configure"),
        )
        receipt_delay = r2.number_input(
            "Receipt delay days",
            min_value=0,
            max_value=30,
            step=1,
            value=int(scenario.receipt_delay_days),
            key="fl_receipt_delay",
            disabled=not can("configure"),
        )
        supplier_delay = r3.number_input(
            "Supplier delay days",
            min_value=0,
            max_value=30,
            step=1,
            value=int(scenario.supplier_delay_days),
            key="fl_supplier_delay",
            disabled=not can("configure"),
        )
        payroll_delay = r4.number_input(
            "Payroll delay days",
            min_value=0,
            max_value=14,
            step=1,
            value=int(scenario.payroll_delay_days),
            key="fl_payroll_delay",
            disabled=not can("configure"),
        )

        r5, r6, r7, r8 = st.columns(4)
        minimum_cash = r5.number_input(
            "Minimum cash",
            min_value=0.0,
            max_value=1000000.0,
            step=1000.0,
            value=float(scenario.minimum_cash),
            key="fl_minimum_cash",
            disabled=not can("configure"),
            format="%.0f",
        )
        financing_buffer = r6.number_input(
            "Financing buffer",
            min_value=0.0,
            max_value=250000.0,
            step=1000.0,
            value=float(scenario.financing_buffer),
            key="fl_financing_buffer",
            disabled=not can("configure"),
            format="%.0f",
        )
        receipt_confidence = r7.slider(
            "Receipt confidence",
            min_value=0.50,
            max_value=1.00,
            step=0.01,
            value=float(scenario.receipt_confidence),
            key="fl_receipt_confidence",
            disabled=not can("configure"),
        )
        financing_rate = r8.number_input(
            "Financing APR",
            min_value=0.0,
            max_value=0.30,
            step=0.005,
            value=float(scenario.financing_interest_rate),
            key="fl_financing_rate",
            disabled=not can("configure"),
            format="%.3f",
        )

        seasonality = st.slider(
            "Seasonality factor",
            min_value=0.80,
            max_value=1.20,
            step=0.05,
            value=float(scenario.seasonality_factor),
            key="fl_seasonality",
            disabled=not can("configure"),
        )

        new_scenario = Scenario(
            horizon_days=int(horizon),
            receipt_delay_days=int(receipt_delay),
            receipt_confidence=float(receipt_confidence),
            payroll_delay_days=int(payroll_delay),
            supplier_delay_days=int(supplier_delay),
            minimum_cash=float(minimum_cash),
            financing_buffer=float(financing_buffer),
            financing_interest_rate=float(financing_rate),
            seasonality_factor=float(seasonality),
        )
        st.session_state.scenario = new_scenario
        scenario = new_scenario

    try:
        base_lines = project_cashflow(
            opening_balance=ctx["opening_balance"],
            invoices=ctx["invoices"],
            payroll=ctx["payroll"],
            suppliers=ctx["suppliers"],
            facilities=ctx["facilities"],
            scenario=scenario,
        )
    except ValueError as exc:
        st.error(f"Scenario validation failed: {exc}")
        return

    lines = apply_overrides(base_lines, st.session_state.overrides, scenario)
    st.session_state.current_lines = lines

    quality_score, quality_issues = compute_data_quality(
        ctx["invoices"],
        ctx["payroll"],
        ctx["suppliers"],
        ctx["facilities"],
    )
    st.session_state.data_quality_score = quality_score
    st.session_state.data_quality_issues = quality_issues

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Opening balance", f"{ctx['opening_balance']:,.0f}")
    m2.metric("30-day minimum", f"{lines[lines['day_index'] <= 30]['closing_balance'].min():,.0f}")
    m3.metric("Horizon ending", f"{lines['closing_balance'].iloc[-1]:,.0f}")
    m4.metric("Lowest balance", f"{lines['closing_balance'].min():,.0f}")
    m5.metric("Total financing need", f"{lines['financing_need'].sum():,.0f}")
    m6.metric("Data quality", f"{quality_score:.0%}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lines["date"],
            y=lines["confidence_upper"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            name="Upper bound",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lines["date"],
            y=lines["confidence_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            name="Uncertainty band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lines["date"],
            y=lines["closing_balance"],
            mode="lines",
            line=dict(color="#4C78A8", width=3),
            name="Projected closing balance",
        )
    )

    all_dates = pd.to_datetime(lines["date"], errors="coerce").dropna()
    if not all_dates.empty:
        x_min = all_dates.min()
        x_max = all_dates.max()
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[scenario.minimum_cash, scenario.minimum_cash],
                mode="lines",
                line=dict(color="#E45756", dash="dash", width=2),
                name="Minimum cash",
                hoverinfo="y",
            )
        )

    fig.update_layout(
        title="Deterministic liquidity projection",
        xaxis_title="Date",
        yaxis_title="Cash balance",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    movement = lines.melt(
        id_vars=["date"],
        value_vars=["receipts", "payroll", "supplier_payments", "financing_in", "financing_out"],
        var_name="stream",
        value_name="amount",
    )
    movement["signed_amount"] = movement.apply(
        lambda r: r["amount"] if r["stream"] in ["receipts", "financing_in"] else -r["amount"],
        axis=1,
    )
    fig2 = px.bar(
        movement,
        x="date",
        y="signed_amount",
        color="stream",
        barmode="relative",
        title="Daily cash movement composition",
        labels={"signed_amount": "Signed cash impact", "date": "Date"},
        height=330,
    )
    st.plotly_chart(fig2, use_container_width=True)

    display_cols = [
        "day_index",
        "date",
        "opening_balance",
        "receipts",
        "payroll",
        "supplier_payments",
        "financing_in",
        "financing_out",
        "closing_balance",
        "financing_need",
        "confidence_lower",
        "confidence_upper",
    ]
    st.dataframe(lines[display_cols], use_container_width=True, height=280)

    st.markdown("#### Day-level explanation")
    max_day = max(0, len(lines) - 1)
    selected_idx = st.slider(
        "Selected day",
        min_value=0,
        max_value=max_day,
        value=min(max_day, int(st.session_state.get("forecast_selected_day", 0))),
        key="forecast_selected_day",
    )
    selected_row = lines.iloc[selected_idx]

    try:
        explanation = json.loads(selected_row["explanation_json"])
    except Exception:
        explanation = {}
    try:
        data_quality = json.loads(selected_row["data_quality_json"])
    except Exception:
        data_quality = []

    st.json(
        {
            "date": selected_row["date"],
            "opening_balance": float(selected_row["opening_balance"]),
            "receipts": float(selected_row["receipts"]),
            "payroll": float(selected_row["payroll"]),
            "supplier_payments": float(selected_row["supplier_payments"]),
            "financing_in": float(selected_row["financing_in"]),
            "financing_out": float(selected_row["financing_out"]),
            "closing_balance": float(selected_row["closing_balance"]),
            "financing_need": float(selected_row["financing_need"]),
            "confidence_interval": [
                float(selected_row["confidence_lower"]),
                float(selected_row["confidence_upper"]),
            ],
            "explanation": explanation,
            "data_quality": data_quality,
            "global_data_quality_issues": quality_issues[:10],
        }
    )

    if can("override"):
        with st.form("forecast_override_form"):
            st.markdown("##### Governed manual override")
            o1, o2, o3 = st.columns([1, 1, 2])
            override_day = o1.number_input("Day index", min_value=0, max_value=max_day, value=selected_idx, step=1)
            override_delta = o2.number_input("Adjustment", value=0.0, step=1000.0, format="%.0f")
            override_reason = o3.text_input("Reason", placeholder="Required for audit trail")
            submitted = st.form_submit_button("Apply override")
        if submitted:
            if not override_reason.strip():
                st.error("Override reason is required.")
            else:
                st.session_state.overrides[str(int(override_day))] = {
                    "delta": float(override_delta),
                    "reason": override_reason.strip(),
                }
                record_audit(
                    action="forecast_override",
                    entity_type="forecast_line",
                    entity_id=st.session_state.get("current_run_id") or "unsaved_forecast_projection",
                    after={"day_index": int(override_day), "delta": float(override_delta), "reason": override_reason},
                    reason=override_reason,
                )
                st.success("Override applied and audited.")
                st.rerun()

    action1, action2, action3, action4 = st.columns(4)
    if action1.button("Save forecast run", disabled=not can("create_forecast"), use_container_width=True):
        run_id = save_forecast_run(
            scenario=scenario,
            lines=lines,
            opening_balance=ctx["opening_balance"],
            actor_role=st.session_state.role,
            overrides=st.session_state.overrides,
        )
        st.session_state.current_run_id = run_id
        st.session_state.current_run_status = "draft"
        st.success(f"Saved forecast run {run_id}")

    if st.session_state.current_run_id:
        if action2.button("Validate", disabled=not can("validate"), use_container_width=True):
            try:
                change_forecast_status(st.session_state.current_run_id, "validated", st.session_state.role, "Validated from forecast lab.")
                st.session_state.current_run_status = "validated"
                st.success("Forecast validated.")
            except ValueError as exc:
                st.error(str(exc))

        if action3.button("Approve", disabled=not can("approve"), use_container_width=True):
            try:
                change_forecast_status(st.session_state.current_run_id, "approved", st.session_state.role, "Approved from forecast lab.")
                st.session_state.current_run_status = "approved"
                st.success("Forecast approved.")
            except ValueError as exc:
                st.error(str(exc))

        if action4.button("Archive", disabled=not can("archive"), use_container_width=True):
            try:
                change_forecast_status(st.session_state.current_run_id, "archived", st.session_state.role, "Archived from forecast lab.")
                st.session_state.current_run_status = "archived"
                st.success("Forecast archived.")
            except ValueError as exc:
                st.error(str(exc))

        st.caption(f"Current run: {st.session_state.current_run_id} | status: {st.session_state.current_run_status}")


def render_import_tab() -> None:
    st.subheader("Data import and validation")
    st.caption("CSV and XLSX first. Original uploaded files are preserved with checksums and metadata.")

    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"], key="import_uploader")
    import_type = st.selectbox("Record type", list(IMPORT_SPECS.keys()), key="import_type")

    if uploaded is None:
        st.info("Upload a file to preview validation. Sample synthetic data is already loaded in the Live Simulation and Forecast Lab tabs.")
        return

    file_bytes = uploaded.getvalue()
    filename = uploaded.name

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    valid_df, errors_df, warnings_df = validate_import(df, import_type, check_database=True)

    st.markdown("#### Preview")
    pv1, pv2, pv3 = st.tabs(["Valid rows", "Errors", "Warnings"])
    with pv1:
        st.dataframe(valid_df.head(200), use_container_width=True)
        st.caption(f"{len(valid_df)} valid row(s).")
    with pv2:
        if errors_df.empty:
            st.success("No validation errors detected.")
        else:
            st.dataframe(errors_df, use_container_width=True)
            st.download_button(
                "Download error report",
                errors_df.to_csv(index=False).encode("utf-8"),
                file_name="import_errors.csv",
                mime="text/csv",
            )
    with pv3:
        if warnings_df.empty:
            st.info("No warnings.")
        else:
            st.dataframe(warnings_df, use_container_width=True)

    commit_disabled = (not can("import")) or valid_df.empty
    if st.button("Commit validated rows", disabled=commit_disabled):
        try:
            batch_id, count = commit_import(
                import_type=import_type,
                valid_df=valid_df,
                errors_df=errors_df,
                filename=filename,
                file_bytes=file_bytes,
                actor_role=st.session_state.role,
            )
            st.success(f"Committed {count} row(s). Batch ID: {batch_id}")
            st.session_state.ctx = None
            st.session_state.sim_state = None
            st.session_state.current_lines = None
            st.session_state.demo_video_bytes = None
            st.session_state.demo_video_error = None
            st.rerun()
        except Exception as exc:
            st.error(f"Import commit failed: {exc}")
    elif not can("import"):
        st.warning("Your role does not permit import commits.")


def render_records_tab() -> None:
    st.subheader("Operational records and lifecycle")

    entity = st.selectbox(
        "Entity",
        [
            "transactions",
            "invoices",
            "payroll_runs",
            "supplier_obligations",
            "financing_facilities",
            "forecast_runs",
            "forecast_lines",
            "simulation_runs",
            "simulation_events",
            "audit_events",
            "import_batches",
            "import_errors",
        ],
    )

    df = load_table(entity, limit=1000)
    st.dataframe(df, use_container_width=True, height=360)

    if entity == "forecast_runs" and not df.empty:
        st.markdown("#### Forecast lifecycle actions")
        run_options = df["id"].tolist()
        selected_run = st.selectbox("Forecast run", run_options, format_func=lambda x: str(x)[:8])
        row = df[df["id"] == selected_run].iloc[0]
        st.write(f"Current status: **{row['status']}**")

        a1, a2, a3, a4 = st.columns(4)
        reason = st.text_input("Action reason", key="lifecycle_reason")

        if a1.button("Validate", disabled=not can("validate")):
            try:
                change_forecast_status(selected_run, "validated", st.session_state.role, reason or "Validated from records tab.")
                st.success("Status updated.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if a2.button("Approve", disabled=not can("approve")):
            try:
                change_forecast_status(selected_run, "approved", st.session_state.role, reason or "Approved from records tab.")
                st.success("Status updated.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if a3.button("Reject", disabled=not can("approve")):
            try:
                change_forecast_status(selected_run, "rejected", st.session_state.role, reason or "Rejected from records tab.")
                st.success("Status updated.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if a4.button("Archive", disabled=not can("archive")):
            try:
                change_forecast_status(selected_run, "archived", st.session_state.role, reason or "Archived from records tab.")
                st.success("Status updated.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def render_analytics_tab(ctx: Dict[str, Any]) -> None:
    st.subheader("Baseline analytics and AI design")
    st.caption("Deterministic engine is primary. ML is evaluated only as a residual baseline on synthetic history.")

    if st.button("Run reproducible evaluation"):
        with st.spinner("Training chronological holdout baseline..."):
            st.session_state.ml_result = run_ml_evaluation(ctx["transactions"], holdout_days=30)

    result = st.session_state.get("ml_result")
    if not result:
        st.info("Run evaluation to see metrics, baseline comparison, and limitations.")
        return

    if "error" in result:
        st.error(result["error"])
        return

    st.markdown("#### Evaluation metrics")
    st.dataframe(result["metrics"], use_container_width=True)

    st.markdown("#### Holdout actual vs predicted")
    plot_long = result["plot_df"].melt(
        id_vars="date",
        value_vars=["amount", "Model prediction", "Naive lag-1"],
        var_name="series",
        value_name="net_cash",
    )
    fig = px.line(
        plot_long,
        x="date",
        y="net_cash",
        color="series",
        title="Daily net cash flow: actual vs model vs naive baseline",
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model coefficients")
    st.dataframe(result["coefficients"], use_container_width=True)

    st.markdown("#### Model card and limitations")
    st.json(result["info"])


def render_governance_tab() -> None:
    st.subheader("Product governance, requirements, and quality")

    section = st.radio(
        "Section",
        [
            "Product brief",
            "Functional requirements",
            "Workflow diagram",
            "Prioritized backlog",
            "Acceptance checklist",
            "Risk register",
            "Data dictionary",
            "ERD and lineage",
            "Audit log",
            "Software self-tests",
            "Simulation test results",
        ],
        horizontal=True,
    )

    if section == "Product brief":
        st.markdown(PRODUCT_BRIEF_MD)

    elif section == "Functional requirements":
        st.dataframe(FUNCTIONAL_REQUIREMENTS_DF, use_container_width=True)

    elif section == "Workflow diagram":
        st.code(WORKFLOW_DIAGRAM, language="text")

    elif section == "Prioritized backlog":
        st.dataframe(BACKLOG_DF, use_container_width=True)

    elif section == "Acceptance checklist":
        st.dataframe(ACCEPTANCE_DF, use_container_width=True)

    elif section == "Risk register":
        st.dataframe(RISK_DF, use_container_width=True)

    elif section == "Data dictionary":
        st.dataframe(DATA_DICTIONARY_DF, use_container_width=True)

    elif section == "ERD and lineage":
        st.code(ERD_TEXT, language="text")
        st.markdown(
            """
            Lineage design:
            - Every imported row stores batch ID, filename, checksum, row number, source system, and schema version.
            - Every forecast run stores scenario JSON, engine version, opening balance, as-of date, and status.
            - Every forecast line stores explanation JSON and data-quality JSON.
            - Every simulation run stores seed, config, scenario snapshot, ending balance, minimum balance, and status.
            - Every simulation event stores day, date, event type, amount, balance after, severity, and payload.
            - Manual overrides preserve original explanation and append override reason.
            - Audit events record actor role, action, entity, before/after snapshots, reason, and timestamp.
            """
        )

    elif section == "Audit log":
        audit_df = load_table("audit_events", limit=200)
        st.dataframe(audit_df, use_container_width=True, height=360)

    elif section == "Software self-tests":
        if st.button("Run automated software self-tests"):
            with st.spinner("Running tests..."):
                results = run_self_tests()
            st.session_state.self_test_results = results

        results = st.session_state.get("self_test_results")
        if results is not None:
            st.dataframe(results, use_container_width=True)
            if (results["Result"] == "PASS").all():
                st.success("All software self-tests passed.")
            else:
                st.error("One or more software self-tests failed or errored.")

    elif section == "Simulation test results":
        results = st.session_state.get("sim_test_results")
        if results is None:
            st.info("Run the automated simulation test suite from the Live Simulation tab.")
        else:
            st.dataframe(results, use_container_width=True)
            if (results["Result"] == "PASS").all():
                st.success("All simulation tests passed.")
            else:
                st.error("One or more simulation tests failed or errored.")


def render_reports_tab(ctx: Dict[str, Any]) -> None:
    st.subheader("Exports and management report")

    lines = st.session_state.get("current_lines")
    if lines is None or lines.empty:
        st.warning("No current projection available. Open the Forecast Lab tab first.")
        return

    scenario = st.session_state.scenario
    summary = build_management_summary(
        lines=lines,
        scenario=scenario,
        org_name=ctx["org_name"],
        opening_balance=ctx["opening_balance"],
        data_quality_score=st.session_state.get("data_quality_score", 0.0),
    )

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download forecast CSV",
        lines.to_csv(index=False).encode("utf-8"),
        file_name="liquidity_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "Download scenario JSON",
        json.dumps(asdict(scenario), indent=2).encode("utf-8"),
        file_name="liquidity_scenario.json",
        mime="application/json",
        use_container_width=True,
    )
    c3.download_button(
        "Download management report",
        summary.encode("utf-8"),
        file_name="liquidity_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    state = st.session_state.get("sim_state")
    if state and state.get("history"):
        sim_df = pd.DataFrame(state["history"])
        st.download_button(
            "Download live simulation CSV",
            sim_df.to_csv(index=False).encode("utf-8"),
            file_name="liquidity_live_simulation.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download simulation events JSON",
            json.dumps(state.get("events", []), indent=2, default=str).encode("utf-8"),
            file_name="liquidity_simulation_events.json",
            mime="application/json",
        )

    st.markdown(summary)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💧", layout="wide")

    try:
        ensure_session_state()
    except Exception as exc:
        st.error(f"Application initialization failed: {exc}")
        return

    ctx = st.session_state.ctx

    st.sidebar.title("Liquidity Intelligence")
    st.sidebar.caption("Local MVP · synthetic data · no API keys")

    role_index = ROLES.index(st.session_state.get("role", "analyst"))
    selected_role = st.sidebar.selectbox("Active role", ROLES, index=role_index)
    st.session_state.role = selected_role

    st.sidebar.markdown(f"**Permissions:** {', '.join(sorted(PERMISSIONS.get(selected_role, set())))}")

    if st.sidebar.button("Reset synthetic demo database"):
        if can("delete"):
            if DB_PATH.exists():
                DB_PATH.unlink()
            video_path = OUTPUT_DIR / "live_demo.mp4"
            if video_path.exists():
                try:
                    video_path.unlink()
                except Exception:
                    pass
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        else:
            st.sidebar.error("Only administrators can reset the demo database.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Operational notes**
        - SQLite file: `liquidity_demo.sqlite`
        - Uploads preserved in: `data/uploads`
        - Demo video saved in: `outputs/live_demo.mp4`
        - Demo data is synthetic and clearly labeled.
        - Financial outputs are decision support only.
        """
    )

    tabs = st.tabs(
        [
            "Live Simulation",
            "Forecast Lab",
            "Import Data",
            "Records",
            "Analytics & AI",
            "Governance",
            "Reports",
        ]
    )

    with tabs[0]:
        render_live_simulation(ctx)

    with tabs[1]:
        render_forecast_lab(ctx)

    with tabs[2]:
        render_import_tab()

    with tabs[3]:
        render_records_tab()

    with tabs[4]:
        render_analytics_tab(ctx)

    with tabs[5]:
        render_governance_tab()

    with tabs[6]:
        render_reports_tab(ctx)


if __name__ == "__main__":
    main()