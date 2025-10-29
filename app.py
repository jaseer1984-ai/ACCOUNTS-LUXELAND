# app.py — Streamlit Mini Accounting (Double Entry)
# ------------------------------------------------
# Features
# - Chart of Accounts (add/edit)
# - Post 2-line vouchers (Debit one account, Credit another)
# - Auto voucher numbering
# - Journal storage in SQLite (local file: accounting.db)
# - Ledger statement with opening balance & running balance
# - Trial Balance (Dr/Cr/Net) with export
# - Simple authentication placeholder (single shared password via st.secrets)
# - Ready for Streamlit Community Cloud / local run
#
# How to run locally:
#   1) pip install -r requirements.txt
#   2) streamlit run app.py
#
# Deploy on Streamlit Community Cloud:
#   - Push this file + requirements.txt to GitHub
#   - In Streamlit Cloud, set a secret APP_PASSWORD="yourpass" for basic lock
#   - NOTE: Streamlit Cloud's filesystem is ephemeral. For durable storage,
#           switch DB_URL to a hosted Postgres (Supabase, Neon, etc.).
#
# Requirements (requirements.txt):
#   streamlit>=1.37
#   pandas>=2.0
#   sqlalchemy>=2.0
#   pydantic>=2.0
#
# Optional (for Postgres):
#   psycopg2-binary>=2.9

from __future__ import annotations
import os
import datetime as dt
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, text, Integer, String, Float, Date, MetaData, Table, Column
)
from sqlalchemy.engine import Engine
from pydantic import BaseModel, Field

APP_TITLE = "📒 Mini Accounting"
DB_FILE = "accounting.db"
DEFAULT_DB_URL = f"sqlite:///{DB_FILE}"

# To use Postgres in Streamlit Cloud, set a secret DB_URL like:
# st.secrets["DB_URL"] = "postgresql+psycopg2://user:pass@host/dbname"
DB_URL = st.secrets.get("DB_URL", DEFAULT_DB_URL)
APP_PASSWORD = st.secrets.get("APP_PASSWORD")  # optional gate

# ------------------------------
# Auth (very simple shared password)
# ------------------------------

def check_auth():
    if APP_PASSWORD:
        # keep a small session flag
        if not st.session_state.get("auth_ok"):
            with st.sidebar:
                st.info("🔒 Enter app password to continue")
                pw = st.text_input("App password", type="password")
                if st.button("Unlock"):
                    if pw == APP_PASSWORD:
                        st.session_state["auth_ok"] = True
                        st.success("Unlocked")
                    else:
                        st.error("Wrong password")
        return st.session_state.get("auth_ok", False)
    return True

# ------------------------------
# DB / Schema
# ------------------------------

def get_engine() -> Engine:
    # sqlite requires special args for multi-thread in Streamlit
    if DB_URL.startswith("sqlite"):
        return create_engine(DB_URL, connect_args={"check_same_thread": False})
    return create_engine(DB_URL)

metadata = MetaData()

accounts = Table(
    "accounts",
    metadata,
    Column("code", String(32), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("type", String(16), nullable=False),  # Asset/Liability/Equity/Income/Expense
)

vouchers = Table(
    "vouchers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", Date, nullable=False),
    Column("type", String(16), nullable=False),  # Payment/Receipt/Journal/Contra
    Column("ref", String(64)),
    Column("narration", String(256)),
)

journal = Table(
    "journal",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("voucher_id", Integer, nullable=False),
    Column("date", Date, nullable=False),
    Column("vtype", String(16), nullable=False),
    Column("ref", String(64)),
    Column("acc_code", String(32), nullable=False),
    Column("acc_name", String(128), nullable=False),
    Column("dr", Float, default=0.0),
    Column("cr", Float, default=0.0),
    Column("narration", String(256)),
)

engine = get_engine()
with engine.begin() as conn:
    metadata.create_all(conn)

# ------------------------------
# Models / Helpers
# ------------------------------

class VoucherInput(BaseModel):
    vdate: dt.date
    vtype: str
    ref: str = ""
    debit_code: str
    credit_code: str
    amount: float = Field(gt=0)
    narration: str = ""

ACCOUNT_TYPES = ["Asset", "Liability", "Equity", "Income", "Expense"]
VOUCHER_TYPES = ["Payment", "Receipt", "Journal", "Contra"]

@st.cache_data(show_spinner=False)
def load_accounts_df() -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM accounts ORDER BY code"), conn)
    return df

@st.cache_data(show_spinner=False)
def load_trial_balance_df() -> pd.DataFrame:
    with engine.begin() as conn:
        # compute dr/cr by account
        q = text(
            """
            SELECT a.code, a.name, a.type,
                   COALESCE(SUM(j.dr),0) AS dr, COALESCE(SUM(j.cr),0) AS cr,
                   COALESCE(SUM(j.dr),0) - COALESCE(SUM(j.cr),0) AS net
            FROM accounts a
            LEFT JOIN journal j ON j.acc_code = a.code
            GROUP BY a.code, a.name, a.type
            ORDER BY a.code
            """
        )
        df = pd.read_sql(q, conn)
    return df

@st.cache_data(show_spinner=False)
def load_journal_df() -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM journal ORDER BY date, id"), conn, parse_dates=["date"]) 
    return df

def clear_data_caches():
    load_accounts_df.clear()
    load_trial_balance_df.clear()
    load_journal_df.clear()

# Opening balance helper: sum before from_date

def opening_balance(acc_code: str, from_date: dt.date) -> float:
    q = text(
        """
        SELECT COALESCE(SUM(dr),0) as dr, COALESCE(SUM(cr),0) as cr
        FROM journal
        WHERE acc_code = :acc AND date < :d
        """
    )
    with engine.begin() as conn:
        row = conn.execute(q, {"acc": acc_code, "d": from_date}).mappings().first()
        dr = row["dr"] if row else 0.0
        cr = row["cr"] if row else 0.0
    return float(dr) - float(cr)

# ------------------------------
# Data entry: Accounts & Vouchers
# ------------------------------

def ui_accounts():
    st.subheader("🧾 Chart of Accounts")
    st.caption("Add your accounts. Use short numeric codes (e.g., 1000 for Cash).")

    df = load_accounts_df()
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("➕ Add Account", expanded=False):
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            code = st.text_input("Code", placeholder="1000")
        with col2:
            name = st.text_input("Name", placeholder="Cash - Office")
        with col3:
            atype = st.selectbox("Type", ACCOUNT_TYPES, index=0)
        if st.button("Save Account", type="primary"):
            if not code or not name:
                st.error("Code and Name are required.")
            else:
                try:
                    with engine.begin() as conn:
                        conn.execute(accounts.insert().values(code=code.strip(), name=name.strip(), type=atype))
                    clear_data_caches()
                    st.success(f"Saved account {code} - {name}")
                except Exception as e:
                    st.error(f"Error: {e}")


def ui_voucher_entry():
    st.subheader("🧾 Post Voucher (2-line)")

    acc_df = load_accounts_df()
    if acc_df.empty:
        st.warning("Add accounts first in the 'Chart of Accounts' tab.")
        return

    # Build dropdown labels "code — name"
    choices = [f"{r.code} — {r.name}" for _, r in acc_df.iterrows()]
    code_by_label = {f"{r.code} — {r.name}": r.code for _, r in acc_df.iterrows()}

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        vdate = st.date_input("Date", value=dt.date.today())
        vtype = st.selectbox("Type", VOUCHER_TYPES, index=2)
    with c2:
        ref = st.text_input("Reference")
        amount = st.number_input("Amount", min_value=0.00, step=100.0, format="%0.2f")
    with c3:
        debit_label = st.selectbox("Debit Account", choices)
        credit_label = st.selectbox("Credit Account", choices)

    narration = st.text_area("Narration", placeholder="(optional)")

    if st.button("Post Voucher", type="primary"):
        try:
            if amount <= 0:
                st.error("Amount must be > 0"); return
            d_code = code_by_label[debit_label]
            c_code = code_by_label[credit_label]
            if d_code == c_code:
                st.error("Debit and Credit cannot be the same account"); return

            # Insert voucher + two journal lines in one transaction
            with engine.begin() as conn:
                v_res = conn.execute(
                    vouchers.insert().values(date=vdate, type=vtype, ref=ref, narration=narration)
                )
                v_id = v_res.inserted_primary_key[0]

                d_name = acc_df.loc[acc_df.code == d_code, "name"].iloc[0]
                c_name = acc_df.loc[acc_df.code == c_code, "name"].iloc[0]

                conn.execute(
                    journal.insert().values(
                        voucher_id=v_id, date=vdate, vtype=vtype, ref=ref,
                        acc_code=d_code, acc_name=d_name, dr=float(amount), cr=0.0, narration=narration
                    )
                )
                conn.execute(
                    journal.insert().values(
                        voucher_id=v_id, date=vdate, vtype=vtype, ref=ref,
                        acc_code=c_code, acc_name=c_name, dr=0.0, cr=float(amount), narration=narration
                    )
                )
            clear_data_caches()
            st.success(f"Voucher posted (#{v_id}).")
        except Exception as e:
            st.error(f"Error posting voucher: {e}")


# ------------------------------
# Ledger & Trial Balance
# ------------------------------

def ui_ledger():
    st.subheader("📑 Ledger Statement")

    acc_df = load_accounts_df()
    if acc_df.empty:
        st.warning("Add accounts first.")
        return

    labels = [f"{r.code} — {r.name}" for _, r in acc_df.iterrows()]
    code_by_label = {f"{r.code} — {r.name}": r.code for _, r in acc_df.iterrows()}

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        label = st.selectbox("Account", labels)
    with c2:
        from_date = st.date_input("From", value=dt.date(dt.date.today().year, dt.date.today().month, 1))
    with c3:
        to_date = st.date_input("To", value=dt.date.today())

    if from_date > to_date:
        st.error("From-Date cannot be after To-Date")
        return

    acc = code_by_label[label]
    op_bal = opening_balance(acc, from_date)

    jdf = load_journal_df()
    j = jdf[(jdf["acc_code"] == acc) & (jdf["date"] >= pd.Timestamp(from_date)) & (jdf["date"] <= pd.Timestamp(to_date))].copy()
    j["dr"] = j["dr"].fillna(0.0)
    j["cr"] = j["cr"].fillna(0.0)

    # Running balance
    bal = op_bal
    balances = []
    for _, row in j.iterrows():
        bal = bal + float(row["dr"]) - float(row["cr"])
        balances.append(bal)
    j["balance"] = balances

    st.metric("Opening Balance", f"{op_bal:,.2f}")

    if j.empty:
        st.info("No transactions in the selected range.")
    else:
        view = j[["date","voucher_id","vtype","ref","narration","dr","cr","balance"]]
        view["date"] = view["date"].dt.strftime("%d/%m/%Y")
        st.dataframe(view, use_container_width=True, hide_index=True)

        # Export
        csv = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"ledger_{acc}_{from_date}_{to_date}.csv")


def ui_trial_balance():
    st.subheader("🧮 Trial Balance")
    df = load_trial_balance_df()

    totals = {
        "dr": float(df["dr"].sum()),
        "cr": float(df["cr"].sum()),
        "net": float(df["net"].sum()),
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Dr", f"{totals['dr']:,.2f}")
    c2.metric("Total Cr", f"{totals['cr']:,.2f}")
    c3.metric("Dr - Cr", f"{(totals['dr']-totals['cr']):,.2f}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="trial_balance.csv")


# ------------------------------
# Journal browser
# ------------------------------

def ui_journal():
    st.subheader("📜 Journal (All Lines)")
    df = load_journal_df()
    if df.empty:
        st.info("No journal entries yet.")
        return
    view = df.copy()
    view["date"] = pd.to_datetime(view["date"]).dt.strftime("%d/%m/%Y")
    st.dataframe(view, use_container_width=True, hide_index=True)


# ------------------------------
# Main
# ------------------------------

def main():
    st.set_page_config(page_title="Mini Accounting", page_icon="📒", layout="wide")
    st.title(APP_TITLE)

    if not check_auth():
        st.stop()

    with st.sidebar:
        st.caption("Navigation")
        page = st.radio(
            "Go to",
            ["Post Voucher", "Ledger", "Trial Balance", "Journal", "Chart of Accounts"],
            index=0,
        )
        st.divider()
        st.caption("Storage")
        st.code(DB_URL, language="bash")
        if DB_URL.startswith("sqlite") and os.path.exists(DB_FILE):
            st.caption(f"SQLite file: {DB_FILE} ({os.path.getsize(DB_FILE)/1024:.1f} KB)")

    if page == "Chart of Accounts":
        ui_accounts()
    elif page == "Post Voucher":
        ui_voucher_entry()
    elif page == "Ledger":
        ui_ledger()
    elif page == "Trial Balance":
        ui_trial_balance()
    else:
        ui_journal()


if __name__ == "__main__":
    main()
