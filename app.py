# Simple Accounting Software — Masters Luxe Land LLP (INR)
# -------------------------------------------------------
# ✅ One-file Streamlit app you can push to GitHub and deploy on Streamlit Cloud.
# ✅ Features: Chart of Accounts (masters), multi-line double‑entry vouchers, Ledger,
#    Trial Balance, Profit & Loss, Balance Sheet, CSV persistence, Indian comma formatting.
# ✅ Currency: INR; all amounts shown with Indian-style comma separators.
# ✅ No sample transactions are created; the app starts clean. (You can add demo data via a checkbox.)
#
# --------------------
# How to run locally
# --------------------
# 1) pip install streamlit pandas numpy
# 2) streamlit run simple_accounting_app.py
#
# -----------------------
# Deploy on Streamlit Cloud
# -----------------------
# 1) Create a new GitHub repo and add this file at repo root.
# 2) (Optional) Add a requirements.txt with: streamlit\npandas\nnumpy
# 3) On share.streamlit.io (Streamlit Community Cloud), connect the repo and deploy.
#
# Data storage: CSV files in a local ./data folder (auto-created). If running on Streamlit Cloud,
# data resets on app restart unless you connect persistent storage (e.g., GitHub write-back
# via API, S3, or Google Drive). This simple version keeps local CSVs only.

from __future__ import annotations
import os
import io
import json
from datetime import datetime, date
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Masters Luxe Land LLP — Simple Accounting"
CURRENCY = "INR"
DATA_DIR = "data"

# ---------------------------
# Utilities: INR formatting
# ---------------------------

def format_inr(n: float | int | pd.Series | None) -> str | pd.Series:
    """Format number in Indian grouping (e.g., 12,34,56,789.50).
    Returns string for scalars, Series of strings for Series.
    """
    def _fmt(x: float | int | None) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return ""
        try:
            neg = x < 0
            x = abs(float(x))
            s = f"{x:.2f}"
            if "." in s:
                whole, frac = s.split(".")
            else:
                whole, frac = s, "00"
            # First group (last 3 digits)
            g3 = whole[-3:]
            rest = whole[:-3]
            parts = []
            while len(rest) > 2:
                parts.append(rest[-2:])
                rest = rest[:-2]
            if rest:
                parts.append(rest)
            parts = parts[::-1]
            out = ",".join([p for p in parts if p] + ([g3] if g3 else []))
            out = out if out else g3
            if neg:
                out = "-" + out
            return f"{out}.{frac}"
        except Exception:
            return str(x)

    if isinstance(n, pd.Series):
        return n.apply(_fmt)
    return _fmt(n)

# ---------------------------
# CSV persistence helpers
# ---------------------------

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def read_csv(name: str, cols: List[str]) -> pd.DataFrame:
    fp = path(name)
    if not os.path.exists(fp):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(fp)
    # Ensure all expected columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    # Order columns
    return df[cols]


def write_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(path(name), index=False)

# ---------------------------
# Data model (tabular CSVs)
# ---------------------------
# accounts.csv: [account_id, name, type, parent_id, is_active]
# vouchers.csv: [voucher_id, date, narration]
# entries.csv:  [voucher_id, line_no, account_id, debit, credit, description]

ACCOUNT_COLS = ["account_id", "name", "type", "parent_id", "is_active"]
VOUCHER_COLS = ["voucher_id", "date", "narration"]
ENTRY_COLS   = ["voucher_id", "line_no", "account_id", "debit", "credit", "description"]

ACCOUNT_TYPES = [
    "Assets",
    "Liabilities",
    "Equity",
    "Income",
    "Expenses",
]

ROOT_ACCOUNTS = [
    ("1000", "Assets", "Assets", None, True),
    ("2000", "Liabilities", "Liabilities", None, True),
    ("3000", "Equity", "Equity", None, True),
    ("4000", "Income", "Income", None, True),
    ("5000", "Expenses", "Expenses", None, True),
]

DEFAULT_LEAF_ACCOUNTS = [
    ("1100", "Bank", "Assets", "1000", True),
    ("1200", "Cash", "Assets", "1000", True),
    ("2100", "Creditors", "Liabilities", "2000", True),
    ("3100", "Capital", "Equity", "3000", True),
    ("4100", "Sales", "Income", "4000", True),
    ("5100", "Purchases", "Expenses", "5000", True),
]

# ---------------------------
# Initialization
# ---------------------------

def init_store() -> None:
    ensure_data_dir()
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)
    vouchers = read_csv("vouchers.csv", VOUCHER_COLS)
    entries  = read_csv("entries.csv", ENTRY_COLS)

    changed = False
    if accounts.empty:
        # Create only root heads by default (no sample leafs).
        acc_df = pd.DataFrame(ROOT_ACCOUNTS, columns=ACCOUNT_COLS)
        write_csv("accounts.csv", acc_df)
        changed = True
    if vouchers.empty:
        write_csv("vouchers.csv", vouchers)
        changed = True
    if entries.empty:
        write_csv("entries.csv", entries)
        changed = True

    if changed:
        st.toast("Storage initialized (./data).", icon="✅")

# ---------------------------
# Business logic
# ---------------------------

def next_voucher_id(vouchers: pd.DataFrame) -> str:
    if vouchers.empty:
        return "V000001"
    last = sorted(vouchers["voucher_id"].tolist())[-1]
    num = int(last[1:]) + 1
    return f"V{num:06d}"


def is_leaf_account(accounts: pd.DataFrame, account_id: str) -> bool:
    return not any(accounts["parent_id"].fillna("") == account_id)


def account_name(accounts: pd.DataFrame, account_id: str) -> str:
    row = accounts.loc[accounts["account_id"] == account_id]
    if row.empty:
        return "?"
    return row.iloc[0]["name"]


def post_voucher(v_date: date, narration: str, lines: List[Dict[str, Any]]) -> tuple[bool, str]:
    """Post a voucher with multiple lines.
    lines: list of {account_id, debit, credit, description}
    Must be balanced (sum debit == sum credit) and accounts must be leaf nodes.
    """
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)
    vouchers = read_csv("vouchers.csv", VOUCHER_COLS)
    entries  = read_csv("entries.csv", ENTRY_COLS)

    # Validate
    if not lines:
        return False, "No lines to post."
    deb_sum = sum(float(l.get("debit") or 0) for l in lines)
    cre_sum = sum(float(l.get("credit") or 0) for l in lines)
    if round(deb_sum - cre_sum, 2) != 0.0:
        return False, f"Voucher not balanced: Dr {deb_sum:.2f} vs Cr {cre_sum:.2f}."
    for l in lines:
        aid = str(l.get("account_id"))
        if not aid or accounts.loc[accounts["account_id"] == aid].empty:
            return False, f"Account {aid} does not exist."
        if not is_leaf_account(accounts, aid):
            return False, f"Account {aid} is a group head; post to a leaf account."

    vid = next_voucher_id(vouchers)
    vouchers = pd.concat([
        vouchers,
        pd.DataFrame([{"voucher_id": vid, "date": str(v_date), "narration": narration.strip()}])
    ], ignore_index=True)

    new_entries = []
    for i, l in enumerate(lines, start=1):
        new_entries.append({
            "voucher_id": vid,
            "line_no": i,
            "account_id": str(l["account_id"]),
            "debit": float(l.get("debit") or 0.0),
            "credit": float(l.get("credit") or 0.0),
            "description": (l.get("description") or "").strip(),
        })
    entries = pd.concat([entries, pd.DataFrame(new_entries)], ignore_index=True)

    write_csv("vouchers.csv", vouchers)
    write_csv("entries.csv", entries)
    return True, vid


def ledger_df(account_id: str, date_from: date | None = None, date_to: date | None = None) -> pd.DataFrame:
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)
    vouchers = read_csv("vouchers.csv", VOUCHER_COLS)
    entries  = read_csv("entries.csv", ENTRY_COLS)

    df = entries.loc[entries["account_id"] == account_id].copy()
    if df.empty:
        return pd.DataFrame(columns=["Date", "Voucher", "Description", "Debit", "Credit", "Balance"])
    df = df.merge(vouchers, on="voucher_id", how="left")
    df["date"] = pd.to_datetime(df["date"]) 

    if date_from:
        df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["date"] <= pd.to_datetime(date_to)]

    df = df.sort_values(["date", "voucher_id", "line_no"]) 
    df["balance"] = df["debit"].astype(float) - df["credit"].astype(float)
    df["Balance"] = df["balance"].cumsum()

    out = pd.DataFrame({
        "Date": df["date"].dt.date,
        "Voucher": df["voucher_id"],
        "Description": df["description"].where(df["description"].ne(""), df["narration"]).fillna(""),
        "Debit": df["debit"].astype(float),
        "Credit": df["credit"].astype(float),
        "Balance": df["Balance"].astype(float),
    })
    return out


def trial_balance(as_on: date | None = None) -> pd.DataFrame:
    vouchers = read_csv("vouchers.csv", VOUCHER_COLS)
    entries  = read_csv("entries.csv", ENTRY_COLS)

    df = entries.copy()
    if df.empty:
        return pd.DataFrame(columns=["Account ID", "Account Name", "Debit", "Credit"])

    if as_on:
        v_ok = vouchers.loc[pd.to_datetime(vouchers["date"]) <= pd.to_datetime(as_on), "voucher_id"]
        df = df[df["voucher_id"].isin(v_ok)]

    df = df.groupby("account_id", as_index=False)[["debit", "credit"]].sum()
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)
    df = df.merge(accounts[["account_id", "name", "type"]], on="account_id", how="left")
    df = df[["account_id", "name", "debit", "credit", "type"]]
    df = df.rename(columns={"account_id": "Account ID", "name": "Account Name", "debit": "Debit", "credit": "Credit"})
    return df


def pl_and_balance_sheet(as_on: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tb = trial_balance(as_on)
    if tb.empty:
        pl = pd.DataFrame(columns=["Head", "Amount (Dr)", "Amount (Cr)"])
        bs = pd.DataFrame(columns=["Head", "Amount (Dr)", "Amount (Cr)"])
        return pl, bs

    def sum_head(df, head_type):
        tmp = df[df["type"] == head_type]
        return tmp[["Debit", "Credit"]].sum(numeric_only=True)

    # Profit & Loss: Income vs Expenses
    inc = sum_head(tb, "Income")
    exp = sum_head(tb, "Expenses")
    income = inc["Credit"] - inc["Debit"]
    expense = exp["Debit"] - exp["Credit"]
    net_profit = income - expense

    pl = pd.DataFrame([
        {"Head": "Income (Cr-Dr)", "Amount (Dr)": 0.0, "Amount (Cr)": float(max(income, 0))},
        {"Head": "Expenses (Dr-Cr)", "Amount (Dr)": float(max(expense, 0)), "Amount (Cr)": 0.0},
        {"Head": "Net Profit / (Loss)", "Amount (Dr)": float(max(-net_profit, 0)), "Amount (Cr)": float(max(net_profit, 0))},
    ])

    # Balance Sheet: Assets = Liabilities + Equity (+ Profit)
    ass = sum_head(tb, "Assets")
    lia = sum_head(tb, "Liabilities")
    eqt = sum_head(tb, "Equity")

    assets = ass["Debit"] - ass["Credit"]
    liabilities = lia["Credit"] - lia["Debit"]
    equity = eqt["Credit"] - eqt["Debit"]

    # Add net profit to equity side
    equity_plus_pnl = equity + net_profit

    bs = pd.DataFrame([
        {"Head": "Assets (Dr-Cr)", "Amount (Dr)": float(max(assets, 0)), "Amount (Cr)": 0.0},
        {"Head": "Liabilities (Cr-Dr)", "Amount (Dr)": 0.0, "Amount (Cr)": float(max(liabilities, 0))},
        {"Head": "Equity + P&L", "Amount (Dr)": 0.0, "Amount (Cr)": float(max(equity_plus_pnl, 0))},
    ])

    return pl, bs

# ---------------------------
# UI Components
# ---------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")
st.title(APP_TITLE)
st.caption("Currency: INR • Numbers show Indian comma separators • Data saved as CSV in ./data")

init_store()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Dashboard",
    "Vouchers",
    "Ledger",
    "Trial Balance",
    "Profit & Loss",
    "Balance Sheet",
    "Masters (Chart of Accounts)",
    "Backup / Export",
])

# Optional demo data seeding (kept off by default)
with st.sidebar.expander("Optional: seed a few leaf accounts"):
    if st.button("Add common leaf accounts"):
        acc = read_csv("accounts.csv", ACCOUNT_COLS)
        ids_existing = set(acc["account_id"].astype(str))
        new_rows = [r for r in DEFAULT_LEAF_ACCOUNTS if r[0] not in ids_existing]
        if new_rows:
            acc = pd.concat([acc, pd.DataFrame(new_rows, columns=ACCOUNT_COLS)], ignore_index=True)
            write_csv("accounts.csv", acc)
            st.success("Added: Bank, Cash, Creditors, Capital, Sales, Purchases.")
        else:
            st.info("Those accounts already exist.")

# --------------
# Dashboard
# --------------
if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    tb = trial_balance()

    # Compute quick totals
    assets = tb[tb["type"] == "Assets"][['Debit','Credit']].sum()
    liabilities = tb[tb["type"] == "Liabilities"][['Debit','Credit']].sum()
    equity = tb[tb["type"] == "Equity"][['Debit','Credit']].sum()

    assets_val = float(assets['Debit'] - assets['Credit']) if not tb.empty else 0.0
    liabilities_val = float(liabilities['Credit'] - liabilities['Debit']) if not tb.empty else 0.0
    equity_val = float(equity['Credit'] - equity['Debit']) if not tb.empty else 0.0

    pl, bs = pl_and_balance_sheet()
    # Net profit from PL table last row
    if not pl.empty:
        net_profit = float(pl.iloc[-1]['Amount (Cr)'] - pl.iloc[-1]['Amount (Dr)'])
    else:
        net_profit = 0.0

    col1.metric("Assets", f"₹ {format_inr(assets_val)}")
    col2.metric("Liabilities", f"₹ {format_inr(liabilities_val)}")
    col3.metric("Equity", f"₹ {format_inr(equity_val)}")
    col4.metric("Net Profit", f"₹ {format_inr(net_profit)}")

    st.divider()
    st.subheader("Recent vouchers")
    vouchers = read_csv("vouchers.csv", VOUCHER_COLS).sort_values("voucher_id", ascending=False).head(10)
    if vouchers.empty:
        st.info("No vouchers yet. Add some from the Vouchers page.")
    else:
        st.dataframe(vouchers, use_container_width=True)

# --------------
# Vouchers
# --------------
elif page == "Vouchers":
    st.subheader("Create Voucher (double-entry, multi-line)")
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)
    leaf_accounts = accounts[[aid for aid in accounts.index if is_leaf_account(accounts, accounts.loc[aid, 'account_id'])]]

    if accounts.empty or len(accounts) < 5:
        st.info("Tip: add leaf accounts under 'Masters' (e.g., Bank, Cash, Sales, Purchases)")

    # Date & narration
    c1, c2 = st.columns(2)
    v_date = c1.date_input("Date", value=date.today())
    narration = c2.text_input("Narration", value="")

    st.write("**Lines** (enter Debits and Credits; totals must match)")

    # Build a dynamic table for lines
    if 'lines' not in st.session_state:
        st.session_state['lines'] = [
            {"account_id": "", "debit": 0.0, "credit": 0.0, "description": ""},
            {"account_id": "", "debit": 0.0, "credit": 0.0, "description": ""},
        ]

    lines = st.session_state['lines']

    # Helper: account options (leaf-only)
    def leaf_options():
        acc = read_csv("accounts.csv", ACCOUNT_COLS)
        # Filter for leaves
        leaves = []
        for _, r in acc.iterrows():
            if is_leaf_account(acc, r['account_id']):
                leaves.append((f"{r['account_id']} — {r['name']} ({r['type']})", r['account_id']))
        return leaves

    remove_idx = None
    for i, row in enumerate(lines):
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([3, 1.2, 1.2, 3, 0.8])
            choice = c1.selectbox(
                f"Account (row {i+1})",
                options=["Select account..."] + [lbl for lbl, _ in leaf_options()],
                index=0,
                key=f"acc_{i}"
            )
            if choice != "Select account...":
                # Map back to account_id
                acc_id = dict(leaf_options())[choice]
                row['account_id'] = acc_id
            row['debit'] = c2.number_input("Debit", min_value=0.0, step=0.01, key=f"dr_{i}", value=float(row.get('debit',0)))
            row['credit'] = c3.number_input("Credit", min_value=0.0, step=0.01, key=f"cr_{i}", value=float(row.get('credit',0)))
            row['description'] = c4.text_input("Line description", value=row.get('description',''), key=f"desc_{i}")
            if c5.button("✖", key=f"del_{i}"):
                remove_idx = i
        st.write("")

    if remove_idx is not None and 0 <= remove_idx < len(lines):
        lines.pop(remove_idx)

    cadd1, cadd2, cadd3 = st.columns([1, 1, 4])
    if cadd1.button("+ Add line"):
        lines.append({"account_id": "", "debit": 0.0, "credit": 0.0, "description": ""})
    if cadd2.button("Clear all lines"):
        st.session_state['lines'] = []
        lines = []

    deb_total = sum(float(l.get("debit") or 0) for l in lines)
    cre_total = sum(float(l.get("credit") or 0) for l in lines)
    diff = round(deb_total - cre_total, 2)

    st.info(f"Total Debit: ₹ {format_inr(deb_total)} | Total Credit: ₹ {format_inr(cre_total)} | Diff: ₹ {format_inr(diff)}")

    if st.button("Post Voucher", type="primary"):
        ok, msg = post_voucher(v_date, narration, lines)
        if ok:
            st.success(f"Voucher posted with ID: {msg}")
            st.session_state['lines'] = []  # reset
        else:
            st.error(msg)

    st.divider()
    st.subheader("Recent vouchers")
    vouchers = read_csv("vouchers.csv", VOUCHER_COLS).sort_values("voucher_id", ascending=False).head(20)
    st.dataframe(vouchers, use_container_width=True)

# --------------
# Ledger
# --------------
elif page == "Ledger":
    st.subheader("Account Ledger")
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)
    # Leaf-only dropdown for ledger
    leaves = []
    for _, r in accounts.iterrows():
        if is_leaf_account(accounts, r['account_id']):
            leaves.append(f"{r['account_id']} — {r['name']} ({r['type']})")

    sel = st.selectbox("Choose account", options=["Select account..."] + leaves)
    if sel != "Select account...":
        acc_id = sel.split(" — ")[0]
        c1, c2, c3 = st.columns([1,1,5])
        d1 = c1.date_input("From", value=date(date.today().year, 1, 1))
        d2 = c2.date_input("To", value=date.today())
        df = ledger_df(acc_id, d1, d2)
        if df.empty:
            st.info("No entries for this account in the selected period.")
        else:
            df_show = df.copy()
            for col in ["Debit", "Credit", "Balance"]:
                df_show[col] = format_inr(df_show[col])
            st.dataframe(df_show, use_container_width=True)

# --------------
# Trial Balance
# --------------
elif page == "Trial Balance":
    st.subheader("Trial Balance")
    d = st.date_input("As on", value=date.today())
    tb = trial_balance(d)
    if tb.empty:
        st.info("No data yet.")
    else:
        tb_show = tb[["Account ID", "Account Name", "Debit", "Credit", "type"]].copy()
        tb_show["Debit"] = format_inr(tb_show["Debit"].astype(float))
        tb_show["Credit"] = format_inr(tb_show["Credit"].astype(float))
        tb_show = tb_show.rename(columns={"type": "Type"})
        st.dataframe(tb_show, use_container_width=True)
        st.caption("Note: Debits and Credits are period-aggregated up to the given date.")

# --------------
# Profit & Loss
# --------------
elif page == "Profit & Loss":
    st.subheader("Profit & Loss")
    d = st.date_input("As on", value=date.today())
    pl, _ = pl_and_balance_sheet(d)
    if pl.empty:
        st.info("No data yet.")
    else:
        pl_show = pl.copy()
        pl_show["Amount (Dr)"] = format_inr(pl_show["Amount (Dr)"].astype(float))
        pl_show["Amount (Cr)"] = format_inr(pl_show["Amount (Cr)"].astype(float))
        st.dataframe(pl_show, use_container_width=True)

# --------------
# Balance Sheet
# --------------
elif page == "Balance Sheet":
    st.subheader("Balance Sheet")
    d = st.date_input("As on", value=date.today())
    _, bs = pl_and_balance_sheet(d)
    if bs.empty:
        st.info("No data yet.")
    else:
        bs_show = bs.copy()
        bs_show["Amount (Dr)"] = format_inr(bs_show["Amount (Dr)"].astype(float))
        bs_show["Amount (Cr)"] = format_inr(bs_show["Amount (Cr)"].astype(float))
        st.dataframe(bs_show, use_container_width=True)

# --------------
# Masters
# --------------
elif page == "Masters (Chart of Accounts)":
    st.subheader("Chart of Accounts")
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)

    with st.expander("Add account"):
        c1, c2 = st.columns([1.2, 1])
        new_id = c1.text_input("Account ID (e.g., 1101)")
        new_name = c2.text_input("Account Name")
        c3, c4, c5 = st.columns(3)
        new_type = c3.selectbox("Type", ACCOUNT_TYPES)
        # Parent: allow None (for root head) or pick any existing account
        parent_opts = ["None"] + accounts["account_id"].astype(str).tolist()
        new_parent = c4.selectbox("Parent ID", parent_opts)
        is_active = c5.checkbox("Active", value=True)
        if st.button("Create Account"):
            if not new_id or not new_name:
                st.error("Please enter both ID and Name.")
            elif new_id in accounts["account_id"].astype(str).tolist():
                st.error("Account ID already exists.")
            else:
                row = {
                    "account_id": str(new_id),
                    "name": new_name.strip(),
                    "type": new_type,
                    "parent_id": None if new_parent == "None" else str(new_parent),
                    "is_active": bool(is_active),
                }
                accounts = pd.concat([accounts, pd.DataFrame([row])], ignore_index=True)
                write_csv("accounts.csv", accounts)
                st.success("Account added.")

    st.write("**Accounts**")
    st.dataframe(accounts, use_container_width=True)

# --------------
# Export / Backup
# --------------
elif page == "Backup / Export":
    st.subheader("Download CSV backups")
    accounts = read_csv("accounts.csv", ACCOUNT_COLS)
    vouchers = read_csv("vouchers.csv", VOUCHER_COLS)
    entries  = read_csv("entries.csv", ENTRY_COLS)

    c1, c2, c3 = st.columns(3)
    c1.download_button("accounts.csv", accounts.to_csv(index=False).encode('utf-8'), file_name="accounts.csv")
    c2.download_button("vouchers.csv", vouchers.to_csv(index=False).encode('utf-8'), file_name="vouchers.csv")
    c3.download_button("entries.csv",  entries.to_csv(index=False).encode('utf-8'),  file_name="entries.csv")

    st.divider()
    st.info("To restore, place these CSVs back into a ./data folder next to the app and restart.")
