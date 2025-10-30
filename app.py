# app.py — Advanced Accounting System (INR) v2.2
# -------------------------------------------------
# New in v2.2
# - Left-side vertical navigation (tabs-like menu)
# - Auto-generation of Account Codes (root & child)
# - Keeps all v2.1 fixes: opening balance in ledger, safer DB, import/export, audit trail

from __future__ import annotations
import os
import io
import json
import sqlite3
from datetime import date
from typing import List, Dict, Any, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

APP_TITLE = "Masters Luxe Land LLP "

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "accounting.db")

# ---------------------------
# Styling
# ---------------------------

def load_css():
    st.markdown(
        """
        <style>
        .main-header {background: linear-gradient(90deg, #1f4e79, #2d5aa0); padding: 1rem; border-radius: 10px; margin-bottom: 1.25rem; color: white; text-align: center;}
        .metric-card {background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #1f4e79;}
        .error-box {background-color:#f8d7da;border:1px solid #f5c6cb;color:#721c24;padding:0.75rem;border-radius:6px;margin:0.5rem 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# INR formatting
# ---------------------------

def format_inr(n: float | int | pd.Series | None) -> str | pd.Series:
    def _fmt(x: float | int | None) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "₹ 0.00"
        neg = False
        try:
            xv = float(x)
            neg = xv < 0
            xv = abs(xv)
            s = f"{xv:.2f}"
            whole, frac = s.split(".")
            if len(whole) <= 3:
                formatted = whole
            else:
                last3 = whole[-3:]
                rest = whole[:-3]
                parts = []
                while len(rest) > 2:
                    parts.append(rest[-2:])
                    rest = rest[:-2]
                if rest:
                    parts.append(rest)
                parts.reverse()
                formatted = ",".join(parts) + "," + last3
            return f"₹ {'-' if neg else ''}{formatted}.{frac}"
        except Exception:
            return f"₹ {x}"
    if isinstance(n, pd.Series):
        return n.apply(_fmt)
    return _fmt(n)

# ---------------------------
# Database
# ---------------------------

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def connect() -> sqlite3.Connection:
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_database():
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN('Assets','Liabilities','Equity','Income','Expenses')),
                parent_id TEXT REFERENCES accounts(account_id) ON UPDATE CASCADE ON DELETE RESTRICT,
                is_active INTEGER DEFAULT 1,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vouchers (
                voucher_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                narration TEXT,
                reference TEXT,
                created_by TEXT DEFAULT 'system',
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                is_posted INTEGER DEFAULT 1
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voucher_id TEXT NOT NULL REFERENCES vouchers(voucher_id) ON DELETE CASCADE,
                line_no INTEGER NOT NULL,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                debit REAL DEFAULT 0,
                credit REAL DEFAULT 0,
                description TEXT,
                UNIQUE(voucher_id, line_no)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_values TEXT,
                new_values TEXT,
                user_id TEXT DEFAULT 'system',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()

# ---------------------------
# Data Access Layer
# ---------------------------

class DB:
    def q(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)
    def x(self, sql: str, params: tuple = ()) -> None:
        with connect() as conn:
            conn.execute(sql, params)
            conn.commit()

db = DB()

# ---------------------------
# Business Logic
# ---------------------------

class Service:
    def seed_roots(self):
        roots = [
            ("1000", "Assets", "Assets", None, 1, "Main asset group"),
            ("2000", "Liabilities", "Liabilities", None, 1, "Main liability group"),
            ("3000", "Equity", "Equity", None, 1, "Main equity group"),
            ("4000", "Income", "Income", None, 1, "Main income group"),
            ("5000", "Expenses", "Expenses", None, 1, "Main expense group"),
        ]
        for r in roots:
            db.x(
                "INSERT OR IGNORE INTO accounts(account_id,name,type,parent_id,is_active,description) VALUES(?,?,?,?,?,?)",
                r,
            )

    def accounts(self) -> pd.DataFrame:
        return db.q("SELECT * FROM accounts ORDER BY account_id")

    def vouchers(self, limit: Optional[int] = None) -> pd.DataFrame:
        sql = "SELECT * FROM vouchers ORDER BY date DESC, voucher_id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return db.q(sql)

    def entries(self, voucher_id: Optional[str] = None) -> pd.DataFrame:
        if voucher_id:
            return db.q("SELECT * FROM entries WHERE voucher_id=? ORDER BY line_no", (voucher_id,))
        return db.q("SELECT * FROM entries ORDER BY voucher_id, line_no")

    def leaf_accounts(self) -> List[Tuple[str, str]]:
        a = self.accounts()
        if a.empty:
            return []
        parent_ids = set(a["parent_id"].dropna().astype(str))
        leaves = []
        for _, r in a.iterrows():
            if str(r["account_id"]) not in parent_ids and pd.notna(r["parent_id"]):
                label = f"{r['account_id']} — {r['name']} ({r['type']})"
                leaves.append((label, str(r["account_id"])) )
        return sorted(leaves)

    def validate_voucher(self, lines: List[Dict]) -> Tuple[bool, str]:
        if not lines:
            return False, "No lines to post."
        td = sum(float(l.get("debit", 0) or 0) for l in lines)
        tc = sum(float(l.get("credit", 0) or 0) for l in lines)
        if round(td - tc, 2) != 0:
            return False, f"Voucher not balanced: Dr {format_inr(td)} vs Cr {format_inr(tc)}"
        leaf_ids = {acc_id for _, acc_id in self.leaf_accounts()}
        for l in lines:
            acc = str(l.get("account_id", "")).strip()
            if not acc:
                return False, "Please select an account for all lines."
            if acc not in leaf_ids:
                return False, f"Account {acc} is not a valid leaf account."
            d = float(l.get("debit", 0) or 0)
            c = float(l.get("credit", 0) or 0)
            if d < 0 or c < 0:
                return False, "Debit and credit cannot be negative."
            if d > 0 and c > 0:
                return False, "A line cannot have both debit and credit."
            if d == 0 and c == 0:
                return False, "Each line must have either a debit or a credit."
        return True, "OK"

    def next_voucher_id(self) -> str:
        v = self.vouchers(limit=1)
        if v.empty:
            return "V000001"
        last = v.iloc[0]["voucher_id"]
        try:
            n = int(str(last)[1:]) + 1
            return f"V{n:06d}"
        except Exception:
            return "V000001"

    def post(self, vdate: date, narration: str, reference: str, lines: List[Dict]) -> Tuple[bool, str]:
        ok, msg = self.validate_voucher(lines)
        if not ok:
            return False, msg
        vid = self.next_voucher_id()
        try:
            with connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO vouchers(voucher_id,date,narration,reference) VALUES(?,?,?,?)",
                    (vid, str(vdate), narration.strip(), reference.strip()),
                )
                for i, l in enumerate(lines, 1):
                    cur.execute(
                        "INSERT INTO entries(voucher_id,line_no,account_id,debit,credit,description) VALUES(?,?,?,?,?,?)",
                        (
                            vid,
                            i,
                            str(l["account_id"]),
                            float(l.get("debit", 0) or 0),
                            float(l.get("credit", 0) or 0),
                            (l.get("description") or "").strip(),
                        ),
                    )
                cur.execute(
                    "INSERT INTO audit_trail(table_name,record_id,action,new_values) VALUES(?,?,?,?)",
                    ("vouchers", vid, "CREATE", json.dumps({"voucher_id": vid, "date": str(vdate)})),
                )
                conn.commit()
            return True, vid
        except Exception as e:
            return False, f"Error posting voucher: {e}"

    def ledger(self, account_id: str, date_from: Optional[str], date_to: Optional[str]) -> pd.DataFrame:
        sql = (
            "SELECT e.*, v.date, v.narration, v.reference "
            "FROM entries e JOIN vouchers v ON e.voucher_id=v.voucher_id "
            "WHERE e.account_id=?"
        )
        params: List[Any] = [account_id]
        if date_from:
            sql += " AND v.date >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND v.date <= ?"
            params.append(date_to)
        sql += " ORDER BY v.date, e.voucher_id, e.line_no"
        return db.q(sql, tuple(params))

    def opening_balance(self, account_id: str, before_date: str) -> float:
        sql = (
            "SELECT COALESCE(SUM(debit)-SUM(credit),0) AS bal "
            "FROM entries e JOIN vouchers v ON e.voucher_id=v.voucher_id "
            "WHERE e.account_id=? AND v.date < ?"
        )
        df = db.q(sql, (account_id, before_date))
        return float(df.iloc[0]["bal"]) if not df.empty else 0.0

    def trial_balance(self, as_on: Optional[str]) -> pd.DataFrame:
        sql = (
            "SELECT e.account_id, a.name AS account_name, a.type AS account_type, "
            "SUM(e.debit) AS total_debit, SUM(e.credit) AS total_credit "
            "FROM entries e JOIN accounts a ON e.account_id=a.account_id "
            "JOIN vouchers v ON e.voucher_id=v.voucher_id"
        )
        params: List[Any] = []
        if as_on:
            sql += " WHERE v.date <= ?"
            params.append(as_on)
        sql += " GROUP BY e.account_id, a.name, a.type ORDER BY e.account_id"
        return db.q(sql, tuple(params))

    def pl(self, as_on: Optional[str]) -> Tuple[pd.DataFrame, float]:
        tb = self.trial_balance(as_on)
        if tb.empty:
            return pd.DataFrame({"Head": [], "Amount": []}), 0.0
        inc = tb[tb["account_type"] == "Income"]
        exp = tb[tb["account_type"] == "Expenses"]
        total_income = (inc["total_credit"] - inc["total_debit"]).sum()
        total_expense = (exp["total_debit"] - exp["total_credit"]).sum()
        net = float(total_income - total_expense)
        df = pd.DataFrame(
            [
                {"Head": "Total Income", "Amount": total_income},
                {"Head": "Total Expenses", "Amount": total_expense},
                {"Head": "Net Profit/Loss", "Amount": net},
            ]
        )
        return df, net

    def balance_sheet(self, as_on: Optional[str]) -> pd.DataFrame:
        tb = self.trial_balance(as_on)
        if tb.empty:
            return pd.DataFrame({"Head": [], "Amount": []})
        _, net = self.pl(as_on)
        assets = tb[tb["account_type"] == "Assets"]
        liab = tb[tb["account_type"] == "Liabilities"]
        eq = tb[tb["account_type"] == "Equity"]
        total_assets = (assets["total_debit"] - assets["total_credit"]).sum()
        total_liab = (liab["total_credit"] - liab["total_debit"]).sum()
        total_equity = (eq["total_credit"] - eq["total_debit"]).sum() + net
        return pd.DataFrame(
            [
                {"Head": "Total Assets", "Amount": float(total_assets)},
                {"Head": "Total Liabilities", "Amount": float(total_liab)},
                {"Head": "Total Equity + P&L", "Amount": float(total_equity)},
            ]
        )

svc = Service()

# ---------------------------
# UI Helpers
# ---------------------------

def header():
    st.markdown(
        f"""
        <div class=\"main-header\"> 
            <h1>{APP_TITLE}</h1>
            <p>Version {VERSION} • Currency: {CURRENCY}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Left Navigation (vertical tabs)

def left_nav(options: list[str], key: str = "_nav", default: Optional[str] = None) -> str:
    if default is None:
        default = options[0]
    if key not in st.session_state:
        st.session_state[key] = default
    with st.container():
        st.markdown("### 🧭 Menu")
        for opt in options:
            active = st.session_state[key] == opt
            btn = st.button(opt, type=("primary" if active else "secondary"), use_container_width=True, key=f"nav_{opt}")
            if btn:
                st.session_state[key] = opt
                st.rerun()
    return st.session_state[key]

# Auto-code generation

def generate_account_code(parent_id: Optional[str], account_type: str) -> str:
    a = svc.accounts()
    prefix_map = {"Assets": "1", "Liabilities": "2", "Equity": "3", "Income": "4", "Expenses": "5"}
    prefix = prefix_map.get(account_type, "9")
    if not parent_id:
        pool = a[a["account_id"].astype(str).str.startswith(prefix)]
        nums = []
        for aid in pool["account_id"].astype(str):
            if aid.isdigit():
                nums.append(int(aid))
        base = int(prefix) * 1000
        next_code = max([base] + nums) + 1
        return f"{next_code:04d}"
    children = a[a["parent_id"].astype(str) == str(parent_id)]
    max_suffix = 0
    for aid in children["account_id"].astype(str):
        if aid.startswith(str(parent_id)):
            suf = aid[len(str(parent_id)):] or "0"
            if suf.isdigit():
                max_suffix = max(max_suffix, int(suf))
    next_suffix = max_suffix + 1
    return f"{parent_id}{next_suffix:02d}"

# ---------------------------
# Pages
# ---------------------------

def page_dashboard():
    st.subheader("📊 Dashboard")
    tb = svc.trial_balance(None)
    if not tb.empty:
        assets = (tb[tb.account_type == "Assets"]["total_debit"] - tb[tb.account_type == "Assets"]["total_credit"]).sum()
        liab = (tb[tb.account_type == "Liabilities"]["total_credit"] - tb[tb.account_type == "Liabilities"]["total_debit"]).sum()
        eq_only = (tb[tb.account_type == "Equity"]["total_credit"] - tb[tb.account_type == "Equity"]["total_debit"]).sum()
        _, net = svc.pl(None)
        eq_total = eq_only + net
    else:
        assets = liab = eq_total = 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets", format_inr(assets))
    c2.metric("Liabilities", format_inr(liab))
    c3.metric("Equity", format_inr(eq_total))

    st.markdown("---")
    st.subheader("Recent Vouchers")
    v = svc.vouchers(limit=10)
    if v.empty:
        st.info("No vouchers yet.")
    else:
        v = v.copy()
        v["date"] = pd.to_datetime(v["date"]).dt.strftime("%d-%m-%Y")
        st.dataframe(v[["voucher_id", "date", "narration", "reference"]], use_container_width=True)


def page_vouchers():
    st.subheader("📝 Voucher Management")
    t1, t2 = st.tabs(["Create Voucher", "View Vouchers"])

    with t1:
        leaves = svc.leaf_accounts()
        if not leaves:
            st.warning("No leaf accounts. Add accounts in Masters first.")
            return
        col1, col2 = st.columns(2)
        with col1:
            vdate = st.date_input("Date", value=date.today())
        with col2:
            ref = st.text_input("Reference/Cheque No.")
        narr = st.text_area("Narration")
        if "lines" not in st.session_state:
            st.session_state.lines = [{"account_id": "", "debit": 0.0, "credit": 0.0, "description": ""}]
        lines = st.session_state.lines
        rm = None
        for i, ln in enumerate(lines):
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([3, 1.3, 1.3, 2.4, 0.6])
                with c1:
                    opts = ["Select account..."] + [lbl for lbl, _ in leaves]
                    amap = dict(leaves)
                    sel = st.selectbox(f"Account (Line {i+1})", opts, key=f"acc_{i}")
                    if sel != "Select account...":
                        ln["account_id"] = amap.get(sel, "")
                with c2:
                    ln["debit"] = st.number_input("Debit", min_value=0.0, step=0.01, value=float(ln.get("debit", 0)), key=f"dr_{i}")
                with c3:
                    ln["credit"] = st.number_input("Credit", min_value=0.0, step=0.01, value=float(ln.get("credit", 0)), key=f"cr_{i}")
                with c4:
                    ln["description"] = st.text_input("Description", value=ln.get("description", ""), key=f"ds_{i}")
                with c5:
                    if st.button("🗑️", key=f"del_{i}"):
                        rm = i
        if rm is not None:
            lines.pop(rm)
            st.rerun()
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("➕ Add Line"):
            lines.append({"account_id": "", "debit": 0.0, "credit": 0.0, "description": ""})
            st.rerun()
        if c2.button("🧹 Clear All"):
            st.session_state.lines = []
            st.rerun()
        td = sum(float(l.get("debit", 0) or 0) for l in lines)
        tc = sum(float(l.get("credit", 0) or 0) for l in lines)
        diff = td - tc
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Debit", format_inr(td))
        d2.metric("Total Credit", format_inr(tc))
        d3.metric("Difference", format_inr(diff))
        if diff != 0:
            st.error("Voucher is not balanced.")
        if st.button("💾 Post Voucher", type="primary", disabled=(diff != 0)):
            if not narr.strip():
                st.error("Please enter a narration.")
            else:
                ok, res = svc.post(vdate, narr, ref, lines)
                if ok:
                    st.success(f"Voucher posted: {res}")
                    st.session_state.lines = []
                    st.rerun()
                else:
                    st.error(res)

    with t2:
        col1, col2 = st.columns([2, 1])
        with col1:
            needle = st.text_input("🔍 Search", placeholder="Voucher ID or narration")
        with col2:
            lim = st.selectbox("Show records", [20, 50, 100, 200], index=0)
        v = svc.vouchers(limit=int(lim))
        if needle:
            v = v[(v.voucher_id.str.contains(needle, case=False, na=False)) | (v.narration.str.contains(needle, case=False, na=False))]
        if v.empty:
            st.info("No vouchers.")
        else:
            v = v.copy()
            v["date"] = pd.to_datetime(v["date"]).dt.strftime("%d-%m-%Y")
            st.dataframe(v, use_container_width=True)
            st.download_button("📥 Download CSV", v.to_csv(index=False), file_name=f"vouchers_{date.today()}.csv", mime="text/csv")


def page_ledger():
    st.subheader("📚 Account Ledger")
    leaves = svc.leaf_accounts()
    if not leaves:
        st.warning("No accounts. Add in Masters.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        opts = ["Select account..."] + [l for l, _ in leaves]
        amap = dict(leaves)
        sel = st.selectbox("Choose Account", opts)
    if sel == "Select account...":
        return
    acc = amap[sel]
    with c2:
        dfrom = st.date_input("From", value=date(date.today().year, 1, 1))
    with c3:
        dto = st.date_input("To", value=date.today())
    ob = svc.opening_balance(acc, str(dfrom))
    df = svc.ledger(acc, str(dfrom), str(dto))
    rows = []
    rows.append({"date": "", "voucher_id": "", "description": "Opening Balance", "debit": "", "credit": "", "running_balance": ob})
    run = ob
    for _, r in df.iterrows():
        delta = float(r.debit) - float(r.credit)
        run += delta
        rows.append(
            {
                "date": pd.to_datetime(r.date).strftime("%d-%m-%Y"),
                "voucher_id": r.voucher_id,
                "description": r.description if r.description else (r.narration or ""),
                "debit": float(r.debit) if r.debit else "",
                "credit": float(r.credit) if r.credit else "",
                "running_balance": run,
            }
        )
    out = pd.DataFrame(rows)
    disp = out.copy()
    disp["debit"] = disp["debit"].apply(lambda x: format_inr(x) if isinstance(x, (int, float)) and x > 0 else "")
    disp["credit"] = disp["credit"].apply(lambda x: format_inr(x) if isinstance(x, (int, float)) and x > 0 else "")
    disp["running_balance"] = disp["running_balance"].apply(format_inr)
    disp.rename(columns={"date": "Date", "voucher_id": "Voucher", "description": "Description", "debit": "Debit", "credit": "Credit", "running_balance": "Balance"}, inplace=True)
    st.dataframe(disp, use_container_width=True)
    tot_dr = df["debit"].sum()
    tot_cr = df["credit"].sum()
    st.caption(f"Total Debits: {format_inr(tot_dr)} | Total Credits: {format_inr(tot_cr)} | Closing Balance: {format_inr(run)}")
    st.download_button("📥 Download Ledger", disp.to_csv(index=False), file_name=f"ledger_{acc}_{date.today()}.csv", mime="text/csv")


def page_trial_balance():
    st.subheader("⚖️ Trial Balance")
    as_on = st.date_input("As on", value=date.today())
    tb = svc.trial_balance(str(as_on))
    if tb.empty:
        st.info("No data.")
        return
    show = tb.copy()
    show.rename(columns={"account_id": "Account ID", "account_name": "Account Name", "account_type": "Type", "total_debit": "Total Debit", "total_credit": "Total Credit"}, inplace=True)
    show["Total Debit"] = show["Total Debit"].apply(format_inr)
    show["Total Credit"] = show["Total Credit"].apply(format_inr)
    st.dataframe(show, use_container_width=True)
    td = float(tb["total_debit"].sum())
    tc = float(tb["total_credit"].sum())
    st.caption(f"Total Debits: {format_inr(td)} | Total Credits: {format_inr(tc)}")
    if abs(td - tc) < 0.01:
        st.success("✅ Trial Balance is balanced")
    else:
        st.error(f"❌ Not balanced. Difference: {format_inr(td - tc)}")
    st.download_button("📥 Download TB", show.to_csv(index=False), file_name=f"trial_balance_{as_on}.csv", mime="text/csv")


def page_pl():
    st.subheader("📈 Profit & Loss Statement")
    as_on = st.date_input("As on", value=date.today())
    df, net = svc.pl(str(as_on))
    if df.empty:
        st.info("No data.")
        return
    show = df.copy()
    show["Amount"] = show["Amount"].apply(format_inr)
    st.dataframe(show, use_container_width=True)
    if net > 0:
        st.success(f"Net Profit: {format_inr(net)}")
    elif net < 0:
        st.error(f"Net Loss: {format_inr(abs(net))}")
    else:
        st.info("Break-even")
    fig = px.bar(df[df["Head"] != "Net Profit/Loss"], x="Head", y="Amount", title="Income vs Expenses", color="Head")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("📥 Download P&L", show.to_csv(index=False), file_name=f"pl_{as_on}.csv", mime="text/csv")


def page_bs():
    st.subheader("🏛️ Balance Sheet")
    as_on = st.date_input("As on", value=date.today())
    bs = svc.balance_sheet(str(as_on))
    if bs.empty:
        st.info("No data.")
        return
    show = bs.copy()
    show["Amount"] = show["Amount"].apply(format_inr)
    st.dataframe(show, use_container_width=True)
    ta = float(bs.loc[bs.Head == "Total Assets", "Amount"].iloc[0]) if not bs.empty else 0.0
    tle = float(bs[bs.Head != "Total Assets"]["Amount"].sum())
    if abs(ta - tle) < 0.01:
        st.success("✅ Balance Sheet is balanced")
    else:
        st.error(f"❌ Difference: {format_inr(ta - tle)}")
    fig = px.pie(bs, values="Amount", names="Head", title="Balance Sheet Composition")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("📥 Download BS", show.to_csv(index=False), file_name=f"balance_sheet_{as_on}.csv", mime="text/csv")


def page_masters():
    st.subheader("🏗️ Chart of Accounts (Masters)")
    t1, t2 = st.tabs(["View Accounts", "Add Account"]) 
    with t1:
        a = svc.accounts()
        if a.empty:
            st.info("No accounts.")
        else:
            disp = a.copy()
            disp["created_date"] = pd.to_datetime(disp["created_date"]).dt.strftime("%d-%m-%Y")
            st.dataframe(disp, use_container_width=True)
            st.download_button("📥 Download Accounts", disp.to_csv(index=False), file_name=f"accounts_{date.today()}.csv", mime="text/csv")
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            aid = st.text_input("Account ID", key="aid", placeholder="Auto or enter manually")
            aname = st.text_input("Account Name", placeholder="e.g., State Bank of India")
        with c2:
            atype = st.selectbox("Account Type", ["Assets", "Liabilities", "Equity", "Income", "Expenses"], key="atype")
            a = svc.accounts()
            parent_opts = ["None (Root)"] + [f"{r.account_id} — {r.name}" for _, r in a.iterrows()] if not a.empty else ["None (Root)"]
            parent_sel = st.selectbox("Parent Account", parent_opts, key="parent_sel")
            parent_id = None if parent_sel.startswith("None") else parent_sel.split(" — ")[0]
        desc = st.text_area("Description (Optional)")
        active = st.checkbox("Active", value=True)

        gen_col, create_col = st.columns([1,1])
        with gen_col:
            if st.button("⚙️ Auto-generate Code", use_container_width=True):
                try:
                    code = generate_account_code(parent_id, st.session_state.get("atype", "Assets"))
                    st.session_state["aid"] = code
                    st.success(f"Suggested Code: {code}")
                except Exception as e:
                    st.error(f"Auto-code failed: {e}")
        with create_col:
            if st.button("➕ Create Account", type="primary", use_container_width=True):
                aid_val = (st.session_state.get("aid") or "").strip()
                if not aid_val or not aname.strip():
                    st.error("Account ID and Name are required.")
                else:
                    try:
                        db.x(
                            "INSERT INTO accounts(account_id,name,type,parent_id,is_active,description) VALUES(?,?,?,?,?,?)",
                            (aid_val, aname.strip(), st.session_state.get("atype"), parent_id, 1 if active else 0, desc.strip()),
                        )
                        st.success("Account created")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")


def page_backup():
    st.subheader("💾 Backup & Export")
    a = svc.accounts(); v = svc.vouchers(); e = svc.entries()
    st.markdown("#### One-click ZIP backup (CSV + raw)")
    if st.button("📦 Create Backup ZIP"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            if not a.empty:
                z.writestr("accounts.csv", a.to_csv(index=False))
            if not v.empty:
                z.writestr("vouchers.csv", v.to_csv(index=False))
            if not e.empty:
                z.writestr("entries.csv", e.to_csv(index=False))
            z.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "export_date": str(date.today()),
                        "app_version": VERSION,
                        "totals": {"accounts": len(a), "vouchers": len(v), "entries": len(e)},
                    },
                    indent=2,
                ),
            )
        buf.seek(0)
        st.download_button("📥 Download Backup ZIP", buf.getvalue(), file_name=f"accounting_backup_{date.today()}.zip", mime="application/zip")

    st.markdown("---")
    st.subheader("📤 Import Data (CSV)")
    up = st.file_uploader("Upload CSV", type=["csv"])
    kind = st.selectbox("Data type", ["accounts", "vouchers", "entries"])
    if up is not None:
        df = pd.read_csv(up)
        st.dataframe(df.head())
        if st.button("Import", type="primary"):
            try:
                if kind == "accounts":
                    req = {"account_id", "name", "type"}
                    if not req.issubset(set(df.columns)):
                        st.error("Accounts CSV must include: account_id, name, type")
                    else:
                        for _, r in df.iterrows():
                            db.x(
                                "INSERT OR REPLACE INTO accounts(account_id,name,type,parent_id,is_active,description) VALUES(?,?,?,?,?,?)",
                                (
                                    str(r["account_id"]),
                                    r.get("name", ""),
                                    r.get("type", "Assets"),
                                    (None if pd.isna(r.get("parent_id")) else str(r.get("parent_id"))),
                                    int(1 if r.get("is_active", 1) in (1, True, "1", "true", "True") else 0),
                                    r.get("description", ""),
                                ),
                            )
                elif kind == "vouchers":
                    req = {"voucher_id", "date"}
                    if not req.issubset(set(df.columns)):
                        st.error("Vouchers CSV must include: voucher_id, date")
                    else:
                        for _, r in df.iterrows():
                            db.x(
                                "INSERT OR REPLACE INTO vouchers(voucher_id,date,narration,reference,created_by,is_posted) VALUES(?,?,?,?,?,?)",
                                (
                                    str(r["voucher_id"]),
                                    str(r["date"]),
                                    r.get("narration", ""),
                                    r.get("reference", ""),
                                    r.get("created_by", "import"),
                                    int(1 if r.get("is_posted", 1) in (1, True, "1", "true", "True") else 0),
                                ),
                            )
                            db.x(
                                "INSERT INTO audit_trail(table_name,record_id,action,new_values,user_id) VALUES(?,?,?,?,?)",
                                ("vouchers", str(r["voucher_id"]), "IMPORT", json.dumps(r.to_dict()), "import"),
                            )
                else:  # entries
                    req = {"voucher_id", "line_no", "account_id"}
                    if not req.issubset(set(df.columns)):
                        st.error("Entries CSV must include: voucher_id, line_no, account_id")
                    else:
                        for _, r in df.iterrows():
                            db.x(
                                "INSERT OR REPLACE INTO entries(voucher_id,line_no,account_id,debit,credit,description) VALUES(?,?,?,?,?,?)",
                                (
                                    str(r["voucher_id"]),
                                    int(r["line_no"]),
                                    str(r["account_id"]),
                                    float(r.get("debit", 0) or 0),
                                    float(r.get("credit", 0) or 0),
                                    r.get("description", ""),
                                ),
                            )
                st.success("Import completed")
            except Exception as e:
                st.error(f"Import failed: {e}")

# ---------------------------
# Main
# ---------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")
    load_css()
    init_database()
    svc.seed_roots()
    header()

    col_nav, col_body = st.columns([0.22, 0.78])
    with col_nav:
        page = left_nav([
            "📊 Dashboard", "📝 Vouchers", "📚 Ledger", "⚖️ Trial Balance", "📈 Profit & Loss", "🏛️ Balance Sheet", "🏗️ Masters", "💾 Backup & Export"
        ])
        st.caption(f"Version {VERSION}")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    with col_body:
        if page == "📊 Dashboard":
            page_dashboard()
        elif page == "📝 Vouchers":
            page_vouchers()
        elif page == "📚 Ledger":
            page_ledger()
        elif page == "⚖️ Trial Balance":
            page_trial_balance()
        elif page == "📈 Profit & Loss":
            page_pl()
        elif page == "🏛️ Balance Sheet":
            page_bs()
        elif page == "🏗️ Masters":
            page_masters()
        elif page == "💾 Backup & Export":
            page_backup()

if __name__ == "__main__":
    main()



