# -*- coding: utf-8 -*-
# advanced_accounting_app.py — Enterprise Accounting System (INR, No Demo)
# =========================================================================
# 🏢 ADVANCED FEATURES:
# - Multi-company/entity support
# - Hierarchical chart of accounts with automatic rollups
# - Multi-line transactions (unlimited accounts per voucher)
# - Complete financial statements (P&L, Balance Sheet)
# - Role-based permissions, audit trail
# - Budget management scaffolding
# - Bank reconciliation scaffolding
# - Fixed assets scaffolding
# - Multi-currency capable (base=INR)
# - CSV/Excel COA import (in Setup Wizard)
# - Ledger, Trial Balance, Journal Browser, COA manager
# - Export CSV, KPI metrics

from __future__ import annotations

import os
import json
import hashlib
import datetime as dt
from decimal import Decimal
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import uuid

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sqlalchemy import (
    create_engine, text, Integer, String, Float, Date, DateTime, Boolean,
    MetaData, Table, Column, ForeignKey, Index, JSON, Text, Numeric
)
from sqlalchemy.engine import Engine
from pydantic import BaseModel, Field, validator

# ============================
# CONFIGURATION & CONSTANTS
# ============================
APP_TITLE = "🏢 Enterprise Accounting System"
VERSION = "2.0.0"
DB_FILE = "enterprise_accounting.db"
DEFAULT_DB_URL = f"sqlite:///{DB_FILE}"

# Environment configuration
DB_URL = st.secrets.get("DB_URL", DEFAULT_DB_URL)
APP_PASSWORD = st.secrets.get("APP_PASSWORD")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")
JWT_SECRET = st.secrets.get("JWT_SECRET", "accounting_secret_key")

# Business constants
class AccountType(Enum):
    ASSET = "Asset"
    LIABILITY = "Liability"
    EQUITY = "Equity"
    INCOME = "Income"
    EXPENSE = "Expense"

class VoucherType(Enum):
    PAYMENT = "Payment"
    RECEIPT = "Receipt"
    JOURNAL = "Journal"
    CONTRA = "Contra"
    PURCHASE = "Purchase"
    SALES = "Sales"
    ADJUSTMENT = "Adjustment"

class UserRole(Enum):
    ADMIN = "Admin"
    ACCOUNTANT = "Accountant"
    MANAGER = "Manager"
    VIEWER = "Viewer"

class TransactionStatus(Enum):
    DRAFT = "Draft"
    POSTED = "Posted"
    APPROVED = "Approved"
    CANCELLED = "Cancelled"

# Currencies: base INR
CURRENCIES = ["INR", "USD", "EUR", "GBP", "CAD", "AUD", "JPY", "SAR"]
DEFAULT_CURRENCY = "INR"

# ============================
# DATABASE SCHEMA
# ============================
def get_engine() -> Engine:
    if DB_URL.startswith("sqlite"):
        return create_engine(
            DB_URL,
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True
        )
    return create_engine(DB_URL, pool_pre_ping=True)

metadata = MetaData()

# Companies/Entities
companies = Table(
    "companies", metadata,
    Column("id", String(36), primary_key=True),
    Column("code", String(16), unique=True, nullable=False),
    Column("name", String(128), nullable=False),
    Column("address", Text),
    Column("tax_id", String(32)),
    Column("base_currency", String(3), default=DEFAULT_CURRENCY),
    Column("fiscal_year_start", Integer, default=1),  # Month (1-12)
    Column("created_at", DateTime, default=dt.datetime.utcnow),
    Column("is_active", Boolean, default=True),
)

# Users with role-based access
users = Table(
    "users", metadata,
    Column("id", String(36), primary_key=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("email", String(128), unique=True, nullable=False),
    Column("password_hash", String(256), nullable=False),
    Column("full_name", String(128), nullable=False),
    Column("role", String(16), nullable=False),
    Column("company_access", JSON),  # List of company IDs user can access
    Column("created_at", DateTime, default=dt.datetime.utcnow),
    Column("last_login", DateTime),
    Column("is_active", Boolean, default=True),
)

# Chart of Accounts (hierarchical)
accounts = Table(
    "accounts", metadata,
    Column("id", String(36), primary_key=True),
    Column("company_id", String(36), ForeignKey("companies.id"), nullable=False),
    Column("code", String(32), nullable=False),
    Column("name", String(128), nullable=False),
    Column("type", String(16), nullable=False),
    Column("parent_id", String(36), ForeignKey("accounts.id")),
    Column("level", Integer, default=0),
    Column("is_group", Boolean, default=False),
    Column("currency", String(3), default=DEFAULT_CURRENCY),
    Column("tax_code", String(16)),
    Column("description", Text),
    Column("is_active", Boolean, default=True),
    Column("created_at", DateTime, default=dt.datetime.utcnow),
    Index("idx_accounts_company_code", "company_id", "code"),
)

# Voucher Header
vouchers = Table(
    "vouchers", metadata,
    Column("id", String(36), primary_key=True),
    Column("company_id", String(36), ForeignKey("companies.id"), nullable=False),
    Column("number", String(32), nullable=False),
    Column("date", Date, nullable=False),
    Column("type", String(16), nullable=False),
    Column("reference", String(128)),
    Column("narration", Text),
    Column("total_amount", Numeric(15, 2), default=0),
    Column("currency", String(3), default=DEFAULT_CURRENCY),
    Column("exchange_rate", Numeric(10, 6), default=1),
    Column("status", String(16), default=TransactionStatus.DRAFT.value),
    Column("created_by", String(36), ForeignKey("users.id")),
    Column("approved_by", String(36), ForeignKey("users.id")),
    Column("created_at", DateTime, default=dt.datetime.utcnow),
    Column("posted_at", DateTime),
    Column("tags", JSON),  # For categorization
    Index("idx_vouchers_company_date", "company_id", "date"),
    Index("idx_vouchers_number", "company_id", "number"),
)

# Journal
journal = Table(
    "journal", metadata,
    Column("id", String(36), primary_key=True),
    Column("voucher_id", String(36), ForeignKey("vouchers.id"), nullable=False),
    Column("company_id", String(36), ForeignKey("companies.id"), nullable=False),
    Column("account_id", String(36), ForeignKey("accounts.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("debit", Numeric(15, 2), default=0),
    Column("credit", Numeric(15, 2), default=0),
    Column("currency", String(3), default=DEFAULT_CURRENCY),
    Column("exchange_rate", Numeric(10, 6), default=1),
    Column("base_debit", Numeric(15, 2), default=0),
    Column("base_credit", Numeric(15, 2), default=0),
    Column("description", Text),
    Column("line_number", Integer, default=1),
    Index("idx_journal_account_date", "account_id", "date"),
    Index("idx_journal_company_date", "company_id", "date"),
)

# Budgets
budgets = Table(
    "budgets", metadata,
    Column("id", String(36), primary_key=True),
    Column("company_id", String(36), ForeignKey("companies.id"), nullable=False),
    Column("account_id", String(36), ForeignKey("accounts.id"), nullable=False),
    Column("period_start", Date, nullable=False),
    Column("period_end", Date, nullable=False),
    Column("budget_amount", Numeric(15, 2), nullable=False),
    Column("notes", Text),
    Column("created_at", DateTime, default=dt.datetime.utcnow),
    Index("idx_budgets_account_period", "account_id", "period_start", "period_end"),
)

# Fixed Assets (scaffold)
fixed_assets = Table(
    "fixed_assets", metadata,
    Column("id", String(36), primary_key=True),
    Column("company_id", String(36), ForeignKey("companies.id"), nullable=False),
    Column("asset_account_id", String(36), ForeignKey("accounts.id"), nullable=False),
    Column("depreciation_account_id", String(36), ForeignKey("accounts.id"), nullable=False),
    Column("expense_account_id", String(36), ForeignKey("accounts.id"), nullable=False),
    Column("name", String(128), nullable=False),
    Column("purchase_date", Date, nullable=False),
    Column("purchase_cost", Numeric(15, 2), nullable=False),
    Column("useful_life_years", Integer, nullable=False),
    Column("salvage_value", Numeric(15, 2), default=0),
    Column("depreciation_method", String(16), default="STRAIGHT_LINE"),
    Column("is_active", Boolean, default=True),
)

# Audit Trail
audit_log = Table(
    "audit_log", metadata,
    Column("id", String(36), primary_key=True),
    Column("company_id", String(36), ForeignKey("companies.id")),
    Column("user_id", String(36), ForeignKey("users.id")),
    Column("action", String(32), nullable=False),
    Column("table_name", String(64)),
    Column("record_id", String(36)),
    Column("old_values", JSON),
    Column("new_values", JSON),
    Column("ip_address", String(45)),
    Column("user_agent", Text),
    Column("timestamp", DateTime, default=dt.datetime.utcnow),
    Index("idx_audit_company_date", "company_id", "timestamp"),
)

# Bank Reconciliation scaffold
bank_statements = Table(
    "bank_statements", metadata,
    Column("id", String(36), primary_key=True),
    Column("company_id", String(36), ForeignKey("companies.id"), nullable=False),
    Column("account_id", String(36), ForeignKey("accounts.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("description", String(256)),
    Column("amount", Numeric(15, 2), nullable=False),
    Column("balance", Numeric(15, 2)),
    Column("is_reconciled", Boolean, default=False),
    Column("journal_id", String(36), ForeignKey("journal.id")),
)

# Initialize DB
engine = get_engine()
with engine.begin() as conn:
    metadata.create_all(conn)

# ============================
# DATA CLASSES & MODELS
# ============================
@dataclass
class CompanyContext:
    id: str
    code: str
    name: str
    base_currency: str
    fiscal_year_start: int

@dataclass
class UserContext:
    id: str
    username: str
    role: UserRole
    company_access: List[str]
    current_company: Optional[CompanyContext] = None

class MultiLineVoucherInput(BaseModel):
    company_id: str
    date: dt.date
    type: VoucherType
    reference: str = ""
    narration: str = ""
    currency: str = DEFAULT_CURRENCY
    exchange_rate: float = 1.0
    lines: List[Dict[str, Any]]
    tags: List[str] = []

    @validator('lines')
    def validate_balanced_entries(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 lines required")
        total_dr = sum(Decimal(str(line.get('debit', 0))) for line in v)
        total_cr = sum(Decimal(str(line.get('credit', 0))) for line in v)
        if total_dr != total_cr:
            raise ValueError(f"Transaction not balanced: Dr={total_dr}, Cr={total_cr}")
        return v

class FinancialReport(BaseModel):
    company_id: str
    report_type: str
    period_start: dt.date
    period_end: dt.date
    data: Dict[str, Any]
    currency: str = DEFAULT_CURRENCY

# ============================
# AUTH & SESSION
# ============================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def init_admin_user():
    """Create default admin user if none exists (no demo companies)."""
    with engine.begin() as conn:
        result = conn.execute(text("SELECT COUNT(*) as count FROM users")).fetchone()
        if result.count == 0:
            admin_id = str(uuid.uuid4())
            conn.execute(users.insert().values(
                id=admin_id,
                username="admin",
                email="admin@company.com",
                password_hash=hash_password(ADMIN_PASSWORD),
                full_name="System Administrator",
                role=UserRole.ADMIN.value,
                company_access=["*"]
            ))
            st.info(f"🔑 Default admin created. Username: admin, Password: {ADMIN_PASSWORD}")

def authenticate_user(username: str, password: str) -> Optional[UserContext]:
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT * FROM users WHERE username = :u AND is_active = 1"),
            {"u": username}
        ).fetchone()
        if result and verify_password(password, result.password_hash):
            conn.execute(
                text("UPDATE users SET last_login = :now WHERE id = :id"),
                {"now": dt.datetime.utcnow(), "id": result.id}
            )
            return UserContext(
                id=result.id,
                username=result.username,
                role=UserRole(result.role),
                company_access=result.company_access or []
            )
    return None

def check_advanced_auth():
    """Login screen; no demo auto-creation."""
    init_admin_user()

    if not st.session_state.get("user_context"):
        st.sidebar.markdown("### 🔐 Login")
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state["user_context"] = user
                    st.success(f"Welcome {user.username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ Note")
        st.sidebar.info("On first login, you’ll be prompted to create your real company. No demo data will be created.")
        return False
    return True

# ============================
# COMPANY ACCESS & HELPERS
# ============================
def _company_count() -> int:
    with engine.begin() as conn:
        return conn.execute(text(
            "SELECT COUNT(*) AS c FROM companies WHERE is_active = 1"
        )).fetchone().c

def get_user_companies(user: UserContext) -> List[CompanyContext]:
    with engine.begin() as conn:
        if "*" in user.company_access:  # Admin/full
            query = text("SELECT * FROM companies WHERE is_active = 1 ORDER BY name")
            results = conn.execute(query).fetchall()
        else:
            if not user.company_access:
                return []
            placeholders = ",".join([f":id{i}" for i in range(len(user.company_access))])
            query = text(f"""
                SELECT * FROM companies
                WHERE id IN ({placeholders}) AND is_active = 1
                ORDER BY name
            """)
            params = {f"id{i}": cid for i, cid in enumerate(user.company_access)}
            results = conn.execute(query, params).fetchall()

        return [
            CompanyContext(
                id=r.id, code=r.code, name=r.name,
                base_currency=r.base_currency, fiscal_year_start=r.fiscal_year_start
            ) for r in results
        ]

def log_audit_event(conn, company_id: str, user_id: str, action: str,
                    table_name: str, record_id: str, old_values: dict, new_values: dict):
    conn.execute(audit_log.insert().values(
        id=str(uuid.uuid4()),
        company_id=company_id,
        user_id=user_id,
        action=action,
        table_name=table_name,
        record_id=record_id,
        old_values=old_values,
        new_values=new_values,
        timestamp=dt.datetime.utcnow()
    ))

def create_basic_accounts(conn, company_id: str):
    """Creates a standard COA (parents first, then children)."""
    basic_accounts = [
        # Assets
        ("1000", "Current Assets", AccountType.ASSET.value, True, None),
        ("1100", "Cash and Bank", AccountType.ASSET.value, False, "1000"),
        ("1200", "Accounts Receivable", AccountType.ASSET.value, False, "1000"),
        ("1300", "Inventory", AccountType.ASSET.value, False, "1000"),
        ("1400", "Prepaid Expenses", AccountType.ASSET.value, False, "1000"),

        ("1500", "Fixed Assets", AccountType.ASSET.value, True, None),
        ("1510", "Equipment", AccountType.ASSET.value, False, "1500"),
        ("1520", "Furniture & Fixtures", AccountType.ASSET.value, False, "1500"),
        ("1530", "Accumulated Depreciation", AccountType.ASSET.value, False, "1500"),

        # Liabilities
        ("2000", "Current Liabilities", AccountType.LIABILITY.value, True, None),
        ("2100", "Accounts Payable", AccountType.LIABILITY.value, False, "2000"),
        ("2200", "Accrued Expenses", AccountType.LIABILITY.value, False, "2000"),
        ("2300", "Short-term Loans", AccountType.LIABILITY.value, False, "2000"),

        ("2500", "Long-term Liabilities", AccountType.LIABILITY.value, True, None),
        ("2510", "Long-term Loans", AccountType.LIABILITY.value, False, "2500"),

        # Equity
        ("3000", "Equity", AccountType.EQUITY.value, True, None),
        ("3100", "Share Capital", AccountType.EQUITY.value, False, "3000"),
        ("3200", "Retained Earnings", AccountType.EQUITY.value, False, "3000"),
        ("3300", "Current Year Earnings", AccountType.EQUITY.value, False, "3000"),

        # Income
        ("4000", "Revenue", AccountType.INCOME.value, True, None),
        ("4100", "Sales Revenue", AccountType.INCOME.value, False, "4000"),
        ("4200", "Service Revenue", AccountType.INCOME.value, False, "4000"),
        ("4300", "Other Income", AccountType.INCOME.value, False, "4000"),

        # Expenses
        ("5000", "Operating Expenses", AccountType.EXPENSE.value, True, None),
        ("5100", "Cost of Goods Sold", AccountType.EXPENSE.value, False, "5000"),
        ("5200", "Salaries & Wages", AccountType.EXPENSE.value, False, "5000"),
        ("5300", "Rent Expense", AccountType.EXPENSE.value, False, "5000"),
        ("5400", "Utilities Expense", AccountType.EXPENSE.value, False, "5000"),
        ("5500", "Office Supplies", AccountType.EXPENSE.value, False, "5000"),
        ("5600", "Marketing Expense", AccountType.EXPENSE.value, False, "5000"),
        ("5700", "Insurance Expense", AccountType.EXPENSE.value, False, "5000"),
        ("5800", "Depreciation Expense", AccountType.EXPENSE.value, False, "5000"),
        ("5900", "Other Expenses", AccountType.EXPENSE.value, False, "5000"),
    ]

    account_map: Dict[str, str] = {}

    # parents
    for code, name, acc_type, is_group, parent_code in basic_accounts:
        if is_group:
            account_id = str(uuid.uuid4())
            account_map[code] = account_id
            conn.execute(accounts.insert().values(
                id=account_id,
                company_id=company_id,
                code=code,
                name=name,
                type=acc_type,
                parent_id=None,
                level=0,
                is_group=True,
                currency=DEFAULT_CURRENCY,
                is_active=True
            ))

    # children
    for code, name, acc_type, is_group, parent_code in basic_accounts:
        if not is_group:
            account_id = str(uuid.uuid4())
            parent_id = account_map.get(parent_code) if parent_code else None
            level = 1 if parent_id else 0
            conn.execute(accounts.insert().values(
                id=account_id,
                company_id=company_id,
                code=code,
                name=name,
                type=acc_type,
                parent_id=parent_id,
                level=level,
                is_group=False,
                currency=DEFAULT_CURRENCY,
                is_active=True
            ))

# ============================
# SETUP WIZARD
# ============================
def render_setup_wizard(user: UserContext):
    st.header("🧭 First-Run Setup — Create Your Company")
    st.markdown("No companies found. Create your **original company** to begin.")

    with st.form("company_setup"):
        col1, col2 = st.columns(2)
        with col1:
            company_code = st.text_input("Company Code", placeholder="UNI")
            company_name = st.text_input("Company Name", placeholder="Unitech India Pvt Ltd")
            base_currency = st.selectbox("Base Currency", CURRENCIES, index=CURRENCIES.index(DEFAULT_CURRENCY))
        with col2:
            tax_id = st.text_input("Tax ID (optional)")
            fiscal_start = st.selectbox(
                "Fiscal Year Start",
                ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"],
                index=3  # April (India common)
            )
            create_accounts = st.checkbox("Create basic chart of accounts (recommended)", value=True)

        address = st.text_area("Address (optional)", placeholder="Registered office address…")
        st.markdown("**Optional:** Upload your COA file (CSV/Excel) with columns: `code,name,type,parent_code,is_group`.")
        coa_file = st.file_uploader("Upload COA", type=["csv", "xlsx"], accept_multiple_files=False)

        submitted = st.form_submit_button("Create Company")

    if submitted:
        if not company_code or not company_name:
            st.error("Company code and name are required.")
            return
        try:
            months = ["January","February","March","April","May","June",
                      "July","August","September","October","November","December"]
            fiscal_month = months.index(fiscal_start) + 1
            with engine.begin() as conn:
                company_id = str(uuid.uuid4())
                conn.execute(companies.insert().values(
                    id=company_id,
                    code=company_code.strip().upper(),
                    name=company_name.strip(),
                    address=address.strip() if address else None,
                    tax_id=tax_id.strip() if tax_id else None,
                    base_currency=base_currency,
                    fiscal_year_start=fiscal_month
                ))

                # Seed basic accounts if requested and no COA uploaded
                if create_accounts and coa_file is None:
                    create_basic_accounts(conn, company_id)

                # Or import COA
                if coa_file is not None:
                    if coa_file.name.lower().endswith(".csv"):
                        df = pd.read_csv(coa_file)
                    else:
                        df = pd.read_excel(coa_file)

                    cols_lower = {c.lower(): c for c in df.columns}
                    required = {"code", "name", "type"}
                    if not required.issubset(set(cols_lower.keys())):
                        st.error("COA file must include columns: code, name, type")
                        return

                    rows = df.to_dict(orient="records")
                    id_by_code: Dict[str, str] = {}

                    # Insert top-level rows first
                    for r in rows:
                        parent_code = str(r.get(cols_lower.get("parent_code",""), "")).strip() or None
                        if parent_code is None:
                            acc_id = str(uuid.uuid4())
                            id_by_code[str(r[cols_lower["code"]]).strip()] = acc_id
                            conn.execute(accounts.insert().values(
                                id=acc_id, company_id=company_id,
                                code=str(r[cols_lower["code"]]).strip(),
                                name=str(r[cols_lower["name"]]).strip(),
                                type=str(r[cols_lower["type"]]).strip().title(),
                                parent_id=None, level=0,
                                is_group=bool(r.get(cols_lower.get("is_group",""), False)),
                                currency=base_currency, is_active=True
                            ))

                    # Children (single-level)
                    for r in rows:
                        parent_code = str(r.get(cols_lower.get("parent_code",""), "")).strip() or None
                        if parent_code:
                            acc_id = str(uuid.uuid4())
                            id_by_code[str(r[cols_lower["code"]]).strip()] = acc_id
                            parent_id = id_by_code.get(parent_code)
                            lvl = 1 if parent_id else 0
                            conn.execute(accounts.insert().values(
                                id=acc_id, company_id=company_id,
                                code=str(r[cols_lower["code"]]).strip(),
                                name=str(r[cols_lower["name"]]).strip(),
                                type=str(r[cols_lower["type"]]).strip().title(),
                                parent_id=parent_id, level=lvl,
                                is_group=bool(r.get(cols_lower.get("is_group",""), False)),
                                currency=base_currency, is_active=True
                            ))

                # Audit
                log_audit_event(conn, company_id, user.id, "COMPANY_CREATED",
                                "companies", company_id, {}, {"code": company_code, "name": company_name})

            st.success(f"✅ Company {company_code} — {company_name} created.")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error creating company: {e}")

# ============================
# DATA LOADERS
# ============================
@st.cache_data(show_spinner=False)
def load_hierarchical_accounts(company_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        query = text("""
            WITH RECURSIVE account_hierarchy AS (
                SELECT id, code, name, type, parent_id, level, is_group,
                       code as path, name as full_name
                FROM accounts
                WHERE company_id = :company_id AND parent_id IS NULL

                UNION ALL

                SELECT a.id, a.code, a.name, a.type, a.parent_id, a.level, a.is_group,
                       h.path || '.' || a.code as path,
                       h.full_name || ' → ' || a.name as full_name
                FROM accounts a
                JOIN account_hierarchy h ON a.parent_id = h.id
                WHERE a.company_id = :company_id
            )
            SELECT * FROM account_hierarchy
            ORDER BY path
        """)
        df = pd.read_sql(query, conn, params={"company_id": company_id})
    return df

@st.cache_data(show_spinner=False)
def load_advanced_trial_balance(company_id: str, as_of_date: dt.date) -> pd.DataFrame:
    with engine.begin() as conn:
        query = text("""
            WITH account_balances AS (
                SELECT
                    a.id, a.code, a.name, a.type, a.parent_id, a.level, a.is_group,
                    COALESCE(SUM(j.base_debit), 0) as debit,
                    COALESCE(SUM(j.base_credit), 0) as credit,
                    COALESCE(SUM(j.base_debit), 0) - COALESCE(SUM(j.base_credit), 0) as balance
                FROM accounts a
                LEFT JOIN journal j ON j.account_id = a.id AND j.date <= :as_of_date
                WHERE a.company_id = :company_id AND a.is_active = 1
                GROUP BY a.id, a.code, a.name, a.type, a.parent_id, a.level, a.is_group
            )
            SELECT * FROM account_balances
            ORDER BY code
        """)
        df = pd.read_sql(query, conn, params={
            "company_id": company_id,
            "as_of_date": as_of_date
        })
    return df

# ============================
# POSTING ENGINE
# ============================
def get_next_voucher_number(conn, company_id: str, voucher_type: str) -> str:
    result = conn.execute(text("""
        SELECT COUNT(*) + 1 as next_num
        FROM vouchers
        WHERE company_id = :company_id AND type = :type
    """), {"company_id": company_id, "type": voucher_type}).fetchone()

    company = conn.execute(text(
        "SELECT code FROM companies WHERE id = :id"
    ), {"id": company_id}).fetchone()

    prefix = company.code if company else "GEN"
    return f"{prefix}-{voucher_type[:3].upper()}-{result.next_num:06d}"

def post_multiline_voucher(voucher: MultiLineVoucherInput, user_id: str):
    with engine.begin() as conn:
        voucher_id = str(uuid.uuid4())
        next_num = get_next_voucher_number(conn, voucher.company_id, voucher.type.value)

        total_amount = sum(Decimal(str(line.get('debit', 0))) for line in voucher.lines)

        conn.execute(vouchers.insert().values(
            id=voucher_id,
            company_id=voucher.company_id,
            number=next_num,
            date=voucher.date,
            type=voucher.type.value,
            reference=voucher.reference,
            narration=voucher.narration,
            total_amount=total_amount,
            currency=voucher.currency,
            exchange_rate=voucher.exchange_rate,
            status=TransactionStatus.POSTED.value,
            created_by=user_id,
            posted_at=dt.datetime.utcnow(),
            tags=voucher.tags
        ))

        for i, line in enumerate(voucher.lines, 1):
            debit = Decimal(str(line.get('debit', 0)))
            credit = Decimal(str(line.get('credit', 0)))

            conn.execute(journal.insert().values(
                id=str(uuid.uuid4()),
                voucher_id=voucher_id,
                company_id=voucher.company_id,
                account_id=line['account_id'],
                date=voucher.date,
                debit=debit,
                credit=credit,
                currency=voucher.currency,
                exchange_rate=voucher.exchange_rate,
                base_debit=debit * Decimal(str(voucher.exchange_rate)),
                base_credit=credit * Decimal(str(voucher.exchange_rate)),
                description=line.get('description', ''),
                line_number=i
            ))

        log_audit_event(conn, voucher.company_id, user_id, "VOUCHER_POSTED",
                        "vouchers", voucher_id, {}, voucher.dict())
        return voucher_id, next_num

# ============================
# FINANCIAL STATEMENTS ENGINE
# ============================
class FinancialStatementsEngine:
    def __init__(self, company_id: str):
        self.company_id = company_id

    def generate_income_statement(self, start_date: dt.date, end_date: dt.date) -> Dict:
        with engine.begin() as conn:
            query = text("""
                SELECT
                    a.type, a.code, a.name,
                    COALESCE(SUM(j.base_credit - j.base_debit), 0) as amount
                FROM accounts a
                LEFT JOIN journal j ON j.account_id = a.id
                    AND j.date BETWEEN :start_date AND :end_date
                WHERE a.company_id = :company_id
                    AND a.type IN ('Income', 'Expense')
                    AND a.is_active = 1
                GROUP BY a.type, a.code, a.name
                ORDER BY a.type, a.code
            """)
            results = conn.execute(query, {
                "company_id": self.company_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchall()

            income_accounts = []
            expense_accounts = []
            total_income = Decimal('0')
            total_expenses = Decimal('0')

            for row in results:
                amt = Decimal(str(row.amount))
                account_data = {"code": row.code, "name": row.name, "amount": float(amt)}
                if row.type == "Income":
                    income_accounts.append(account_data)
                    total_income += amt
                else:
                    expense_accounts.append(account_data)
                    total_expenses += amt

            net_income = total_income - total_expenses

            gross_margin = float((total_income / total_income) * 100) if total_income != 0 else 0.0
            expense_ratio = float((total_expenses / total_income) * 100) if total_income != 0 else 0.0

            return {
                "period": f"{start_date} to {end_date}",
                "income": {"accounts": income_accounts, "total": float(total_income)},
                "expenses": {"accounts": expense_accounts, "total": float(total_expenses)},
                "net_income": float(net_income),
                "metrics": {
                    "gross_margin": gross_margin,
                    "expense_ratio": expense_ratio
                }
            }

    def generate_balance_sheet(self, as_of_date: dt.date) -> Dict:
        with engine.begin() as conn:
            query = text("""
                SELECT
                    a.type, a.code, a.name,
                    COALESCE(SUM(j.base_debit - j.base_credit), 0) as amount
                FROM accounts a
                LEFT JOIN journal j ON j.account_id = a.id AND j.date <= :as_of_date
                WHERE a.company_id = :company_id
                    AND a.type IN ('Asset', 'Liability', 'Equity')
                    AND a.is_active = 1
                GROUP BY a.type, a.code, a.name
                HAVING ABS(COALESCE(SUM(j.base_debit - j.base_credit), 0)) > 0.01
                ORDER BY a.type, a.code
            """)
            results = conn.execute(query, {
                "company_id": self.company_id,
                "as_of_date": as_of_date
            }).fetchall()

            assets, liabilities, equity = [], [], []
            total_assets = Decimal('0')
            total_liabilities = Decimal('0')
            total_equity = Decimal('0')

            for row in results:
                amount = Decimal(str(row.amount))
                account_data = {"code": row.code, "name": row.name, "amount": float(abs(amount))}
                if row.type == "Asset":
                    assets.append(account_data)
                    total_assets += abs(amount)
                elif row.type == "Liability":
                    liabilities.append(account_data)
                    total_liabilities += abs(amount)
                else:
                    equity.append(account_data)
                    total_equity += abs(amount)

            return {
                "as_of_date": str(as_of_date),
                "assets": {"accounts": assets, "total": float(total_assets)},
                "liabilities": {"accounts": liabilities, "total": float(total_liabilities)},
                "equity": {"accounts": equity, "total": float(total_equity)},
                "total_liab_equity": float(total_liabilities + total_equity),
                "balanced": abs(float(total_assets - total_liabilities - total_equity)) < 0.01
            }
# ============================
# UI COMPONENTS
# ============================
def render_dashboard(company: CompanyContext, user: UserContext):
    st.header(f"📊 Dashboard — {company.name}")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=dt.date(dt.date.today().year, 4, 1))  # FY start Apr 1 (IN)
    with col2:
        end_date = st.date_input("To", value=dt.date.today())

    fs_engine = FinancialStatementsEngine(company.id)
    income_stmt = fs_engine.generate_income_statement(start_date, end_date)
    balance_sheet = fs_engine.generate_balance_sheet(end_date)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"{company.base_currency} {income_stmt['income']['total']:,.2f}")
    with col2:
        st.metric("Net Income", f"{company.base_currency} {income_stmt['net_income']:,.2f}",
                  delta=f"{income_stmt['metrics']['gross_margin']:.1f}% margin")
    with col3:
        st.metric("Total Assets", f"{company.base_currency} {balance_sheet['assets']['total']:,.2f}")
    with col4:
        st.metric("Total Equity", f"{company.base_currency} {balance_sheet['equity']['total']:,.2f}")

    col1, col2 = st.columns(2)
    with col1:
        if income_stmt['income']['accounts'] or income_stmt['expenses']['accounts']:
            fig = go.Figure(data=[
                go.Bar(name='Income', x=['Income'], y=[income_stmt['income']['total']]),
                go.Bar(name='Expenses', x=['Expenses'], y=[income_stmt['expenses']['total']])
            ])
            fig.update_layout(title="Income vs Expenses", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        labels = ['Assets', 'Liabilities', 'Equity']
        values = [
            balance_sheet['assets']['total'],
            balance_sheet['liabilities']['total'],
            balance_sheet['equity']['total']
        ]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title="Balance Sheet Composition")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 Recent Transactions")
    with engine.begin() as conn:
        recent_query = text("""
            SELECT v.number, v.date, v.type, v.reference, v.total_amount, u.full_name
            FROM vouchers v
            LEFT JOIN users u ON v.created_by = u.id
            WHERE v.company_id = :company_id
            ORDER BY v.created_at DESC
            LIMIT 10
        """)
        recent_df = pd.read_sql(recent_query, conn, params={"company_id": company.id})
        if not recent_df.empty:
            recent_df['date'] = pd.to_datetime(recent_df['date']).dt.strftime('%Y-%m-%d')
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent transactions")

def render_multiline_voucher_entry(company: CompanyContext, user: UserContext):
    st.header("📝 Multi-Line Voucher Entry")

    accounts_df = load_hierarchical_accounts(company.id)
    if accounts_df.empty:
        st.warning("No accounts found. Please create accounts first.")
        return

    account_choices: Dict[str, str] = {}
    for _, row in accounts_df.iterrows():
        if not row['is_group']:
            display_name = f"{row['code']} — {row['full_name']}"
            account_choices[display_name] = row['id']

    if not account_choices:
        st.warning("No leaf accounts available for transactions.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        v_date = st.date_input("Date", value=dt.date.today())
        v_type = st.selectbox("Type", [v.value for v in VoucherType])
    with col2:
        reference = st.text_input("Reference")
        currency = st.selectbox("Currency", CURRENCIES, index=CURRENCIES.index(company.base_currency))
    with col3:
        exchange_rate = st.number_input("Exchange Rate", value=1.0, min_value=0.0001,
                                        step=0.0001, format="%.4f")
        tags = st.text_input("Tags (comma-separated)")

    narration = st.text_area("Narration")

    st.subheader("Transaction Lines")
    if 'voucher_lines' not in st.session_state:
        st.session_state.voucher_lines = [
            {"account": "", "debit": 0.0, "credit": 0.0, "description": ""}
        ]

    total_dr = total_cr = 0.0
    for i, line in enumerate(st.session_state.voucher_lines):
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 0.5])
        with col1:
            account_key = f"account_{i}"
            selected_account = st.selectbox(
                "Account", [""] + list(account_choices.keys()),
                key=account_key,
                index=0 if not line["account"] else list(account_choices.keys()).index(line["account"]) + 1
            )
            if selected_account:
                st.session_state.voucher_lines[i]["account"] = selected_account
        with col2:
            debit = st.number_input("Debit", value=line["debit"], min_value=0.0, key=f"debit_{i}", format="%.2f")
            st.session_state.voucher_lines[i]["debit"] = debit
            total_dr += debit
        with col3:
            credit = st.number_input("Credit", value=line["credit"], min_value=0.0, key=f"credit_{i}", format="%.2f")
            st.session_state.voucher_lines[i]["credit"] = credit
            total_cr += credit
        with col4:
            description = st.text_input("Description", value=line["description"], key=f"desc_{i}")
            st.session_state.voucher_lines[i]["description"] = description
        with col5:
            if st.button("🗑️", key=f"delete_{i}", help="Delete line"):
                if len(st.session_state.voucher_lines) > 1:
                    st.session_state.voucher_lines.pop(i)
                    st.rerun()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("➕ Add Line"):
            st.session_state.voucher_lines.append({"account": "", "debit": 0.0, "credit": 0.0, "description": ""})
            st.rerun()

    difference = total_dr - total_cr
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Debits", f"{total_dr:,.2f}")
    with col2:
        st.metric("Total Credits", f"{total_cr:,.2f}")
    with col3:
        if abs(difference) < 0.01:
            st.success("✅ Balanced")
        else:
            st.error(f"❌ Out of balance by {difference:,.2f}")

    if st.button("📤 Post Voucher", type="primary", disabled=abs(difference) >= 0.01):
        try:
            voucher_lines = []
            for line in st.session_state.voucher_lines:
                if line["account"] and (line["debit"] > 0 or line["credit"] > 0):
                    voucher_lines.append({
                        "account_id": account_choices[line["account"]],
                        "debit": line["debit"],
                        "credit": line["credit"],
                        "description": line["description"]
                    })

            if len(voucher_lines) < 2:
                st.error("At least 2 lines with amounts required")
                return

            voucher = MultiLineVoucherInput(
                company_id=company.id,
                date=v_date,
                type=VoucherType(v_type),
                reference=reference,
                narration=narration,
                currency=currency,
                exchange_rate=exchange_rate,
                lines=voucher_lines,
                tags=[tag.strip() for tag in tags.split(",") if tag.strip()]
            )
            voucher_id, voucher_number = post_multiline_voucher(voucher, user.id)
            st.success(f"✅ Voucher {voucher_number} posted successfully!")
            st.session_state.voucher_lines = [{"account": "", "debit": 0.0, "credit": 0.0, "description": ""}]
        except Exception as e:
            st.error(f"Error posting voucher: {str(e)}")

def render_financial_statements(company: CompanyContext):
    st.header("📊 Financial Statements")

    col1, col2, col3 = st.columns(3)
    with col1:
        report_type = st.selectbox("Report Type", ["Income Statement", "Balance Sheet"])
    with col2:
        start_date = st.date_input("Start Date", value=dt.date(dt.date.today().year, 4, 1))
    with col3:
        end_date = st.date_input("End Date", value=dt.date.today())

    fs_engine = FinancialStatementsEngine(company.id)

    if report_type == "Income Statement":
        report_data = fs_engine.generate_income_statement(start_date, end_date)
        st.subheader("📈 Income Statement")
        st.caption(f"Period: {report_data['period']}")

        st.markdown("**INCOME**")
        income_df = pd.DataFrame(report_data['income']['accounts'])
        if not income_df.empty:
            st.dataframe(income_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Total Income: {company.base_currency} {report_data['income']['total']:,.2f}**")

        st.divider()

        st.markdown("**EXPENSES**")
        expense_df = pd.DataFrame(report_data['expenses']['accounts'])
        if not expense_df.empty:
            st.dataframe(expense_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Total Expenses: {company.base_currency} {report_data['expenses']['total']:,.2f}**")

        st.divider()

        net_color = "green" if report_data['net_income'] >= 0 else "red"
        st.markdown(f"**NET INCOME: :{net_color}[{company.base_currency} {report_data['net_income']:,.2f}]**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gross Margin", f"{report_data['metrics']['gross_margin']:.1f}%")
        with col2:
            st.metric("Expense Ratio", f"{report_data['metrics']['expense_ratio']:.1f}%")

    elif report_type == "Balance Sheet":
        report_data = fs_engine.generate_balance_sheet(end_date)
        st.subheader("🏦 Balance Sheet")
        st.caption(f"As of: {report_data['as_of_date']}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ASSETS**")
            assets_df = pd.DataFrame(report_data['assets']['accounts'])
            if not assets_df.empty:
                st.dataframe(assets_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Total Assets: {company.base_currency} {report_data['assets']['total']:,.2f}**")
        with col2:
            st.markdown("**LIABILITIES**")
            liab_df = pd.DataFrame(report_data['liabilities']['accounts'])
            if not liab_df.empty:
                st.dataframe(liab_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Total Liabilities: {company.base_currency} {report_data['liabilities']['total']:,.2f}**")

        st.markdown("**EQUITY**")
        equity_df = pd.DataFrame(report_data['equity']['accounts'])
        if not equity_df.empty:
            st.dataframe(equity_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Total Equity: {company.base_currency} {report_data['equity']['total']:,.2f}**")

        st.markdown(f"**Total Liab + Equity: {company.base_currency} {report_data['total_liab_equity']:,.2f}**")

        if report_data['balanced']:
            st.success("✅ Balance Sheet is balanced!")
        else:
            st.error("❌ Balance Sheet is out of balance!")

def render_ledger_statement(company: CompanyContext):
    st.header("📑 Ledger Statement")

    accounts_df = load_hierarchical_accounts(company.id)
    if accounts_df.empty:
        st.warning("No accounts found.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        account_choices = {f"{row['code']} — {row['name']}": row['id']
                           for _, row in accounts_df.iterrows() if not row['is_group']}
        selected_account = st.selectbox("Account", list(account_choices.keys()))
    with col2:
        start_date = st.date_input("From", value=dt.date(dt.date.today().year, 4, 1))
    with col3:
        end_date = st.date_input("To", value=dt.date.today())
    with col4:
        show_zero_balance = st.checkbox("Show zero balance lines", value=False)

    if selected_account:
        account_id = account_choices[selected_account]
        with engine.begin() as conn:
            opening_query = text("""
                SELECT COALESCE(SUM(base_debit - base_credit), 0) as balance
                FROM journal
                WHERE account_id = :account_id AND date < :start_date
            """)
            opening_balance = conn.execute(opening_query, {
                "account_id": account_id, "start_date": start_date
            }).fetchone().balance

            ledger_query = text("""
                SELECT j.date, v.number, v.type, v.reference, j.description,
                       j.debit, j.credit, j.base_debit, j.base_credit
                FROM journal j
                JOIN vouchers v ON j.voucher_id = v.id
                WHERE j.account_id = :account_id
                    AND j.date BETWEEN :start_date AND :end_date
                ORDER BY j.date, v.number, j.line_number
            """)
            ledger_df = pd.read_sql(ledger_query, conn, params={
                "account_id": account_id,
                "start_date": start_date,
                "end_date": end_date
            })

        st.metric("Opening Balance", f"{company.base_currency} {float(opening_balance):,.2f}")

        if not ledger_df.empty:
            ledger_df['balance'] = 0.0
            running_balance = float(opening_balance)
            for idx, row in ledger_df.iterrows():
                running_balance += float(row['base_debit']) - float(row['base_credit'])
                ledger_df.loc[idx, 'balance'] = running_balance

            if not show_zero_balance:
                ledger_df = ledger_df[(ledger_df['debit'] != 0) | (ledger_df['credit'] != 0)]

            display_df = ledger_df.copy()
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
            display_df = display_df[['date', 'number', 'type', 'reference',
                                     'description', 'debit', 'credit', 'balance']]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            total_debits = ledger_df['debit'].sum()
            total_credits = ledger_df['credit'].sum()
            closing_balance = running_balance

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Debits", f"{company.base_currency} {total_debits:,.2f}")
            with col2:
                st.metric("Total Credits", f"{company.base_currency} {total_credits:,.2f}")
            with col3:
                st.metric("Closing Balance", f"{company.base_currency} {closing_balance:,.2f}")

            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                csv,
                file_name=f"ledger_{account_id}_{start_date}_{end_date}.csv"
            )
        else:
            st.info("No transactions found for the selected period.")

def render_trial_balance(company: CompanyContext):
    st.header("🧮 Trial Balance")

    col1, col2 = st.columns(2)
    with col1:
        as_of_date = st.date_input("As of Date", value=dt.date.today())
    with col2:
        show_zero_balances = st.checkbox("Show zero balances", value=False)

    trial_balance_df = load_advanced_trial_balance(company.id, as_of_date)
    if not trial_balance_df.empty:
        if not show_zero_balances:
            trial_balance_df = trial_balance_df[
                (trial_balance_df['debit'].abs() > 0.01) | (trial_balance_df['credit'].abs() > 0.01)
            ]

        display_df = trial_balance_df[['code', 'name', 'type', 'debit', 'credit', 'balance']].copy()

        total_debits = trial_balance_df['debit'].sum()
        total_credits = trial_balance_df['credit'].sum()
        net_balance = total_debits - total_credits

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Debits", f"{company.base_currency} {total_debits:,.2f}")
        with col2:
            st.metric("Total Credits", f"{company.base_currency} {total_credits:,.2f}")
        with col3:
            st.metric("Net Balance", f"{company.base_currency} {net_balance:,.2f}")
            if abs(net_balance) < 0.01:
                st.success("✅ Trial Balance is balanced!")
            else:
                st.error("❌ Trial Balance is out of balance!")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"trial_balance_{company.code}_{as_of_date}.csv"
        )

        if len(trial_balance_df) > 0:
            type_summary = trial_balance_df.groupby('type')['balance'].sum().reset_index()
            fig = px.bar(type_summary, x='type', y='balance', title="Balance by Account Type", color='balance')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No account balances found.")

def render_journal_browser(company: CompanyContext):
    st.header("📜 Journal Browser")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("From Date", value=dt.date.today() - dt.timedelta(days=30))
    with col2:
        end_date = st.date_input("To Date", value=dt.date.today())
    with col3:
        voucher_types = ["All"] + [v.value for v in VoucherType]
        selected_type = st.selectbox("Voucher Type", voucher_types)
    with col4:
        search_term = st.text_input("Search (reference, narration)")

    where_conditions = ["j.company_id = :company_id", "j.date BETWEEN :start_date AND :end_date"]
    params = {"company_id": company.id, "start_date": start_date, "end_date": end_date}

    if selected_type != "All":
        where_conditions.append("v.type = :voucher_type")
        params["voucher_type"] = selected_type

    if search_term:
        where_conditions.append("(v.reference LIKE :search OR v.narration LIKE :search)")
        params["search"] = f"%{search_term}%"

    with engine.begin() as conn:
        journal_query = text(f"""
            SELECT
                j.date, v.number, v.type, v.reference, a.code as account_code,
                a.name as account_name, j.debit, j.credit, j.description,
                v.narration, u.full_name as created_by
            FROM journal j
            JOIN vouchers v ON j.voucher_id = v.id
            JOIN accounts a ON j.account_id = a.id
            LEFT JOIN users u ON v.created_by = u.id
            WHERE {' AND '.join(where_conditions)}
            ORDER BY j.date DESC, v.number, j.line_number
            LIMIT 1000
        """)
        journal_df = pd.read_sql(journal_query, conn, params=params)

    if not journal_df.empty:
        total_entries = len(journal_df)
        total_debits = journal_df['debit'].sum()
        total_credits = journal_df['credit'].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", f"{total_entries:,}")
        with col2:
            st.metric("Total Debits", f"{company.base_currency} {total_debits:,.2f}")
        with col3:
            st.metric("Total Credits", f"{company.base_currency} {total_credits:,.2f}")

        display_df = journal_df.copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"journal_{company.code}_{start_date}_{end_date}.csv"
        )
    else:
        st.info("No journal entries found matching the criteria.")

def render_chart_of_accounts(company: CompanyContext, user: UserContext):
    st.header("🏗️ Chart of Accounts")

    accounts_df = load_hierarchical_accounts(company.id)

    if not accounts_df.empty:
        st.subheader("Current Accounts")
        display_df = accounts_df[['code', 'full_name', 'type', 'level', 'is_group']].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("➕ Add New Account")

    col1, col2 = st.columns(2)
    with col1:
        account_code = st.text_input("Account Code", placeholder="1001")
        account_name = st.text_input("Account Name", placeholder="Cash in Hand")
        account_type = st.selectbox("Account Type", [t.value for t in AccountType])
        is_group = st.checkbox("Group Account (cannot post transactions)")
        parent_options = ["None (Top Level)"]
        if not accounts_df.empty:
            group_accounts = accounts_df[accounts_df['is_group'] == True]
            parent_options.extend([f"{row['code']} — {row['name']}" for _, row in group_accounts.iterrows()])
        parent_account = st.selectbox("Parent Account", parent_options)
    with col2:
        currency = st.selectbox("Currency", CURRENCIES, index=CURRENCIES.index(company.base_currency))
        tax_code = st.text_input("Tax Code", placeholder="GST18")
        description = st.text_area("Description")

    if st.button("💾 Save Account", type="primary"):
        if not account_code or not account_name:
            st.error("Code and Name are required")
            return
        try:
            parent_id = None
            level = 0
            if parent_account != "None (Top Level)" and not accounts_df.empty:
                parent_code = parent_account.split(" — ")[0]
                parent_row = accounts_df[accounts_df['code'] == parent_code].iloc[0]
                parent_id = parent_row['id']
                level = parent_row['level'] + 1

            with engine.begin() as conn:
                account_id = str(uuid.uuid4())
                conn.execute(accounts.insert().values(
                    id=account_id,
                    company_id=company.id,
                    code=account_code.strip(),
                    name=account_name.strip(),
                    type=account_type,
                    parent_id=parent_id,
                    level=level,
                    is_group=is_group,
                    currency=currency,
                    tax_code=tax_code.strip() if tax_code else None,
                    description=description.strip() if description else None
                ))

                log_audit_event(conn, company.id, user.id, "ACCOUNT_CREATED",
                                "accounts", account_id, {}, {
                                    "code": account_code, "name": account_name, "type": account_type
                                })
            st.success(f"✅ Account {account_code} - {account_name} created successfully!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error creating account: {str(e)}")

def render_user_management(company: CompanyContext, user: UserContext):
    st.header("👥 User Management")

    if user.role != UserRole.ADMIN:
        st.error("Access denied. Admin role required.")
        return

    with engine.begin() as conn:
        users_query = text("""
            SELECT id, username, email, full_name, role, is_active, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        """)
        users_df = pd.read_sql(users_query, conn)

    if not users_df.empty:
        st.subheader("Current Users")
        display_df = users_df[['username', 'full_name', 'email', 'role', 'is_active', 'last_login']].copy()
        display_df['last_login'] = pd.to_datetime(display_df['last_login']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("➕ Add New User")

    col1, col2 = st.columns(2)
    with col1:
        new_username = st.text_input("Username")
        new_email = st.text_input("Email")
        new_full_name = st.text_input("Full Name")
    with col2:
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", [r.value for r in UserRole])

        all_companies = get_user_companies(user)  # Admin can see all
        company_options = [c.name for c in all_companies]
        selected_companies = st.multiselect("Company Access", company_options)

    if st.button("👤 Create User", type="primary"):
        if not all([new_username, new_email, new_full_name, new_password]):
            st.error("All fields are required")
            return
        try:
            company_ids = []
            for comp_name in selected_companies:
                for comp in all_companies:
                    if comp.name == comp_name:
                        company_ids.append(comp.id)
                        break

            with engine.begin() as conn:
                new_user_id = str(uuid.uuid4())
                conn.execute(users.insert().values(
                    id=new_user_id,
                    username=new_username.strip(),
                    email=new_email.strip(),
                    password_hash=hash_password(new_password),
                    full_name=new_full_name.strip(),
                    role=new_role,
                    company_access=company_ids
                ))
                log_audit_event(conn, company.id, user.id, "USER_CREATED",
                                "users", new_user_id, {}, {"username": new_username, "role": new_role})
            st.success(f"✅ User {new_username} created successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error creating user: {str(e)}")

def render_company_management(user: UserContext):
    st.header("🏢 Company Management")

    if user.role != UserRole.ADMIN:
        st.error("Access denied. Admin role required.")
        return

    with engine.begin() as conn:
        companies_query = text("""
            SELECT id, code, name, address, tax_id, base_currency,
                   fiscal_year_start, is_active, created_at
            FROM companies
            ORDER BY created_at DESC
        """)
        companies_df = pd.read_sql(companies_query, conn)

    if not companies_df.empty:
        st.subheader("Current Companies")
        display_df = companies_df[['code', 'name', 'base_currency', 'fiscal_year_start', 'is_active']].copy()
        display_df['fiscal_year_start'] = display_df['fiscal_year_start'].apply(
            lambda x: dt.date(2024, int(x), 1).strftime('%B')
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("➕ Add New Company")

    col1, col2 = st.columns(2)
    with col1:
        company_code = st.text_input("Company Code", placeholder="UNI")
        company_name = st.text_input("Company Name", placeholder="Unitech India Pvt Ltd")
        base_currency = st.selectbox("Base Currency", CURRENCIES, index=CURRENCIES.index(DEFAULT_CURRENCY))
    with col2:
        tax_id = st.text_input("Tax ID", placeholder="GSTIN / PAN")
        fiscal_start = st.selectbox(
            "Fiscal Year Start",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
            index=3  # April
        )
        create_accounts = st.checkbox("Create basic chart of accounts", value=True)

    address = st.text_area("Address", placeholder="Registered office address…")

    if st.button("🏢 Create Company", type="primary"):
        if not company_code or not company_name:
            st.error("Company code and name are required")
            return
        try:
            fiscal_month = ["January","February","March","April","May","June",
                            "July","August","September","October","November","December"].index(fiscal_start) + 1
            with engine.begin() as conn:
                company_id = str(uuid.uuid4())
                conn.execute(companies.insert().values(
                    id=company_id,
                    code=company_code.strip().upper(),
                    name=company_name.strip(),
                    address=address.strip() if address else None,
                    tax_id=tax_id.strip() if tax_id else None,
                    base_currency=base_currency,
                    fiscal_year_start=fiscal_month
                ))

                if create_accounts:
                    create_basic_accounts(conn, company_id)

                log_audit_event(conn, company_id, user.id, "COMPANY_CREATED",
                                "companies", company_id, {}, {"code": company_code, "name": company_name})

            st.success(f"✅ Company {company_code} - {company_name} created successfully!")
            if create_accounts:
                st.info("📊 Basic chart of accounts created.")
            st.rerun()
        except Exception as e:
            st.error(f"Error creating company: {str(e)}")

def render_audit_trail(company: CompanyContext, user: UserContext):
    st.header("🔍 Audit Trail")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("From Date", value=dt.date.today() - dt.timedelta(days=7))
    with col2:
        end_date = st.date_input("To Date", value=dt.date.today())
    with col3:
        action_filter = st.text_input("Action Filter", placeholder="VOUCHER_POSTED")

    with engine.begin() as conn:
        where_conditions = ["company_id = :company_id", "DATE(timestamp) BETWEEN :start_date AND :end_date"]
        params = {"company_id": company.id, "start_date": start_date, "end_date": end_date}
        if action_filter:
            where_conditions.append("action LIKE :action")
            params["action"] = f"%{action_filter}%"

        audit_query = text(f"""
            SELECT
                a.timestamp, u.username, a.action, a.table_name,
                a.record_id, a.ip_address
            FROM audit_log a
            LEFT JOIN users u ON a.user_id = u.id
            WHERE {' AND '.join(where_conditions)}
            ORDER BY a.timestamp DESC
            LIMIT 500
        """)
        audit_df = pd.read_sql(audit_query, conn, params=params)

    if not audit_df.empty:
        st.metric("Total Events", len(audit_df))
        display_df = audit_df.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"audit_trail_{company.code}_{start_date}_{end_date}.csv"
        )
    else:
        st.info("No audit events found for the selected criteria.")

# ============================
# MAIN APP
# ============================
def main():
    st.set_page_config(
        page_title="Enterprise Accounting",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_TITLE}</h1>
        <p>Version {VERSION} | Advanced Enterprise Accounting Solution (INR)</p>
    </div>
    """, unsafe_allow_html=True)

    # Authentication
    if not check_advanced_auth():
        st.stop()

    user = st.session_state["user_context"]

    # If no companies exist yet, force setup wizard
    if _company_count() == 0:
        render_setup_wizard(user)
        st.stop()

    # Company selection
    companies_list = get_user_companies(user)
    if not companies_list:
        st.error("No companies accessible. Please contact your administrator.")
        st.stop()

    with st.sidebar:
        st.markdown(f"### 👤 {user.username}")
        st.caption(f"Role: {user.role.value}")
        if st.button("🚪 Logout"):
            del st.session_state["user_context"]
            st.rerun()

        st.divider()
        company_names = [f"{c.code} - {c.name}" for c in companies_list]
        selected_idx = st.selectbox("Select Company", range(len(companies_list)),
                                    format_func=lambda x: company_names[x])
        current_company = companies_list[selected_idx]

        st.divider()
        if user.role in [UserRole.ADMIN, UserRole.ACCOUNTANT]:
            pages = [
                "📊 Dashboard",
                "📝 Multi-Line Voucher",
                "📊 Financial Statements",
                "📑 Ledger Statement",
                "🧮 Trial Balance",
                "📜 Journal Browser",
                "🏗️ Chart of Accounts",
                "👥 User Management",
                "🏢 Company Management",
                "🔍 Audit Trail"
            ]
        else:
            pages = [
                "📊 Dashboard",
                "📊 Financial Statements",
                "📑 Ledger Statement",
                "🧮 Trial Balance",
                "📜 Journal Browser"
            ]
        selected_page = st.radio("Navigation", pages)
        st.divider()
        st.caption(f"Database: {DB_URL.split('://')[-1]}")

    # Clear caches when company changes
    if 'current_company_id' not in st.session_state or st.session_state.current_company_id != current_company.id:
        st.session_state.current_company_id = current_company.id
        st.cache_data.clear()

    # Routing
    if selected_page == "📊 Dashboard":
        render_dashboard(current_company, user)
    elif selected_page == "📝 Multi-Line Voucher":
        render_multiline_voucher_entry(current_company, user)
    elif selected_page == "📊 Financial Statements":
        render_financial_statements(current_company)
    elif selected_page == "📑 Ledger Statement":
        render_ledger_statement(current_company)
    elif selected_page == "🧮 Trial Balance":
        render_trial_balance(current_company)
    elif selected_page == "📜 Journal Browser":
        render_journal_browser(current_company)
    elif selected_page == "🏗️ Chart of Accounts":
        render_chart_of_accounts(current_company, user)
    elif selected_page == "👥 User Management" and user.role == UserRole.ADMIN:
        render_user_management(current_company, user)
    elif selected_page == "🏢 Company Management" and user.role == UserRole.ADMIN:
        render_company_management(user)
    elif selected_page == "🔍 Audit Trail":
        render_audit_trail(current_company, user)

if __name__ == "__main__":
    main()
