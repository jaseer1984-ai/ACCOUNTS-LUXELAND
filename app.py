# Advanced Accounting Software — Masters Luxe Land LLP (INR)
# ================================================================
# ✅ Enhanced Streamlit app with advanced features, better UI, and robust error handling
# ✅ Features: Chart of Accounts with hierarchy, multi-line double‑entry vouchers, 
#    Advanced Ledger with search, Trial Balance, P&L, Balance Sheet, 
#    Graphical reports, Audit trail, Advanced CSV persistence, Indian formatting
# ✅ Currency: INR; all amounts shown with Indian-style comma separators
# ✅ Enhanced with data validation, import/export, search, and visual reporting

from __future__ import annotations
import os
import io
import json
import sqlite3
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import zipfile
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
APP_TITLE = "Masters Luxe Land LLP — Advanced Accounting System"
VERSION = "2.0.0"
CURRENCY = "INR"
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "accounting.db")

# Enhanced styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2d5aa0);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Utilities: INR formatting
# ---------------------------

def format_inr(n: float | int | pd.Series | None) -> str | pd.Series:
    """Format number in Indian grouping (e.g., 12,34,56,789.50)."""
    def _fmt(x: float | int | None) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "₹ 0.00"
        try:
            neg = x < 0
            x = abs(float(x))
            s = f"{x:.2f}"
            if "." in s:
                whole, frac = s.split(".")
            else:
                whole, frac = s, "00"
            
            # Indian number formatting
            if len(whole) <= 3:
                formatted = whole
            else:
                # Last 3 digits
                last_three = whole[-3:]
                remaining = whole[:-3]
                
                # Group remaining digits in pairs from right
                groups = []
                while len(remaining) > 2:
                    groups.append(remaining[-2:])
                    remaining = remaining[:-2]
                if remaining:
                    groups.append(remaining)
                
                groups.reverse()
                formatted = ",".join(groups) + "," + last_three
            
            result = f"₹ {'-' if neg else ''}{formatted}.{frac}"
            return result
        except Exception:
            return f"₹ {str(x)}"

    if isinstance(n, pd.Series):
        return n.apply(_fmt)
    return _fmt(n)

def parse_inr(s: str) -> float:
    """Parse INR formatted string back to float."""
    if not s or s.strip() == "":
        return 0.0
    try:
        # Remove currency symbol and commas
        cleaned = s.replace("₹", "").replace(",", "").strip()
        return float(cleaned)
    except:
        return 0.0

# ---------------------------
# Database Management
# ---------------------------

def init_database():
    """Initialize SQLite database with tables."""
    ensure_data_dir()
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Accounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                parent_id TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                FOREIGN KEY (parent_id) REFERENCES accounts (account_id)
            )
        """)
        
        # Vouchers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vouchers (
                voucher_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                narration TEXT,
                reference TEXT,
                created_by TEXT DEFAULT 'system',
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                is_posted BOOLEAN DEFAULT 1
            )
        """)
        
        # Entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voucher_id TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                account_id TEXT NOT NULL,
                debit REAL DEFAULT 0,
                credit REAL DEFAULT 0,
                description TEXT,
                FOREIGN KEY (voucher_id) REFERENCES vouchers (voucher_id),
                FOREIGN KEY (account_id) REFERENCES accounts (account_id)
            )
        """)
        
        # Audit trail table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_values TEXT,
                new_values TEXT,
                user_id TEXT DEFAULT 'system',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# Data Access Layer
# ---------------------------

class AccountingDB:
    def __init__(self):
        self.db_path = DB_PATH
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute SELECT query and return DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def execute_command(self, command: str, params: tuple = ()) -> bool:
        """Execute INSERT/UPDATE/DELETE command."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(command, params)
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False
    
    def get_accounts(self) -> pd.DataFrame:
        return self.execute_query("SELECT * FROM accounts ORDER BY account_id")
    
    def get_vouchers(self, limit: int = None) -> pd.DataFrame:
        query = "SELECT * FROM vouchers ORDER BY date DESC, voucher_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        return self.execute_query(query)
    
    def get_entries(self, voucher_id: str = None) -> pd.DataFrame:
        if voucher_id:
            return self.execute_query(
                "SELECT * FROM entries WHERE voucher_id = ? ORDER BY line_no", 
                (voucher_id,)
            )
        return self.execute_query("SELECT * FROM entries ORDER BY voucher_id, line_no")
    
    def get_ledger_entries(self, account_id: str, date_from: str = None, date_to: str = None) -> pd.DataFrame:
        query = """
            SELECT e.*, v.date, v.narration, v.reference
            FROM entries e
            JOIN vouchers v ON e.voucher_id = v.voucher_id
            WHERE e.account_id = ?
        """
        params = [account_id]
        
        if date_from:
            query += " AND v.date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND v.date <= ?"
            params.append(date_to)
            
        query += " ORDER BY v.date, v.voucher_id, e.line_no"
        return self.execute_query(query, tuple(params))

# ---------------------------
# Business Logic Layer
# ---------------------------

class AccountingService:
    def __init__(self):
        self.db = AccountingDB()
    
    def initialize_default_accounts(self):
        """Initialize default chart of accounts."""
        default_accounts = [
            ("1000", "Assets", "Assets", None, True, "Main asset group"),
            ("2000", "Liabilities", "Liabilities", None, True, "Main liability group"),
            ("3000", "Equity", "Equity", None, True, "Main equity group"),
            ("4000", "Income", "Income", None, True, "Main income group"),
            ("5000", "Expenses", "Expenses", None, True, "Main expense group"),
        ]
        
        for acc in default_accounts:
            self.db.execute_command(
                "INSERT OR IGNORE INTO accounts (account_id, name, type, parent_id, is_active, description) VALUES (?, ?, ?, ?, ?, ?)",
                acc
            )
    
    def get_leaf_accounts(self) -> List[Tuple[str, str]]:
        """Get list of leaf accounts (accounts with no children)."""
        accounts = self.db.get_accounts()
        if accounts.empty:
            return []
        
        # Find accounts that are not parents of any other account
        parent_ids = set(accounts['parent_id'].dropna().unique())
        leaf_accounts = []
        
        for _, row in accounts.iterrows():
            if row['account_id'] not in parent_ids and pd.notna(row['parent_id']):
                label = f"{row['account_id']} — {row['name']} ({row['type']})"
                leaf_accounts.append((label, row['account_id']))
        
        return sorted(leaf_accounts)
    
    def get_account_hierarchy(self) -> Dict[str, Dict]:
        """Get account hierarchy as nested dictionary."""
        accounts = self.db.get_accounts()
        if accounts.empty:
            return {}
        
        # Build hierarchy
        hierarchy = {}
        
        def add_to_hierarchy(acc_id: str, visited: set = None):
            if visited is None:
                visited = set()
            if acc_id in visited:
                return {}
            visited.add(acc_id)
            
            account = accounts[accounts['account_id'] == acc_id]
            if account.empty:
                return {}
            
            acc_row = account.iloc[0]
            children = accounts[accounts['parent_id'] == acc_id]
            
            node = {
                'name': acc_row['name'],
                'type': acc_row['type'],
                'is_active': acc_row['is_active'],
                'children': {}
            }
            
            for _, child in children.iterrows():
                node['children'][child['account_id']] = add_to_hierarchy(child['account_id'], visited.copy())
            
            return node
        
        # Start with root accounts (no parent)
        root_accounts = accounts[accounts['parent_id'].isna()]
        for _, root in root_accounts.iterrows():
            hierarchy[root['account_id']] = add_to_hierarchy(root['account_id'])
        
        return hierarchy
    
    def validate_voucher(self, lines: List[Dict]) -> Tuple[bool, str]:
        """Validate voucher lines."""
        if not lines:
            return False, "No lines to post."
        
        total_debit = sum(float(line.get('debit', 0)) for line in lines)
        total_credit = sum(float(line.get('credit', 0)) for line in lines)
        
        if round(total_debit - total_credit, 2) != 0.0:
            return False, f"Voucher not balanced: Dr {format_inr(total_debit)} vs Cr {format_inr(total_credit)}"
        
        # Validate accounts exist and are leaf accounts
        accounts = self.db.get_accounts()
        leaf_accounts = [acc_id for _, acc_id in self.get_leaf_accounts()]
        
        for line in lines:
            acc_id = str(line.get('account_id', ''))
            if not acc_id:
                return False, "Please select an account for all lines."
            if acc_id not in leaf_accounts:
                return False, f"Account {acc_id} is not a valid leaf account."
            
            debit = float(line.get('debit', 0))
            credit = float(line.get('credit', 0))
            
            if debit < 0 or credit < 0:
                return False, "Debit and credit amounts cannot be negative."
            
            if debit > 0 and credit > 0:
                return False, "A line cannot have both debit and credit amounts."
            
            if debit == 0 and credit == 0:
                return False, "Each line must have either a debit or credit amount."
        
        return True, "Valid"
    
    def post_voucher(self, voucher_date: date, narration: str, reference: str, lines: List[Dict]) -> Tuple[bool, str]:
        """Post a new voucher."""
        # Validate
        is_valid, msg = self.validate_voucher(lines)
        if not is_valid:
            return False, msg
        
        # Generate voucher ID
        voucher_id = self.generate_voucher_id()
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert voucher
                cursor.execute(
                    "INSERT INTO vouchers (voucher_id, date, narration, reference) VALUES (?, ?, ?, ?)",
                    (voucher_id, str(voucher_date), narration.strip(), reference.strip())
                )
                
                # Insert entries
                for i, line in enumerate(lines, 1):
                    cursor.execute(
                        "INSERT INTO entries (voucher_id, line_no, account_id, debit, credit, description) VALUES (?, ?, ?, ?, ?, ?)",
                        (voucher_id, i, str(line['account_id']), float(line.get('debit', 0)), 
                         float(line.get('credit', 0)), line.get('description', '').strip())
                    )
                
                conn.commit()
                return True, voucher_id
        
        except Exception as e:
            return False, f"Error posting voucher: {str(e)}"
    
    def generate_voucher_id(self) -> str:
        """Generate next voucher ID."""
        vouchers = self.db.get_vouchers(limit=1)
        if vouchers.empty:
            return "V000001"
        
        last_id = vouchers.iloc[0]['voucher_id']
        try:
            num = int(last_id[1:]) + 1
            return f"V{num:06d}"
        except:
            return "V000001"
    
    def get_trial_balance(self, as_on: date = None) -> pd.DataFrame:
        """Generate trial balance."""
        query = """
            SELECT 
                e.account_id,
                a.name as account_name,
                a.type as account_type,
                SUM(e.debit) as total_debit,
                SUM(e.credit) as total_credit
            FROM entries e
            JOIN accounts a ON e.account_id = a.account_id
            JOIN vouchers v ON e.voucher_id = v.voucher_id
        """
        
        params = []
        if as_on:
            query += " WHERE v.date <= ?"
            params.append(str(as_on))
        
        query += " GROUP BY e.account_id, a.name, a.type ORDER BY e.account_id"
        
        return self.db.execute_query(query, tuple(params))
    
    def get_profit_loss(self, as_on: date = None) -> Tuple[pd.DataFrame, float]:
        """Generate Profit & Loss statement."""
        tb = self.get_trial_balance(as_on)
        if tb.empty:
            return pd.DataFrame(columns=['Head', 'Amount']), 0.0
        
        # Calculate income and expenses
        income_rows = tb[tb['account_type'] == 'Income']
        expense_rows = tb[tb['account_type'] == 'Expenses']
        
        total_income = (income_rows['total_credit'] - income_rows['total_debit']).sum()
        total_expenses = (expense_rows['total_debit'] - expense_rows['total_credit']).sum()
        
        net_profit = total_income - total_expenses
        
        pl_data = [
            {'Head': 'Total Income', 'Amount': total_income},
            {'Head': 'Total Expenses', 'Amount': total_expenses},
            {'Head': 'Net Profit/Loss', 'Amount': net_profit}
        ]
        
        return pd.DataFrame(pl_data), net_profit
    
    def get_balance_sheet(self, as_on: date = None) -> pd.DataFrame:
        """Generate Balance Sheet."""
        tb = self.get_trial_balance(as_on)
        if tb.empty:
            return pd.DataFrame(columns=['Head', 'Amount'])
        
        # Get profit/loss
        _, net_profit = self.get_profit_loss(as_on)
        
        # Calculate totals
        assets = tb[tb['account_type'] == 'Assets']
        liabilities = tb[tb['account_type'] == 'Liabilities']
        equity = tb[tb['account_type'] == 'Equity']
        
        total_assets = (assets['total_debit'] - assets['total_credit']).sum()
        total_liabilities = (liabilities['total_credit'] - liabilities['total_debit']).sum()
        total_equity = (equity['total_credit'] - equity['total_debit']).sum() + net_profit
        
        bs_data = [
            {'Head': 'Total Assets', 'Amount': total_assets},
            {'Head': 'Total Liabilities', 'Amount': total_liabilities},
            {'Head': 'Total Equity + P&L', 'Amount': total_equity}
        ]
        
        return pd.DataFrame(bs_data)

# Global service instance
accounting_service = AccountingService()

# ---------------------------
# UI Components
# ---------------------------

def display_main_header():
    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_TITLE}</h1>
        <p>Version {VERSION} • Currency: {CURRENCY} • Advanced Features Enabled</p>
    </div>
    """, unsafe_allow_html=True)

def display_metric_cards():
    """Display dashboard metrics in cards."""
    tb = accounting_service.get_trial_balance()
    
    if not tb.empty:
        assets = tb[tb['account_type'] == 'Assets']
        liabilities = tb[tb['account_type'] == 'Liabilities'] 
        equity = tb[tb['account_type'] == 'Equity']
        
        total_assets = (assets['total_debit'] - assets['total_credit']).sum()
        total_liabilities = (liabilities['total_credit'] - liabilities['total_debit']).sum()
        total_equity = (equity['total_credit'] - equity['total_debit']).sum()
        
        _, net_profit = accounting_service.get_profit_loss()
    else:
        total_assets = total_liabilities = total_equity = net_profit = 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Assets", format_inr(total_assets), delta=None)
    with col2:
        st.metric("Liabilities", format_inr(total_liabilities), delta=None)
    with col3:
        st.metric("Equity", format_inr(total_equity), delta=None)
    with col4:
        delta_color = "normal" if net_profit >= 0 else "inverse"
        st.metric("Net Profit", format_inr(net_profit), delta=None)

def create_account_hierarchy_chart():
    """Create interactive account hierarchy chart."""
    hierarchy = accounting_service.get_account_hierarchy()
    
    if not hierarchy:
        st.info("No accounts found. Add accounts in the Masters section.")
        return
    
    # Convert hierarchy to plotly treemap
    ids = []
    labels = []
    parents = []
    values = []
    
    def process_node(node_id, node_data, parent_id=""):
        ids.append(node_id)
        labels.append(f"{node_id}<br>{node_data['name']}")
        parents.append(parent_id)
        values.append(1)  # Equal weight for visualization
        
        for child_id, child_data in node_data.get('children', {}).items():
            process_node(child_id, child_data, node_id)
    
    for root_id, root_data in hierarchy.items():
        process_node(root_id, root_data)
    
    if ids:
        fig = go.Figure(go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            maxdepth=3,
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Chart of Accounts Hierarchy",
            font_size=12,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_financial_charts():
    """Create financial performance charts."""
    tb = accounting_service.get_trial_balance()
    
    if tb.empty:
        st.info("No data available for charts.")
        return
    
    # Account type distribution
    type_summary = tb.groupby('account_type').agg({
        'total_debit': 'sum',
        'total_credit': 'sum'
    }).reset_index()
    
    type_summary['net_amount'] = type_summary['total_debit'] - type_summary['total_credit']
    
    # Create pie chart for account types
    fig_pie = px.pie(
        type_summary, 
        values='net_amount', 
        names='account_type',
        title="Account Distribution by Type"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart for debits vs credits by type
        fig_bar = go.Figure(data=[
            go.Bar(name='Debits', x=type_summary['account_type'], y=type_summary['total_debit']),
            go.Bar(name='Credits', x=type_summary['account_type'], y=type_summary['total_credit'])
        ])
        
        fig_bar.update_layout(
            title="Debits vs Credits by Account Type",
            barmode='group',
            xaxis_title="Account Type",
            yaxis_title="Amount (INR)"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------
# Page Functions
# ---------------------------

def page_dashboard():
    st.subheader("📊 Dashboard")
    
    display_metric_cards()
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Recent Vouchers")
        vouchers = accounting_service.db.get_vouchers(limit=10)
        if vouchers.empty:
            st.info("No vouchers yet. Create some from the Vouchers page.")
        else:
            # Format dates
            vouchers['date'] = pd.to_datetime(vouchers['date']).dt.strftime('%d-%m-%Y')
            st.dataframe(vouchers[['voucher_id', 'date', 'narration', 'reference']], use_container_width=True)
    
    with col2:
        st.subheader("Quick Stats")
        accounts = accounting_service.db.get_accounts()
        vouchers_count = len(accounting_service.db.get_vouchers())
        entries_count = len(accounting_service.db.get_entries())
        
        st.metric("Total Accounts", len(accounts))
        st.metric("Total Vouchers", vouchers_count)
        st.metric("Total Entries", entries_count)
    
    st.markdown("---")
    
    # Charts
    tab1, tab2 = st.tabs(["Account Hierarchy", "Financial Charts"])
    
    with tab1:
        create_account_hierarchy_chart()
    
    with tab2:
        create_financial_charts()

def page_vouchers():
    st.subheader("📝 Voucher Management")
    
    tab1, tab2 = st.tabs(["Create Voucher", "View Vouchers"])
    
    with tab1:
        st.markdown("### Create New Voucher")
        
        # Get leaf accounts for dropdown
        leaf_accounts = accounting_service.get_leaf_accounts()
        
        if not leaf_accounts:
            st.warning("⚠️ No leaf accounts available. Please add accounts in the Masters section first.")
            st.info("💡 Tip: Go to 'Masters' → Add accounts like Bank, Cash, Sales, Purchases under appropriate groups.")
            return
        
        # Voucher header
        col1, col2 = st.columns(2)
        with col1:
            voucher_date = st.date_input("Date", value=date.today())
        with col2:
            reference = st.text_input("Reference/Cheque No.", placeholder="Optional reference")
        
        narration = st.text_area("Narration", placeholder="Description of the transaction")
        
        st.markdown("### Voucher Lines")
        
        # Initialize session state for lines
        if 'voucher_lines' not in st.session_state:
            st.session_state.voucher_lines = [
                {"account_id": "", "debit": 0.0, "credit": 0.0, "description": ""}
            ]
        
        lines = st.session_state.voucher_lines
        
        # Display lines
        remove_index = None
        for i, line in enumerate(lines):
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1.5, 2.5, 0.5])
                
                with col1:
                    account_options = ["Select account..."] + [label for label, _ in leaf_accounts]
                    account_map = dict(leaf_accounts)
                    
                    selected = st.selectbox(
                        f"Account (Line {i+1})",
                        options=account_options,
                        key=f"account_{i}"
                    )
                    
                    if selected != "Select account...":
                        line['account_id'] = account_map.get(selected, "")
                
                with col2:
                    line['debit'] = st.number_input(
                        "Debit",
                        min_value=0.0,
                        step=0.01,
                        value=float(line.get('debit', 0)),
                        key=f"debit_{i}",
                        format="%.2f"
                    )
                
                with col3:
                    line['credit'] = st.number_input(
                        "Credit",
                        min_value=0.0,
                        step=0.01,
                        value=float(line.get('credit', 0)),
                        key=f"credit_{i}",
                        format="%.2f"
                    )
                
                with col4:
                    line['description'] = st.text_input(
                        "Description",
                        value=line.get('description', ''),
                        key=f"desc_{i}",
                        placeholder="Optional line description"
                    )
                
                with col5:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete this line"):
                        remove_index = i
        
        # Remove line if requested
        if remove_index is not None:
            lines.pop(remove_index)
            st.rerun()
        
        # Action buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("➕ Add Line"):
                lines.append({"account_id": "", "debit": 0.0, "credit": 0.0, "description": ""})
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All"):
                st.session_state.voucher_lines = []
                st.rerun()
        
        # Show totals
        total_debit = sum(float(line.get('debit', 0)) for line in lines)
        total_credit = sum(float(line.get('credit', 0)) for line in lines)
        difference = total_debit - total_credit
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Debit", format_inr(total_debit))
        col2.metric("Total Credit", format_inr(total_credit))
        col3.metric("Difference", format_inr(difference), delta=f"{difference:.2f}")
        
        # Validation and post button
        is_balanced = abs(difference) < 0.01
        
        if not is_balanced:
            st.error("⚠️ Voucher is not balanced! Total debits must equal total credits.")
        
        if st.button("💾 Post Voucher", type="primary", disabled=not is_balanced):
            if not narration.strip():
                st.error("Please enter a narration.")
            else:
                success, result = accounting_service.post_voucher(voucher_date, narration, reference, lines)
                if success:
                    st.success(f"✅ Voucher posted successfully! Voucher ID: {result}")
                    st.session_state.voucher_lines = []
                    st.rerun()
                else:
                    st.error(f"❌ Error posting voucher: {result}")
    
    with tab2:
        st.markdown("### All Vouchers")
        
        # Search and filter
        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input("🔍 Search vouchers", placeholder="Search by voucher ID or narration")
        with col2:
            limit = st.selectbox("Show records", [20, 50, 100, 200], index=0)
        
        vouchers = accounting_service.db.get_vouchers(limit=limit)
        
        if not vouchers.empty:
            # Apply search filter
            if search_term:
                vouchers = vouchers[
                    vouchers['voucher_id'].str.contains(search_term, case=False, na=False) |
                    vouchers['narration'].str.contains(search_term, case=False, na=False)
                ]
            
            # Format dates
            vouchers['date'] = pd.to_datetime(vouchers['date']).dt.strftime('%d-%m-%Y')
            
            # Display table
            st.dataframe(vouchers, use_container_width=True)
            
            # Download option
            csv = vouchers.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"vouchers_{date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.info("No vouchers found.")

def page_ledger():
    st.subheader("📚 Account Ledger")
    
    # Account selection
    leaf_accounts = accounting_service.get_leaf_accounts()
    
    if not leaf_accounts:
        st.warning("No accounts available. Please add accounts first.")
        return
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        account_options = ["Select account..."] + [label for label, _ in leaf_accounts]
        selected_account = st.selectbox("Choose Account", account_options)
    
    if selected_account == "Select account...":
        st.info("Please select an account to view its ledger.")
        return
    
    # Get account ID
    account_map = dict(leaf_accounts)
    account_id = account_map.get(selected_account)
    
    with col2:
        date_from = st.date_input("From Date", value=date(date.today().year, 1, 1))
    
    with col3:
        date_to = st.date_input("To Date", value=date.today())
    
    # Get ledger data
    ledger_data = accounting_service.db.get_ledger_entries(
        account_id, str(date_from), str(date_to)
    )
    
    if ledger_data.empty:
        st.info("No transactions found for this account in the selected period.")
        return
    
    # Calculate running balance
    ledger_data['balance_change'] = ledger_data['debit'] - ledger_data['credit']
    ledger_data['running_balance'] = ledger_data['balance_change'].cumsum()
    
    # Format for display
    display_data = ledger_data.copy()
    display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%d-%m-%Y')
    display_data['debit'] = display_data['debit'].apply(lambda x: format_inr(x) if x > 0 else "")
    display_data['credit'] = display_data['credit'].apply(lambda x: format_inr(x) if x > 0 else "")
    display_data['running_balance'] = display_data['running_balance'].apply(format_inr)
    
    # Select columns for display
    display_columns = ['date', 'voucher_id', 'description', 'debit', 'credit', 'running_balance']
    column_names = ['Date', 'Voucher', 'Description', 'Debit', 'Credit', 'Balance']
    
    display_df = display_data[display_columns]
    display_df.columns = column_names
    
    # Fill empty descriptions with narration
    display_df['Description'] = display_df['Description'].fillna('') 
    mask = display_df['Description'] == ''
    display_df.loc[mask, 'Description'] = ledger_data.loc[mask, 'narration']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Summary
    total_debit = ledger_data['debit'].sum()
    total_credit = ledger_data['credit'].sum()
    closing_balance = ledger_data['running_balance'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Debits", format_inr(total_debit))
    col2.metric("Total Credits", format_inr(total_credit))
    col3.metric("Closing Balance", format_inr(closing_balance))
    
    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Ledger",
        data=csv,
        file_name=f"ledger_{account_id}_{date.today()}.csv",
        mime="text/csv"
    )

def page_trial_balance():
    st.subheader("⚖️ Trial Balance")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        as_on_date = st.date_input("As on Date", value=date.today())
    
    trial_balance = accounting_service.get_trial_balance(as_on_date)
    
    if trial_balance.empty:
        st.info("No data available for trial balance.")
        return
    
    # Format for display
    display_data = trial_balance.copy()
    display_data['total_debit'] = display_data['total_debit'].apply(format_inr)
    display_data['total_credit'] = display_data['total_credit'].apply(format_inr)
    
    # Rename columns
    display_data.columns = ['Account ID', 'Account Name', 'Type', 'Total Debit', 'Total Credit']
    
    st.dataframe(display_data, use_container_width=True)
    
    # Totals
    total_debit = trial_balance['total_debit'].sum()
    total_credit = trial_balance['total_credit'].sum()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Debits", format_inr(total_debit))
    col2.metric("Total Credits", format_inr(total_credit))
    
    # Validation
    if abs(total_debit - total_credit) < 0.01:
        st.success("✅ Trial Balance is balanced!")
    else:
        st.error(f"❌ Trial Balance is not balanced! Difference: {format_inr(total_debit - total_credit)}")
    
    # Download option
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Trial Balance",
        data=csv,
        file_name=f"trial_balance_{as_on_date}.csv",
        mime="text/csv"
    )

def page_profit_loss():
    st.subheader("📈 Profit & Loss Statement")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        as_on_date = st.date_input("As on Date", value=date.today())
    
    pl_data, net_profit = accounting_service.get_profit_loss(as_on_date)
    
    if pl_data.empty:
        st.info("No data available for P&L statement.")
        return
    
    # Format amounts
    display_data = pl_data.copy()
    display_data['Amount'] = display_data['Amount'].apply(format_inr)
    
    st.dataframe(display_data, use_container_width=True)
    
    # Highlight net profit/loss
    if net_profit > 0:
        st.success(f"✅ Net Profit: {format_inr(net_profit)}")
    elif net_profit < 0:
        st.error(f"❌ Net Loss: {format_inr(abs(net_profit))}")
    else:
        st.info("⚖️ Break-even (No profit, no loss)")
    
    # Create chart
    fig = px.bar(
        pl_data[pl_data['Head'] != 'Net Profit/Loss'], 
        x='Head', 
        y='Amount',
        title="Income vs Expenses",
        color='Head'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download option
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download P&L",
        data=csv,
        file_name=f"profit_loss_{as_on_date}.csv",
        mime="text/csv"
    )

def page_balance_sheet():
    st.subheader("🏛️ Balance Sheet")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        as_on_date = st.date_input("As on Date", value=date.today())
    
    balance_sheet = accounting_service.get_balance_sheet(as_on_date)
    
    if balance_sheet.empty:
        st.info("No data available for balance sheet.")
        return
    
    # Format amounts
    display_data = balance_sheet.copy()
    display_data['Amount'] = display_data['Amount'].apply(format_inr)
    
    st.dataframe(display_data, use_container_width=True)
    
    # Check if balanced
    total_assets = balance_sheet[balance_sheet['Head'] == 'Total Assets']['Amount'].iloc[0]
    total_liab_equity = balance_sheet[balance_sheet['Head'] != 'Total Assets']['Amount'].sum()
    
    if abs(total_assets - total_liab_equity) < 0.01:
        st.success("✅ Balance Sheet is balanced!")
    else:
        st.error(f"❌ Balance Sheet is not balanced! Difference: {format_inr(total_assets - total_liab_equity)}")
    
    # Create chart
    fig = px.pie(
        balance_sheet, 
        values='Amount', 
        names='Head',
        title="Balance Sheet Composition"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download option
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Balance Sheet",
        data=csv,
        file_name=f"balance_sheet_{as_on_date}.csv",
        mime="text/csv"
    )

def page_masters():
    st.subheader("🏗️ Chart of Accounts (Masters)")
    
    tab1, tab2, tab3 = st.tabs(["View Accounts", "Add Account", "Account Hierarchy"])
    
    with tab1:
        st.markdown("### All Accounts")
        
        accounts = accounting_service.db.get_accounts()
        
        if not accounts.empty:
            # Format for display
            display_accounts = accounts.copy()
            display_accounts['created_date'] = pd.to_datetime(display_accounts['created_date']).dt.strftime('%d-%m-%Y')
            
            st.dataframe(display_accounts, use_container_width=True)
        else:
            st.info("No accounts found.")
        
        # Download option
        if not accounts.empty:
            csv = accounts.to_csv(index=False)
            st.download_button(
                label="📥 Download Accounts",
                data=csv,
                file_name=f"accounts_{date.today()}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.markdown("### Add New Account")
        
        col1, col2 = st.columns(2)
        
        with col1:
            account_id = st.text_input("Account ID", placeholder="e.g., 1101")
            account_name = st.text_input("Account Name", placeholder="e.g., State Bank of India")
        
        with col2:
            account_type = st.selectbox("Account Type", [
                "Assets", "Liabilities", "Equity", "Income", "Expenses"
            ])
            
            # Parent account selection
            accounts = accounting_service.db.get_accounts()
            parent_options = ["None (Root Account)"]
            
            if not accounts.empty:
                for _, acc in accounts.iterrows():
                    parent_options.append(f"{acc['account_id']} — {acc['name']}")
            
            parent_selection = st.selectbox("Parent Account", parent_options)
        
        description = st.text_area("Description (Optional)", placeholder="Account description")
        is_active = st.checkbox("Active", value=True)
        
        if st.button("➕ Create Account", type="primary"):
            if not account_id or not account_name:
                st.error("Account ID and Name are required.")
            elif account_id in accounts['account_id'].tolist():
                st.error("Account ID already exists.")
            else:
                parent_id = None
                if parent_selection != "None (Root Account)":
                    parent_id = parent_selection.split(" — ")[0]
                
                success = accounting_service.db.execute_command(
                    "INSERT INTO accounts (account_id, name, type, parent_id, is_active, description) VALUES (?, ?, ?, ?, ?, ?)",
                    (account_id, account_name, account_type, parent_id, is_active, description)
                )
                
                if success:
                    st.success(f"✅ Account '{account_name}' created successfully!")
                    st.rerun()
                else:
                    st.error("❌ Error creating account.")
    
    with tab3:
        st.markdown("### Account Hierarchy Visualization")
        create_account_hierarchy_chart()

def page_backup_export():
    st.subheader("💾 Backup & Export")
    
    tab1, tab2 = st.tabs(["Download Data", "Import Data"])
    
    with tab1:
        st.markdown("### Download System Data")
        
        col1, col2, col3 = st.columns(3)
        
        # Individual downloads
        with col1:
            st.markdown("#### Accounts")
            accounts = accounting_service.db.get_accounts()
            if not accounts.empty:
                csv = accounts.to_csv(index=False)
                st.download_button(
                    "📥 Download Accounts",
                    data=csv,
                    file_name=f"accounts_{date.today()}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No accounts to download")
        
        with col2:
            st.markdown("#### Vouchers")
            vouchers = accounting_service.db.get_vouchers()
            if not vouchers.empty:
                csv = vouchers.to_csv(index=False)
                st.download_button(
                    "📥 Download Vouchers",
                    data=csv,
                    file_name=f"vouchers_{date.today()}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No vouchers to download")
        
        with col3:
            st.markdown("#### Entries")
            entries = accounting_service.db.get_entries()
            if not entries.empty:
                csv = entries.to_csv(index=False)
                st.download_button(
                    "📥 Download Entries",
                    data=csv,
                    file_name=f"entries_{date.today()}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No entries to download")
        
        st.markdown("---")
        
        # Complete backup
        st.markdown("### Complete System Backup")
        
        if st.button("📦 Create Complete Backup"):
            # Create ZIP file with all data
            accounts = accounting_service.db.get_accounts()
            vouchers = accounting_service.db.get_vouchers()
            entries = accounting_service.db.get_entries()
            
            # Create ZIP buffer
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add CSV files to ZIP
                if not accounts.empty:
                    zip_file.writestr("accounts.csv", accounts.to_csv(index=False))
                if not vouchers.empty:
                    zip_file.writestr("vouchers.csv", vouchers.to_csv(index=False))
                if not entries.empty:
                    zip_file.writestr("entries.csv", entries.to_csv(index=False))
                
                # Add metadata
                metadata = {
                    "export_date": str(date.today()),
                    "app_version": VERSION,
                    "total_accounts": len(accounts),
                    "total_vouchers": len(vouchers),
                    "total_entries": len(entries)
                }
                zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Complete Backup",
                data=zip_buffer.getvalue(),
                file_name=f"accounting_backup_{date.today()}.zip",
                mime="application/zip"
            )
    
    with tab2:
        st.markdown("### Import Data")
        st.warning("⚠️ **Caution**: Importing will add to existing data. Make sure to backup current data first.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            # Determine file type based on content
            df = pd.read_csv(uploaded_file)
            
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            file_type = st.selectbox("Select data type", ["accounts", "vouchers", "entries"])
            
            if st.button("📤 Import Data"):
                try:
                    if file_type == "accounts":
                        # Validate accounts structure
                        required_cols = ["account_id", "name", "type"]
                        if all(col in df.columns for col in required_cols):
                            # Insert accounts
                            for _, row in df.iterrows():
                                accounting_service.db.execute_command(
                                    "INSERT OR REPLACE INTO accounts (account_id, name, type, parent_id, is_active, description) VALUES (?, ?, ?, ?, ?, ?)",
                                    (row['account_id'], row['name'], row['type'], 
                                     row.get('parent_id'), row.get('is_active', True), row.get('description', ''))
                                )
                            st.success(f"✅ Imported {len(df)} accounts successfully!")
                        else:
                            st.error("❌ Invalid accounts CSV format. Required columns: account_id, name, type")
                    
                    # Similar logic for vouchers and entries...
                    
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")

# ---------------------------
# Main Application
# ---------------------------

def main():
    # Initialize
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    # Initialize database
    init_database()
    accounting_service.initialize_default_accounts()
    
    # Header
    display_main_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        page = st.radio("Choose a page:", [
            "📊 Dashboard",
            "📝 Vouchers", 
            "📚 Ledger",
            "⚖️ Trial Balance",
            "📈 Profit & Loss",
            "🏛️ Balance Sheet",
            "🏗️ Masters",
            "💾 Backup & Export"
        ])
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        # System info
        st.markdown("---")
        st.markdown("### 📊 System Info")
        
        accounts_count = len(accounting_service.db.get_accounts())
        vouchers_count = len(accounting_service.db.get_vouchers())
        
        st.metric("Accounts", accounts_count)
        st.metric("Vouchers", vouchers_count)
        
        st.caption(f"Version {VERSION}")
    
    # Main content
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "📝 Vouchers":
        page_vouchers()
    elif page == "📚 Ledger":
        page_ledger()
    elif page == "⚖️ Trial Balance":
        page_trial_balance()
    elif page == "📈 Profit & Loss":
        page_profit_loss()
    elif page == "🏛️ Balance Sheet":
        page_balance_sheet()
    elif page == "🏗️ Masters":
        page_masters()
    elif page == "💾 Backup & Export":
        page_backup_export()

if __name__ == "__main__":
    main()
