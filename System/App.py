#!/usr/bin/env python3
"""
Options Trading Analysis System - UI v3.0
==========================================
Features:
1. Market data collection and analysis
2. Judge LLM verdict visualization with improved payoff charts
3. Strategy Builder - Create and analyze custom strategies

Author: Options Trading System
Version: 3.0

Changes from v2.0:
- Removed LLM Reasoning comparison chart (not useful)
- Added much better interactive payoff visualization
- Added Strategy Builder tab for custom strategy creation
- Clear profit/loss zones with exact breakeven points
- P&L table showing profit at key price levels
"""

import streamlit as st
import json
import os
import sys
import io
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Import data collector
try:
    import data_collector_enhanced
except ImportError:
    st.error("Could not import data_collector_enhanced.py. Make sure it is in the same directory.")

try:
    import llm_runner
except ImportError:
    st.error("Could not import llm_runner.py. Make sure it is in the same directory.")

# ============================================================================

st.set_page_config(
    page_title="Options Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

LOT_SIZES = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
    "SENSEX": 10
}

DEFAULT_SPOT_PRICES = {
    "NIFTY": 24500,
    "BANKNIFTY": 52000,
    "FINNIFTY": 23500,
    "MIDCPNIFTY": 13000,
    "SENSEX": 81000
}

# Strategy Templates
STRATEGY_TEMPLATES = {
    "Iron Condor": {
        "description": "Sell OTM Call + Sell OTM Put, Buy further OTM options for protection",
        "legs": [
            {"action": "SELL", "option_type": "PE", "strike_offset": -200, "premium": 80},
            {"action": "BUY", "option_type": "PE", "strike_offset": -400, "premium": 40},
            {"action": "SELL", "option_type": "CE", "strike_offset": 200, "premium": 80},
            {"action": "BUY", "option_type": "CE", "strike_offset": 400, "premium": 40},
        ]
    },
    "Bull Put Spread": {
        "description": "Sell ATM/OTM Put, Buy lower strike Put",
        "legs": [
            {"action": "SELL", "option_type": "PE", "strike_offset": -100, "premium": 120},
            {"action": "BUY", "option_type": "PE", "strike_offset": -300, "premium": 50},
        ]
    },
    "Bear Call Spread": {
        "description": "Sell ATM/OTM Call, Buy higher strike Call",
        "legs": [
            {"action": "SELL", "option_type": "CE", "strike_offset": 100, "premium": 120},
            {"action": "BUY", "option_type": "CE", "strike_offset": 300, "premium": 50},
        ]
    },
    "Long Straddle": {
        "description": "Buy ATM Call + Buy ATM Put",
        "legs": [
            {"action": "BUY", "option_type": "CE", "strike_offset": 0, "premium": 150},
            {"action": "BUY", "option_type": "PE", "strike_offset": 0, "premium": 150},
        ]
    },
    "Short Straddle": {
        "description": "Sell ATM Call + Sell ATM Put",
        "legs": [
            {"action": "SELL", "option_type": "CE", "strike_offset": 0, "premium": 150},
            {"action": "SELL", "option_type": "PE", "strike_offset": 0, "premium": 150},
        ]
    },
    "Long Strangle": {
        "description": "Buy OTM Call + Buy OTM Put",
        "legs": [
            {"action": "BUY", "option_type": "CE", "strike_offset": 200, "premium": 80},
            {"action": "BUY", "option_type": "PE", "strike_offset": -200, "premium": 80},
        ]
    },
    "Short Strangle": {
        "description": "Sell OTM Call + Sell OTM Put",
        "legs": [
            {"action": "SELL", "option_type": "CE", "strike_offset": 200, "premium": 80},
            {"action": "SELL", "option_type": "PE", "strike_offset": -200, "premium": 80},
        ]
    },
    "Butterfly Spread": {
        "description": "Buy 1 ITM, Sell 2 ATM, Buy 1 OTM (same type)",
        "legs": [
            {"action": "BUY", "option_type": "CE", "strike_offset": -100, "premium": 180},
            {"action": "SELL", "option_type": "CE", "strike_offset": 0, "premium": 120},
            {"action": "SELL", "option_type": "CE", "strike_offset": 0, "premium": 120},
            {"action": "BUY", "option_type": "CE", "strike_offset": 100, "premium": 80},
        ]
    },
}

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0f1419;
        --bg-secondary: #1a1f2e;
        --bg-card: #232b3e;
        --border-color: #3d4a5c;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --text-muted: #718096;
        --accent-green: #00c853;
        --accent-red: #ff5252;
        --accent-blue: #2196f3;
        --accent-yellow: #ffc107;
        --radius-md: 12px;
        --radius-lg: 16px;
    }

    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 100%;
        background: var(--bg-primary);
    }

    #MainMenu, footer, header {visibility: hidden;}

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        color: var(--text-primary);
    }

    .main-header {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border-radius: var(--radius-lg);
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }

    .main-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #fff 0%, #a0aec0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .card {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        border: 1px solid var(--border-color);
    }

    .card-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .card-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .card-value.green { color: var(--accent-green); }
    .card-value.red { color: var(--accent-red); }
    .card-value.blue { color: var(--accent-blue); }

    .strategy-panel {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .strategy-panel h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }

    .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .data-table th {
        background: var(--bg-secondary);
        color: var(--text-muted);
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.7rem;
    }

    .data-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    .data-table .sell { color: var(--accent-red); font-weight: 600; }
    .data-table .buy { color: var(--accent-green); font-weight: 600; }

    .pnl-positive { color: var(--accent-green); font-weight: 600; }
    .pnl-negative { color: var(--accent-red); font-weight: 600; }

    .metric-box {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #1976d2 100%);
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: var(--radius-md);
        border: none;
        width: 100%;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 0.35rem;
        border: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.6rem 1.25rem;
        color: var(--text-secondary);
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-blue);
        color: white;
    }

    .verdict-banner {
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .verdict-banner.trade {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.2) 0%, rgba(0, 150, 60, 0.1) 100%);
        border: 2px solid var(--accent-green);
    }

    .verdict-banner.no-trade {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.2) 0%, rgba(200, 50, 50, 0.1) 100%);
        border: 2px solid var(--accent-red);
    }

    .verdict-text {
        font-size: 2.5rem;
        font-weight: 800;
    }

    .verdict-text.trade { color: var(--accent-green); }
    .verdict-text.no-trade { color: var(--accent-red); }

    .leg-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .leg-badge.buy { background: rgba(0, 200, 83, 0.2); color: var(--accent-green); }
    .leg-badge.sell { background: rgba(255, 82, 82, 0.2); color: var(--accent-red); }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_get(data: Any, *keys, default: Any = None) -> Any:
    result = data
    for key in keys:
        if result is None:
            return default
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return default
    return result if result is not None else default


def parse_currency_value(value: str, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        cleaned = str(value).replace('₹', '').replace(',', '').replace(' ', '').strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def escape_html(text: str) -> str:
    if text is None:
        return ''
    return str(text).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')


def get_lot_size(underlying: str) -> int:
    return LOT_SIZES.get(underlying.upper(), 25)


def get_output_directory(base_dir: str = "analysis_output") -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_dir, today)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================================
# PAYOFF CALCULATION ENGINE
# ============================================================================

class OptionsPayoffCalculator:
    """Calculate payoff for options strategies"""

    def __init__(self, spot_price: float, lot_size: int = 25):
        self.spot_price = spot_price
        self.lot_size = lot_size
        self.legs = []

    def add_leg(self, action: str, option_type: str, strike: float,
                premium: float, lots: int = 1):
        """Add a leg to the strategy"""
        self.legs.append({
            'action': action.upper(),
            'option_type': option_type.upper(),
            'strike': strike,
            'premium': premium,
            'lots': lots
        })

    def clear_legs(self):
        """Clear all legs"""
        self.legs = []

    def calculate_payoff(self, price_range: np.ndarray) -> np.ndarray:
        """Calculate P&L at each price point"""
        pnl = np.zeros_like(price_range, dtype=float)

        for leg in self.legs:
            strike = leg['strike']
            premium = leg['premium']
            action = leg['action']
            opt_type = leg['option_type']
            lots = leg['lots']
            multiplier = lots * self.lot_size

            for i, price in enumerate(price_range):
                # Calculate intrinsic value at expiry
                if opt_type == 'CE':
                    intrinsic = max(0, price - strike)
                else:  # PE
                    intrinsic = max(0, strike - price)

                # Calculate P&L based on position
                if action == 'BUY':
                    leg_pnl = (intrinsic - premium) * multiplier
                else:  # SELL
                    leg_pnl = (premium - intrinsic) * multiplier

                pnl[i] += leg_pnl

        return pnl

    def find_breakevens(self, pnl: np.ndarray, prices: np.ndarray) -> List[float]:
        """Find breakeven points where P&L crosses zero"""
        breakevens = []
        for i in range(1, len(pnl)):
            if (pnl[i - 1] < 0 and pnl[i] >= 0) or (pnl[i - 1] >= 0 and pnl[i] < 0):
                # Linear interpolation
                if pnl[i] != pnl[i - 1]:
                    be = prices[i - 1] + (0 - pnl[i - 1]) * (prices[i] - prices[i - 1]) / (pnl[i] - pnl[i - 1])
                    breakevens.append(round(be, 2))
        return sorted(set(breakevens))

    def get_strategy_metrics(self) -> dict:
        """Calculate key strategy metrics"""
        if not self.legs:
            return {}

        # Generate price range
        min_strike = min(leg['strike'] for leg in self.legs)
        max_strike = max(leg['strike'] for leg in self.legs)
        range_width = max(max_strike - min_strike, self.spot_price * 0.1)

        prices = np.linspace(
            min_strike - range_width,
            max_strike + range_width,
            1000
        )

        pnl = self.calculate_payoff(prices)

        # Find key metrics
        max_profit = float(np.max(pnl))
        max_loss = float(np.min(pnl))
        breakevens = self.find_breakevens(pnl, prices)

        # Net premium (credit/debit)
        net_premium = 0
        for leg in self.legs:
            if leg['action'] == 'SELL':
                net_premium += leg['premium'] * leg['lots'] * self.lot_size
            else:
                net_premium -= leg['premium'] * leg['lots'] * self.lot_size

        # P&L at current spot
        spot_idx = np.argmin(np.abs(prices - self.spot_price))
        pnl_at_spot = float(pnl[spot_idx])

        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakevens': breakevens,
            'net_premium': net_premium,
            'pnl_at_spot': pnl_at_spot,
            'is_credit': net_premium > 0,
            'prices': prices,
            'pnl': pnl
        }


# ============================================================================
# ADVANCED PAYOFF CHART
# ============================================================================

def create_advanced_payoff_chart(calculator: OptionsPayoffCalculator,
                                 title: str = "Strategy Payoff") -> go.Figure:
    """Create a detailed, interactive payoff chart"""

    metrics = calculator.get_strategy_metrics()

    if not metrics:
        fig = go.Figure()
        fig.add_annotation(text="Add legs to see payoff", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=16, color="#666"))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=450)
        return fig

    prices = metrics['prices']
    pnl = metrics['pnl']
    spot = calculator.spot_price
    breakevens = metrics['breakevens']

    fig = go.Figure()

    # Profit zone fill (green)
    profit_mask = pnl >= 0
    fig.add_trace(go.Scatter(
        x=prices,
        y=np.where(profit_mask, pnl, 0),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.15)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
        name='Profit Zone'
    ))

    # Loss zone fill (red)
    loss_mask = pnl < 0
    fig.add_trace(go.Scatter(
        x=prices,
        y=np.where(loss_mask, pnl, 0),
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.15)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
        name='Loss Zone'
    ))

    # Main P&L line
    fig.add_trace(go.Scatter(
        x=prices,
        y=pnl,
        mode='lines',
        name='P&L at Expiry',
        line=dict(color='#2196f3', width=3),
        hovertemplate='<b>Underlying:</b> ₹%{x:,.0f}<br><b>P&L:</b> ₹%{y:,.0f}<extra></extra>'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#9e9e9e", line_width=1.5)

    # Current spot price line
    fig.add_vline(x=spot, line_dash="dash", line_color="#ff9800", line_width=2)
    fig.add_annotation(
        x=spot, y=1.02, yref="paper",
        text=f"Spot: ₹{spot:,.0f}",
        showarrow=False,
        font=dict(size=11, color="#ff9800", family="Inter"),
        bgcolor="rgba(255,255,255,0.9)",
        borderpad=4
    )

    # Breakeven markers
    for i, be in enumerate(breakevens):
        fig.add_vline(x=be, line_dash="dot", line_color="#4caf50", line_width=2)
        fig.add_annotation(
            x=be, y=0,
            text=f"BE: ₹{be:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#4caf50",
            arrowsize=1,
            ax=0, ay=-40 if i % 2 == 0 else 40,
            font=dict(size=10, color="#4caf50"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#4caf50",
            borderwidth=1,
            borderpad=3
        )

    # Strike price markers
    strikes = set(leg['strike'] for leg in calculator.legs)
    for strike in strikes:
        fig.add_vline(x=strike, line_dash="dot", line_color="#9c27b0", line_width=1, opacity=0.5)

    # Max profit/loss annotations
    max_profit = metrics['max_profit']
    max_loss = metrics['max_loss']

    if max_profit > 0:
        max_profit_idx = np.argmax(pnl)
        fig.add_annotation(
            x=prices[max_profit_idx], y=max_profit,
            text=f"Max Profit: ₹{max_profit:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor="#4caf50",
            ax=50, ay=-20,
            font=dict(size=11, color="#4caf50", family="JetBrains Mono"),
            bgcolor="rgba(0, 200, 83, 0.1)",
            bordercolor="#4caf50", borderwidth=1, borderpad=4
        )

    if max_loss < 0:
        max_loss_idx = np.argmin(pnl)
        fig.add_annotation(
            x=prices[max_loss_idx], y=max_loss,
            text=f"Max Loss: ₹{max_loss:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor="#f44336",
            ax=-50, ay=20,
            font=dict(size=11, color="#f44336", family="JetBrains Mono"),
            bgcolor="rgba(255, 82, 82, 0.1)",
            bordercolor="#f44336", borderwidth=1, borderpad=4
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color="#333", family="Inter"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Underlying Price (₹)", font=dict(size=12, color="#666")),
            tickfont=dict(color="#666", family="JetBrains Mono", size=11),
            gridcolor="#f0f0f0",
            tickformat=",",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Profit / Loss (₹)", font=dict(size=12, color="#666")),
            tickfont=dict(color="#666", family="JetBrains Mono", size=11),
            gridcolor="#f0f0f0",
            tickformat=",",
            tickprefix="₹",
            zeroline=True,
            zerolinecolor="#9e9e9e",
            zerolinewidth=1.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=70, r=40, t=60, b=60),
        height=450
    )

    return fig


def create_pnl_table(calculator: OptionsPayoffCalculator) -> pd.DataFrame:
    """Create a P&L table at key price levels"""

    metrics = calculator.get_strategy_metrics()
    if not metrics:
        return pd.DataFrame()

    spot = calculator.spot_price
    breakevens = metrics['breakevens']
    strikes = sorted(set(leg['strike'] for leg in calculator.legs))

    # Key price levels
    price_levels = sorted(set(
        [spot] +
        breakevens +
        strikes +
        [spot - 500, spot - 300, spot - 100, spot + 100, spot + 300, spot + 500]
    ))

    # Filter to reasonable range
    price_levels = [p for p in price_levels if spot * 0.9 <= p <= spot * 1.1]

    data = []
    for price in sorted(price_levels):
        pnl_value = float(calculator.calculate_payoff(np.array([price]))[0])

        # Determine label
        if price == spot:
            label = "Current Spot"
        elif price in breakevens:
            label = "Breakeven"
        elif price in strikes:
            label = "Strike"
        else:
            diff = price - spot
            label = f"+{diff:.0f}" if diff > 0 else f"{diff:.0f}"

        data.append({
            'Price': f"₹{price:,.0f}",
            'Level': label,
            'P&L': f"₹{pnl_value:,.0f}",
            'pnl_raw': pnl_value
        })

    return pd.DataFrame(data)


# ============================================================================
# STRATEGY BUILDER UI
# ============================================================================

def render_strategy_builder():
    """Render the Strategy Builder tab"""

    st.markdown("### 🔧 Strategy Builder")
    st.markdown("Build and analyze custom options strategies with interactive payoff visualization.")

    # Initialize session state
    if 'builder_legs' not in st.session_state:
        st.session_state.builder_legs = []
    if 'builder_spot' not in st.session_state:
        st.session_state.builder_spot = 24500.0

    # Settings row
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY"], key="builder_underlying")
        lot_size = get_lot_size(underlying)

    with col2:
        spot_price = st.number_input("Spot Price", value=st.session_state.builder_spot,
                                     step=50.0, key="builder_spot_input")
        st.session_state.builder_spot = spot_price

    with col3:
        st.markdown("**Quick Templates**")
        template_cols = st.columns(4)
        templates = list(STRATEGY_TEMPLATES.keys())

        for i, template in enumerate(templates[:4]):
            with template_cols[i]:
                if st.button(template, key=f"template_{i}", use_container_width=True):
                    st.session_state.builder_legs = []
                    for leg in STRATEGY_TEMPLATES[template]['legs']:
                        st.session_state.builder_legs.append({
                            'action': leg['action'],
                            'option_type': leg['option_type'],
                            'strike': spot_price + leg['strike_offset'],
                            'premium': leg['premium'],
                            'lots': 1
                        })
                    st.rerun()

    st.markdown("---")

    # Leg builder
    col_legs, col_chart = st.columns([1, 2])

    with col_legs:
        st.markdown("#### Add Option Leg")

        with st.form("add_leg_form"):
            form_cols = st.columns(2)
            with form_cols[0]:
                action = st.selectbox("Action", ["BUY", "SELL"])
                strike = st.number_input("Strike", value=float(round(spot_price / 50) * 50), step=50.0)
            with form_cols[1]:
                option_type = st.selectbox("Type", ["CE", "PE"])
                premium = st.number_input("Premium (₹)", value=100.0, step=5.0, min_value=0.1)

            lots = st.number_input("Lots", value=1, min_value=1, max_value=50)

            submitted = st.form_submit_button("➕ Add Leg", use_container_width=True)

            if submitted:
                st.session_state.builder_legs.append({
                    'action': action,
                    'option_type': option_type,
                    'strike': strike,
                    'premium': premium,
                    'lots': lots
                })
                st.rerun()

        # Current legs
        st.markdown("#### Current Legs")

        if st.session_state.builder_legs:
            for i, leg in enumerate(st.session_state.builder_legs):
                action_class = "buy" if leg['action'] == "BUY" else "sell"
                action_symbol = "🟢" if leg['action'] == "BUY" else "🔴"

                col_leg, col_del = st.columns([4, 1])
                with col_leg:
                    st.markdown(f"""
                    <div style="background: var(--bg-secondary); padding: 0.5rem 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;">
                        {action_symbol} <span class="{action_class}" style="font-weight:600;">{leg['action']}</span> 
                        {leg['lots']} lot × <b>{leg['strike']:.0f} {leg['option_type']}</b> @ ₹{leg['premium']}
                    </div>
                    """, unsafe_allow_html=True)
                with col_del:
                    if st.button("🗑️", key=f"del_leg_{i}"):
                        st.session_state.builder_legs.pop(i)
                        st.rerun()

            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.builder_legs = []
                st.rerun()
        else:
            st.info("No legs added. Add legs or select a template.")

    with col_chart:
        # Create calculator and chart
        calculator = OptionsPayoffCalculator(spot_price, lot_size)

        for leg in st.session_state.builder_legs:
            calculator.add_leg(
                leg['action'], leg['option_type'],
                leg['strike'], leg['premium'], leg['lots']
            )

        # Payoff chart
        fig = create_advanced_payoff_chart(calculator, "Strategy Payoff at Expiry")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Metrics
        metrics = calculator.get_strategy_metrics()

        if metrics:
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                mp = metrics['max_profit']
                color = "green" if mp > 0 else "red"
                mp_display = "Unlimited" if mp > 1e9 else f"₹{mp:,.0f}"
                st.markdown(
                    f'''<div class="metric-box"><div class="metric-label">Max Profit</div><div class="metric-value {color}">{mp_display}</div></div>''',
                    unsafe_allow_html=True)

            with m2:
                ml = metrics['max_loss']
                color = "red" if ml < 0 else "green"
                ml_display = "Unlimited" if ml < -1e9 else f"₹{ml:,.0f}"
                st.markdown(
                    f'''<div class="metric-box"><div class="metric-label">Max Loss</div><div class="metric-value {color}">{ml_display}</div></div>''',
                    unsafe_allow_html=True)

            with m3:
                np_val = metrics['net_premium']
                color = "green" if np_val > 0 else "red"
                label = "Credit" if np_val > 0 else "Debit"
                st.markdown(
                    f'''<div class="metric-box"><div class="metric-label">Net {label}</div><div class="metric-value {color}">₹{abs(np_val):,.0f}</div></div>''',
                    unsafe_allow_html=True)

            with m4:
                bes = metrics['breakevens']
                be_text = " / ".join([f"₹{b:,.0f}" for b in bes]) if bes else "N/A"
                st.markdown(
                    f'''<div class="metric-box"><div class="metric-label">Breakeven</div><div class="metric-value" style="font-size:0.9rem;">{be_text}</div></div>''',
                    unsafe_allow_html=True)

            # P&L Table
            st.markdown("#### P&L at Key Price Levels")
            pnl_df = create_pnl_table(calculator)

            if not pnl_df.empty:
                # Style the dataframe
                def style_pnl(val):
                    if isinstance(val, (int, float)):
                        return 'color: #00c853' if val >= 0 else 'color: #ff5252'
                    if '₹' in str(val) and '-' in str(val):
                        return 'color: #ff5252'
                    if '₹' in str(val):
                        try:
                            num = float(str(val).replace('₹', '').replace(',', ''))
                            return 'color: #00c853' if num >= 0 else 'color: #ff5252'
                        except:
                            pass
                    return ''

                st.dataframe(
                    pnl_df[['Price', 'Level', 'P&L']].style.applymap(
                        style_pnl, subset=['P&L']
                    ),
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================================
# VERDICT VIEWER (Simplified)
# ============================================================================

def render_verdict_viewer():
    """Render the Verdict Viewer tab"""

    st.markdown("### 📈 Judge Verdict Visualization")

    uploaded_file = st.file_uploader("Upload Judge Verdict JSON", type=['json'])

    if uploaded_file is None:
        st.markdown('''
        <div style="text-align:center; padding:4rem; color:#718096;">
            <div style="font-size:3.5rem; margin-bottom:1rem;">📤</div>
            <div style="font-size:1.2rem; margin-bottom:0.5rem; color:#a0aec0;">Upload Judge Verdict JSON</div>
            <div style="font-size:0.8rem;">Drag & drop or click to upload</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    try:
        verdict_data = json.load(uploaded_file)

        # Extract data
        final_verdict = safe_get(verdict_data, 'final_verdict', default={})
        judge_metadata = safe_get(verdict_data, 'judge_metadata', default={})
        execution = safe_get(verdict_data, 'execution_instructions', default={})
        scenarios = safe_get(verdict_data, 'scenario_summary', default={})
        notes = safe_get(verdict_data, 'judge_notes', default={})
        dissent = safe_get(verdict_data, 'dissenting_opinion', default={})

        underlying = safe_get(judge_metadata, 'underlying', default='NIFTY')
        action = safe_get(final_verdict, 'action', default='UNKNOWN').upper()

        # Verdict Banner
        action_class = 'trade' if action == 'TRADE' else 'no-trade'
        action_icon = '✅' if action == 'TRADE' else '🛑'
        one_liner = escape_html(safe_get(verdict_data, 'final_one_liner', default=''))

        st.markdown(f'''
        <div class="verdict-banner {action_class}">
            <div class="verdict-text {action_class}">{action_icon} {action.replace("_", " ")}</div>
            <div style="color:#a0aec0; font-size:0.9rem; margin-top:0.5rem;">{one_liner}</div>
        </div>
        ''', unsafe_allow_html=True)

        # Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'''<div class="card"><div class="card-header">Underlying</div><div class="card-value blue">{underlying}</div></div>''',
                unsafe_allow_html=True)
        with c2:
            conf = safe_get(final_verdict, 'confidence_level', default=0)
            conf_color = "green" if conf >= 7 else "red" if conf < 4 else ""
            st.markdown(
                f'''<div class="card"><div class="card-header">Confidence</div><div class="card-value {conf_color}">{conf}/10</div></div>''',
                unsafe_allow_html=True)
        with c3:
            if action == 'TRADE':
                strategy = safe_get(final_verdict, 'if_trade', 'strategy_name', default='N/A').replace('_', ' ').title()
                st.markdown(
                    f'''<div class="card"><div class="card-header">Strategy</div><div class="card-value" style="font-size:1rem;">{strategy}</div></div>''',
                    unsafe_allow_html=True)
            else:
                reason = safe_get(final_verdict, 'if_no_trade', 'primary_reason', default='N/A')[:30]
                st.markdown(
                    f'''<div class="card"><div class="card-header">Reason</div><div class="card-value red" style="font-size:0.85rem;">{reason}...</div></div>''',
                    unsafe_allow_html=True)
        with c4:
            ts = safe_get(judge_metadata, 'judgment_timestamp', default='N/A')
            st.markdown(
                f'''<div class="card"><div class="card-header">Timestamp</div><div class="card-value" style="font-size:0.8rem;">{ts}</div></div>''',
                unsafe_allow_html=True)

        st.markdown("---")

        # Strategy Payoff (if TRADE)
        if action == 'TRADE':
            trade_data = safe_get(final_verdict, 'if_trade', default={})
            exact_exec = safe_get(trade_data, 'exact_execution', default={})

            if exact_exec:
                col_chart, col_details = st.columns([2, 1])

                with col_chart:
                    st.markdown("#### 📈 Strategy Payoff")

                    # Create calculator from verdict data
                    lot_size = get_lot_size(underlying)

                    # Try to get spot price
                    spot = 24500  # Default
                    for item in safe_get(execution, 'pre_trade_checklist', default=[]):
                        if isinstance(item, str) and 'spot' in item.lower():
                            match = re.search(r'[\d,]+\.?\d*', item)
                            if match:
                                try:
                                    spot = float(match.group().replace(',', ''))
                                except:
                                    pass

                    calculator = OptionsPayoffCalculator(spot, lot_size)

                    for leg_name, leg_data in exact_exec.items():
                        if isinstance(leg_data, dict) and 'strike' in leg_data:
                            calculator.add_leg(
                                leg_data.get('action', 'BUY'),
                                leg_data.get('option_type', 'CE'),
                                leg_data.get('strike', 0),
                                leg_data.get('limit_price', 0) or 0,
                                leg_data.get('lots', 1) or 1
                            )

                    strategy_name = safe_get(trade_data, 'strategy_name', default='Strategy').replace('_', ' ').title()
                    fig = create_advanced_payoff_chart(calculator, strategy_name)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # P&L Table
                    pnl_df = create_pnl_table(calculator)
                    if not pnl_df.empty:
                        st.markdown("**P&L at Key Levels**")
                        st.dataframe(pnl_df[['Price', 'Level', 'P&L']], use_container_width=True, hide_index=True)

                with col_details:
                    st.markdown("#### Trade Details")

                    # Trade Summary
                    summary = safe_get(trade_data, 'trade_summary', default={})
                    st.markdown(f'''
                    <div class="strategy-panel">
                        <h3>📊 Summary</h3>
                        <table class="data-table">
                            <tr><td><b>Max Profit</b></td><td style="text-align:right; color:#00c853;">{safe_get(summary, "max_profit", default="N/A")}</td></tr>
                            <tr><td><b>Max Loss</b></td><td style="text-align:right; color:#ff5252;">{safe_get(summary, "max_loss", default="N/A")}</td></tr>
                            <tr><td><b>Breakeven</b></td><td style="text-align:right;">{safe_get(summary, "breakeven", default="N/A")}</td></tr>
                            <tr><td><b>POP</b></td><td style="text-align:right; color:#2196f3;">{safe_get(summary, "probability_of_profit", default="N/A")}</td></tr>
                        </table>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Order Details
                    st.markdown("**Orders**")
                    for leg_name, leg_data in exact_exec.items():
                        if isinstance(leg_data, dict):
                            action_val = safe_get(leg_data, 'action', default='BUY')
                            badge_class = "buy" if action_val == "BUY" else "sell"
                            strike = safe_get(leg_data, 'strike', default=0)
                            opt_type = safe_get(leg_data, 'option_type', default='CE')
                            premium = safe_get(leg_data, 'limit_price', default=0)

                            st.markdown(f'''
                            <div style="background:#1a1f2e; padding:0.5rem 0.75rem; border-radius:6px; margin-bottom:0.5rem; font-size:0.85rem;">
                                <span class="leg-badge {badge_class}">{action_val}</span>
                                <b>₹{strike:,.0f} {opt_type}</b> @ ₹{premium}
                            </div>
                            ''', unsafe_allow_html=True)

        st.markdown("---")

        # Scenarios & Notes (side by side)
        col_scen, col_notes = st.columns(2)

        with col_scen:
            st.markdown("#### 📊 Scenarios")
            st.markdown(f'''
            <div class="strategy-panel">
                <table class="data-table">
                    <tr><td style="border-left:3px solid #00c853;"><b>Best</b></td><td>{escape_html(safe_get(scenarios, "best_case", "scenario", default="N/A"))}</td></tr>
                    <tr><td style="border-left:3px solid #2196f3;"><b>Expected</b></td><td>{escape_html(safe_get(scenarios, "expected_case", "scenario", default="N/A"))}</td></tr>
                    <tr><td style="border-left:3px solid #ff5252;"><b>Worst</b></td><td>{escape_html(safe_get(scenarios, "worst_case", "scenario", default="N/A"))}</td></tr>
                </table>
            </div>
            ''', unsafe_allow_html=True)

        with col_notes:
            st.markdown("#### 📝 Judge Notes")
            st.markdown(f'''
            <div class="strategy-panel">
                <p style="color:#fff; margin-bottom:0.5rem;"><b style="color:#2196f3;">Key Insight:</b> {escape_html(safe_get(notes, "key_insight", default="N/A"))}</p>
                <p style="color:#fff; margin-bottom:0.5rem;"><b style="color:#ffc107;">What to Watch:</b> {escape_html(safe_get(notes, "what_to_watch", default="N/A"))}</p>
                <p style="color:#fff; margin-bottom:0;"><b style="color:#00c853;">Next Review:</b> {escape_html(safe_get(notes, "next_review_trigger", default="N/A"))}</p>
            </div>
            ''', unsafe_allow_html=True)

        # Dissenting Opinion
        has_dissent = str(safe_get(dissent, 'any_valid_counter_view', default='no')).lower() == 'yes'
        if has_dissent:
            st.markdown("#### ⚖️ Dissenting Opinion")
            st.markdown(f'''
            <div class="strategy-panel" style="border-left:3px solid #ffc107;">
                <p style="color:#fff;"><b style="color:#ffc107;">Argument:</b> {escape_html(safe_get(dissent, "counter_argument", default="N/A"))}</p>
                <p style="color:#fff; margin-bottom:0;"><b>Why Overruled:</b> {escape_html(safe_get(dissent, "why_overruled", default="N/A"))}</p>
            </div>
            ''', unsafe_allow_html=True)

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        with st.expander("Details"):
            st.code(traceback.format_exc())
# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>📊 Options Trading System</h1>
                <div style="font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #718096;">
                    Strategy Builder • Payoff Analysis • LLM Verdicts
                </div>
            </div>
            <div style="text-align: right; font-family: 'JetBrains Mono'; color: #718096; font-size: 0.75rem;">
                v3.1<br>NSE F&O
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for Token
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check for env var
        env_token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        
        token_input = st.text_input(
            "Upstox API Token", 
            value=env_token, 
            type="password",
            help="Enter your Upstox access token here. It will be used for data collection."
        )
        
        if token_input:
            # Set/Update env var for the session
            os.environ["UPSTOX_ACCESS_TOKEN"] = token_input
            st.success("Token loaded!")
        else:
            st.warning("No token provided. Data collection will fail.")

        st.markdown("---")
        st.markdown("### 🧠 LLM Settings")
        
        llm_provider = st.selectbox("LLM Provider", ["Claude", "OpenAI"])
        
        env_llm_key = os.environ.get("LLM_API_KEY", "")
        llm_key_input = st.text_input(
            f"{llm_provider} API Key",
            value=env_llm_key,
            type="password",
            help=f"Enter your {llm_provider} API Key for automated analysis."
        )
        
        if llm_key_input:
            os.environ["LLM_API_KEY"] = llm_key_input
            
        st.markdown("---")
        st.markdown("### About")
        st.info("This system uses a multi-LLM approach to analyze options strategies based on live market data.")


    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔧 Strategy Builder", "📈 Verdict Viewer", "🚀 Data Collection"])

    with tab1:
        render_strategy_builder()

    with tab2:
        render_verdict_viewer()

    with tab3:
        st.markdown("### 🚀 Market Data Collection")
        
        if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
            st.warning("⚠️ Please enter your Upstox Access Token in the sidebar to use this feature.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            instrument = st.selectbox("Instrument", ["NIFTY", "BANKNIFTY"])
        with col2:
            strike_range = st.slider("Strike Range (%)", 5, 20, 10)
        with col3:
            multi_expiry = st.checkbox("Multi-expiry Analysis", value=True, help="Analyze next 3 expiries")

        if st.button("🚀 Run Analysis", use_container_width=True):
            token = os.environ.get("UPSTOX_ACCESS_TOKEN")
            if not token:
                st.error("❌ Access Token is missing. Please check the sidebar.")
            else:
                try:
                    with st.spinner(f"Collecting data for {instrument}... This may take 30-60 seconds."):
                        # Initialize collector
                        collector = data_collector_enhanced.OptionsDataCollector(token)
                        
                        # Run analysis
                        analysis_result = collector.collect_full_analysis(
                            underlying=instrument,
                            brokerage_per_order=20,
                            multi_expiry=multi_expiry,
                            strike_range_percent=strike_range
                        )
                        
                        if analysis_result.get("status") == "error":
                            st.error(f"Analysis Failed: {analysis_result.get('error_message')}")
                            st.json(analysis_result)
                        else:
                            st.session_state['analysis_result'] = analysis_result
                            st.success("✅ Data Collection Complete!")
                            
                            # Save to file
                            # Fix: Use IST (UTC+5:30)
                            ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
                            timestamp = ist_now.strftime("%H%M")
                            filename = f"market_data_{instrument}_{timestamp}.json"
                            json_str, file_path = data_collector_enhanced.generate_json_output(
                                analysis_result, filename, use_date_folder=True
                            )
                            
                            st.markdown(f"**Output saved to:** `{file_path}`")
                            
                            # Download Button
                            st.download_button(
                                label="⬇️ Download Analysis JSON",
                                data=json_str,
                                file_name=filename,
                                mime="application/json"
                            )
                            
                            # Show summary
                            st.markdown("### 📊 Market Context")
                            ctx = analysis_result.get("market_context", {})
                            
                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                st.metric("Spot Price", f"₹{ctx.get('spot_price', 0):,.2f}")
                            with m2:
                                st.metric("India VIX", f"{ctx.get('vix', {}).get('value', 0):.2f}")
                            with m3:
                                st.metric("PCR", f"{analysis_result.get('framework_signals', {}).get('pcr_oi', 'N/A')}")
                            with m4:
                                st.metric("Trend", f"{analysis_result.get('core_analysis', {}).get('mean_reversion', {}).get('trend', 'N/A').upper()}")
                                
                            with st.expander("View Raw JSON"):
                                st.json(analysis_result)
                                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)

        # LLM Analysis Section
        st.markdown("---")
        st.markdown("### 🧠 Automated Analysis")
        
        if st.button("🧠 Run LLM Analysis", disabled=not os.environ.get("LLM_API_KEY")):
            if not os.environ.get("LLM_API_KEY"):
                st.error("Please provide an LLM API Key in the sidebar.")
            elif 'analysis_result' not in st.session_state:
                 st.error("Please run Data Collection first.")
            else:
                analysis_result = st.session_state['analysis_result']
                with st.spinner(f"Running analysis with {llm_provider}..."):
                    # Determine prompt file path
                    # Assuming running from System/ directory, prompts are in root
                    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stage2_llm_analysis_prompt.md")
                    
                    llm_result = llm_runner.run_analysis(
                        provider=llm_provider,
                        api_key=os.environ["LLM_API_KEY"],
                        market_data=analysis_result,
                        prompt_file=prompt_path
                    )
                    
                    if llm_result['status'] == 'success':
                        st.success("Analysis Complete!")
                        with st.expander("View Analysis", expanded=True):
                            st.markdown(llm_result['content'])
                        
                        # Save analysis
                        # Fix: Use IST (UTC+5:30)
                        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
                        timestamp = ist_now.strftime("%H%M")
                        analysis_filename = f"llm_analysis_{instrument}_{timestamp}.json"
                        
                        # Try to extract JSON from content if possible
                        content = llm_result['content']
                        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                        
                        final_json_content = None
                        
                        if json_match:
                            try:
                                parsed_content = json.loads(json_match.group(1))
                                final_json_content = json.dumps(parsed_content, indent=2)
                                # Save as proper JSON
                                output_dir = data_collector_enhanced.get_output_directory()
                                save_path = os.path.join(output_dir, analysis_filename)
                                with open(save_path, 'w') as f:
                                    f.write(final_json_content)
                                st.markdown(f"**Analysis saved to:** `{save_path}`")
                            except:
                                pass
                        
                        # If we couldn't parse JSON, we might still want to offer the raw content or just skip
                        if final_json_content:
                            st.download_button(
                                label="⬇️ Download LLM Analysis JSON",
                                data=final_json_content,
                                file_name=analysis_filename,
                                mime="application/json"
                            )
                    else:
                        st.error(f"LLM Analysis Failed: {llm_result.get('error_message')}")


if __name__ == "__main__":
    main()
