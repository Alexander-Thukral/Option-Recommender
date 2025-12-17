#!/usr/bin/env python3
"""
Options Trading Data Collector & Analyzer
==========================================
Stage 1 of the Multi-LLM Options Trading Decision System

This script collects market data from Upstox API and performs mathematical analysis
based on the Comprehensive Analysis Framework focusing on:
- Core Factors: Volatility (IV vs HV), Theta/Time Decay, Position Sizing context
- Secondary Factors: OI Analysis, PCR, Max Pain, Mean Reversion signals
- Cost Analysis: STT, Brokerage, Slippage estimates

Author: Options Trading System
Version: 1.1
Changes:
- Added multiple expiry support (weekly + monthly)
- Added strike range filtering (±20% of spot)
- Fixed mean reversion historical data parsing
"""

import requests
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the data collector"""

    # Upstox API Base URL
    BASE_URL = "https://api.upstox.com/v2"

    # Instrument Keys
    NIFTY_INDEX = "NSE_INDEX|Nifty 50"
    BANKNIFTY_INDEX = "NSE_INDEX|Nifty Bank"
    INDIA_VIX = "NSE_INDEX|India VIX"

    # Trading Cost Parameters (India-specific)
    STT_RATE_OPTIONS_SELL = 0.000625  # 0.0625% on sell side premium
    STT_RATE_OPTIONS_EXERCISE = 0.00125  # 0.125% on ITM exercise (on full value!)
    SEBI_CHARGES = 0.000001  # ₹10 per crore
    STAMP_DUTY = 0.00003  # 0.003% on buy side
    GST_RATE = 0.18  # 18% on brokerage + transaction charges

    # Lot Sizes
    NIFTY_LOT_SIZE = 25
    BANKNIFTY_LOT_SIZE = 15

    # Analysis Parameters
    IV_HIGH_PERCENTILE = 70  # Above this = consider selling
    IV_LOW_PERCENTILE = 30  # Below this = buying less risky
    OPTIMAL_DTE_MIN = 30  # Minimum days to expiry for theta sweet spot
    OPTIMAL_DTE_MAX = 60  # Maximum days to expiry for theta sweet spot

    # Mean Reversion Parameters
    MEAN_REVERSION_LOOKBACK = 20  # Days for calculating mean
    ZSCORE_OVERBOUGHT = 2.0
    ZSCORE_OVERSOLD = -2.0

    # Strike Range Filter (as percentage of spot price)
    STRIKE_RANGE_PERCENT = 20  # ±20% of spot price

    # Expiry Selection
    MAX_WEEKLY_EXPIRIES = 4  # Number of weekly expiries to capture
    INCLUDE_MONTHLY = True  # Include monthly expiry


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OptionStrike:
    """Individual option strike data"""
    strike: float
    call_ltp: float
    call_iv: float
    call_delta: float
    call_theta: float
    call_gamma: float
    call_vega: float
    call_oi: int
    call_volume: int
    call_bid: float
    call_ask: float
    call_pop: float  # Probability of Profit
    put_ltp: float
    put_iv: float
    put_delta: float
    put_theta: float
    put_gamma: float
    put_vega: float
    put_oi: int
    put_volume: int
    put_bid: float
    put_ask: float
    put_pop: float
    pcr: float


@dataclass
class VolatilityAnalysis:
    """Volatility analysis results"""
    current_iv_atm: float
    iv_percentile: float  # Where current IV sits vs historical
    iv_rank: float  # (Current - Min) / (Max - Min)
    historical_volatility_20d: float
    iv_hv_ratio: float  # IV/HV - above 1 means IV is elevated
    iv_skew: str  # "call_premium", "put_premium", "neutral"
    vix_current: float
    vix_percentile: float
    regime: str  # "low_vol", "normal", "high_vol", "extreme"


@dataclass
class ThetaAnalysis:
    """Time decay analysis"""
    days_to_expiry: int
    theta_regime: str  # "accelerating", "moderate", "slow"
    atm_theta_per_day: float
    theta_capture_potential: float  # % of premium that's time value
    is_optimal_dte: bool  # Within 30-60 DTE sweet spot


@dataclass
class OIAnalysis:
    """Open Interest analysis"""
    total_call_oi: int
    total_put_oi: int
    pcr_oi: float
    pcr_volume: float
    max_pain: float
    significant_call_strikes: List[float]  # High OI call strikes (resistance)
    significant_put_strikes: List[float]  # High OI put strikes (support)
    oi_buildup_bias: str  # "bullish", "bearish", "neutral"


@dataclass
class MeanReversionSignal:
    """Mean reversion analysis for directional bias"""
    current_price: float
    sma_20: float
    sma_50: float
    z_score: float
    bollinger_position: float  # 0-1, where in BB channel
    signal: str  # "overbought", "oversold", "neutral"
    trend: str  # "uptrend", "downtrend", "sideways"


@dataclass
class CostAnalysis:
    """Trading cost estimates"""
    estimated_brokerage_per_lot: float
    stt_estimate: float
    total_cost_per_lot: float
    breakeven_points_impact: float  # How much costs affect breakeven


@dataclass
class MarketContext:
    """Overall market context"""
    timestamp: str
    underlying_symbol: str
    spot_price: float
    futures_price: float
    futures_premium: float  # Futures - Spot
    day_change_percent: float
    day_high: float
    day_low: float
    day_range_percent: float


@dataclass
class ExpiryData:
    """Data for a single expiry"""
    expiry_date: str
    days_to_expiry: int
    is_weekly: bool
    is_monthly: bool
    total_strikes: int
    atm_strike: float
    atm_call_iv: float
    atm_put_iv: float
    atm_straddle_price: float
    strikes_data: List[dict]
    theta_analysis: dict
    oi_analysis: dict


# ============================================================================
# UPSTOX API CLIENT
# ============================================================================

class UpstoxClient:
    """Client for interacting with Upstox API"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling"""
        url = f"{Config.BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return {"status": "error", "error": str(e)}

    def get_option_chain(self, instrument_key: str, expiry_date: str) -> dict:
        """Get complete option chain for an instrument"""
        endpoint = "/option/chain"
        params = {
            'instrument_key': instrument_key,
            'expiry_date': expiry_date
        }
        return self._make_request(endpoint, params)

    def get_market_quote(self, instrument_keys: List[str]) -> dict:
        """Get market quotes for instruments"""
        endpoint = "/market-quote/quotes"
        # Upstox expects comma-separated instrument keys
        params = {'instrument_key': ','.join(instrument_keys)}
        return self._make_request(endpoint, params)

    def get_historical_candles(self, instrument_key: str, interval: str,
                               to_date: str, from_date: str) -> dict:
        """Get historical OHLC data"""
        # URL encode the instrument key properly
        encoded_key = instrument_key.replace("|", "%7C")
        endpoint = f"/historical-candle/{encoded_key}/{interval}/{to_date}/{from_date}"
        return self._make_request(endpoint)

    def get_intraday_candles(self, instrument_key: str, interval: str = "30minute") -> dict:
        """Get intraday OHLC data"""
        encoded_key = instrument_key.replace("|", "%7C")
        endpoint = f"/historical-candle/intraday/{encoded_key}/{interval}"
        return self._make_request(endpoint)

    def get_option_expiries(self, instrument_key: str) -> dict:
        """Get available option expiry dates"""
        endpoint = "/option/contract"
        params = {'instrument_key': instrument_key}
        return self._make_request(endpoint, params)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

class OptionsAnalyzer:
    """Analyzes options data based on the Comprehensive Framework"""

    def __init__(self, client: UpstoxClient):
        self.client = client

    def parse_option_chain(self, chain_data: dict, spot_price: float,
                           filter_strikes: bool = True) -> List[OptionStrike]:
        """Parse raw option chain data into structured format with optional filtering"""
        strikes = []

        if chain_data.get('status') != 'success':
            return strikes

        # Calculate strike range if filtering
        if filter_strikes:
            lower_bound = spot_price * (1 - Config.STRIKE_RANGE_PERCENT / 100)
            upper_bound = spot_price * (1 + Config.STRIKE_RANGE_PERCENT / 100)
        else:
            lower_bound = 0
            upper_bound = float('inf')

        for item in chain_data.get('data', []):
            try:
                strike_price = item.get('strike_price', 0)

                # Filter strikes outside range
                if strike_price < lower_bound or strike_price > upper_bound:
                    continue

                call = item.get('call_options', {})
                put = item.get('put_options', {})
                call_market = call.get('market_data', {})
                put_market = put.get('market_data', {})
                call_greeks = call.get('option_greeks', {})
                put_greeks = put.get('option_greeks', {})

                strike = OptionStrike(
                    strike=strike_price,
                    call_ltp=call_market.get('ltp', 0),
                    call_iv=call_greeks.get('iv', 0),
                    call_delta=call_greeks.get('delta', 0),
                    call_theta=call_greeks.get('theta', 0),
                    call_gamma=call_greeks.get('gamma', 0),
                    call_vega=call_greeks.get('vega', 0),
                    call_oi=call_market.get('oi', 0),
                    call_volume=call_market.get('volume', 0),
                    call_bid=call_market.get('bid_price', 0),
                    call_ask=call_market.get('ask_price', 0),
                    call_pop=call_greeks.get('pop', 0),
                    put_ltp=put_market.get('ltp', 0),
                    put_iv=put_greeks.get('iv', 0),
                    put_delta=put_greeks.get('delta', 0),
                    put_theta=put_greeks.get('theta', 0),
                    put_gamma=put_greeks.get('gamma', 0),
                    put_vega=put_greeks.get('vega', 0),
                    put_oi=put_market.get('oi', 0),
                    put_volume=put_market.get('volume', 0),
                    put_bid=put_market.get('bid_price', 0),
                    put_ask=put_market.get('ask_price', 0),
                    put_pop=put_greeks.get('pop', 0),
                    pcr=item.get('pcr', 0)
                )
                strikes.append(strike)
            except Exception as e:
                print(f"Error parsing strike: {e}")
                continue

        return sorted(strikes, key=lambda x: x.strike)

    def find_atm_strike(self, strikes: List[OptionStrike], spot_price: float) -> OptionStrike:
        """Find the At-The-Money strike"""
        if not strikes:
            return None
        return min(strikes, key=lambda x: abs(x.strike - spot_price))

    def calculate_historical_volatility(self, candles: List[List]) -> float:
        """Calculate 20-day historical volatility from daily candles"""
        if not candles or len(candles) < 15:
            print(f"  Warning: Insufficient candles for HV calculation ({len(candles) if candles else 0} candles)")
            return 0.0

        # Candles are typically in reverse chronological order from Upstox
        # Format: [timestamp, open, high, low, close, volume, oi]
        # Take the most recent candles and reverse to chronological order
        recent_candles = candles[:min(len(candles), 25)]

        # Extract closing prices (index 4 in candle array)
        closes = []
        for c in recent_candles:
            if isinstance(c, list) and len(c) >= 5:
                closes.append(c[4])  # Close price is at index 4

        if len(closes) < 10:
            print(f"  Warning: Only {len(closes)} valid closes found")
            return 0.0

        # Reverse to chronological order for proper return calculation
        closes = closes[::-1]

        # Calculate log returns
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                returns.append(math.log(closes[i] / closes[i - 1]))

        if len(returns) < 5:
            print(f"  Warning: Only {len(returns)} returns calculated")
            return 0.0

        # Calculate annualized volatility
        std_dev = statistics.stdev(returns)
        annualized_vol = std_dev * math.sqrt(252) * 100  # As percentage

        print(
            f"  HV Calculation: {len(closes)} closes, {len(returns)} returns, StdDev={std_dev:.6f}, HV={annualized_vol:.2f}%")

        return round(annualized_vol, 2)

    def calculate_iv_percentile(self, current_iv: float, historical_ivs: List[float]) -> float:
        """Calculate IV percentile (what % of historical IVs are below current)"""
        if not historical_ivs:
            return 50.0

        count_below = sum(1 for iv in historical_ivs if iv < current_iv)
        return round((count_below / len(historical_ivs)) * 100, 2)

    def calculate_iv_rank(self, current_iv: float, historical_ivs: List[float]) -> float:
        """Calculate IV Rank: (Current - Min) / (Max - Min)"""
        if not historical_ivs:
            return 50.0

        min_iv = min(historical_ivs)
        max_iv = max(historical_ivs)

        if max_iv == min_iv:
            return 50.0

        return round(((current_iv - min_iv) / (max_iv - min_iv)) * 100, 2)

    def analyze_volatility(self, strikes: List[OptionStrike], spot_price: float,
                           historical_candles: List, vix_value: float) -> VolatilityAnalysis:
        """Comprehensive volatility analysis - CORE FACTOR #1 & #2"""

        atm = self.find_atm_strike(strikes, spot_price)
        if not atm:
            return None

        # ATM IV (average of call and put)
        current_iv_atm = (atm.call_iv + atm.put_iv) / 2 if atm.call_iv and atm.put_iv else atm.call_iv or atm.put_iv

        # Historical Volatility
        hv_20d = self.calculate_historical_volatility(historical_candles)

        # IV/HV Ratio - key metric from framework
        iv_hv_ratio = round(current_iv_atm / hv_20d, 2) if hv_20d > 0 else 1.0

        # IV Skew analysis
        otm_calls = [s for s in strikes if s.strike > spot_price * 1.02]
        otm_puts = [s for s in strikes if s.strike < spot_price * 0.98]

        avg_call_iv = statistics.mean([s.call_iv for s in otm_calls if s.call_iv > 0]) if otm_calls else 0
        avg_put_iv = statistics.mean([s.put_iv for s in otm_puts if s.put_iv > 0]) if otm_puts else 0

        if avg_put_iv > avg_call_iv * 1.1:
            iv_skew = "put_premium"  # Fear in market
        elif avg_call_iv > avg_put_iv * 1.1:
            iv_skew = "call_premium"  # Greed/euphoria
        else:
            iv_skew = "neutral"

        # Determine volatility regime
        if vix_value < 13:
            regime = "low_vol"
        elif vix_value < 18:
            regime = "normal"
        elif vix_value < 25:
            regime = "high_vol"
        else:
            regime = "extreme"

        # For IV percentile, we'd ideally have historical IV data
        # Using VIX as proxy for now
        iv_percentile = 50.0  # Placeholder - would need historical IV storage

        return VolatilityAnalysis(
            current_iv_atm=round(current_iv_atm, 2),
            iv_percentile=iv_percentile,
            iv_rank=50.0,  # Placeholder
            historical_volatility_20d=hv_20d,
            iv_hv_ratio=iv_hv_ratio,
            iv_skew=iv_skew,
            vix_current=vix_value,
            vix_percentile=50.0,  # Would need historical VIX
            regime=regime
        )

    def analyze_theta(self, atm_strike: OptionStrike, expiry_date: str,
                      spot_price: float) -> ThetaAnalysis:
        """Time decay analysis - CORE FACTOR #2"""

        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        today = datetime.now()
        dte = (expiry - today).days

        # Theta regime based on DTE
        if dte <= 7:
            theta_regime = "accelerating"  # Gamma risk high, theta decay rapid
        elif dte <= 21:
            theta_regime = "moderate"
        else:
            theta_regime = "slow"

        # ATM theta per day (average of call and put)
        atm_theta = (abs(atm_strike.call_theta) + abs(atm_strike.put_theta)) / 2

        # Theta capture potential (time value as % of premium)
        atm_premium = (atm_strike.call_ltp + atm_strike.put_ltp) / 2
        intrinsic_call = max(0, spot_price - atm_strike.strike)
        intrinsic_put = max(0, atm_strike.strike - spot_price)
        time_value = atm_premium - (intrinsic_call + intrinsic_put) / 2
        theta_capture = (time_value / atm_premium * 100) if atm_premium > 0 else 0

        # Is this the optimal DTE window (30-60 days)?
        is_optimal = Config.OPTIMAL_DTE_MIN <= dte <= Config.OPTIMAL_DTE_MAX

        return ThetaAnalysis(
            days_to_expiry=dte,
            theta_regime=theta_regime,
            atm_theta_per_day=round(atm_theta, 2),
            theta_capture_potential=round(theta_capture, 2),
            is_optimal_dte=is_optimal
        )

    def analyze_oi(self, strikes: List[OptionStrike], spot_price: float) -> OIAnalysis:
        """Open Interest analysis for support/resistance and sentiment"""

        total_call_oi = sum(s.call_oi for s in strikes)
        total_put_oi = sum(s.put_oi for s in strikes)
        total_call_vol = sum(s.call_volume for s in strikes)
        total_put_vol = sum(s.put_volume for s in strikes)

        pcr_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0
        pcr_volume = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 0

        # Max Pain calculation (strike where total option buyer loss is maximum)
        max_pain = self._calculate_max_pain(strikes, spot_price)

        # Find significant OI strikes (within reasonable range)
        relevant_strikes = [s for s in strikes if abs(s.strike - spot_price) / spot_price < 0.10]

        if relevant_strikes:
            avg_call_oi = sum(s.call_oi for s in relevant_strikes) / len(relevant_strikes)
            avg_put_oi = sum(s.put_oi for s in relevant_strikes) / len(relevant_strikes)
        else:
            avg_call_oi = total_call_oi / len(strikes) if strikes else 0
            avg_put_oi = total_put_oi / len(strikes) if strikes else 0

        significant_calls = [s.strike for s in strikes
                             if s.call_oi > avg_call_oi * 1.5 and s.strike > spot_price]
        significant_puts = [s.strike for s in strikes
                            if s.put_oi > avg_put_oi * 1.5 and s.strike < spot_price]

        # Sort and take top 5
        significant_calls = sorted(significant_calls)[:5]
        significant_puts = sorted(significant_puts, reverse=True)[:5]

        # OI buildup bias
        if pcr_oi > 1.2:
            oi_bias = "bullish"  # More puts = support building
        elif pcr_oi < 0.8:
            oi_bias = "bearish"  # More calls = resistance building
        else:
            oi_bias = "neutral"

        return OIAnalysis(
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            pcr_oi=pcr_oi,
            pcr_volume=pcr_volume,
            max_pain=max_pain,
            significant_call_strikes=significant_calls,
            significant_put_strikes=significant_puts,
            oi_buildup_bias=oi_bias
        )

    def _calculate_max_pain(self, strikes: List[OptionStrike], spot_price: float) -> float:
        """Calculate max pain strike"""
        min_pain = float('inf')
        max_pain_strike = spot_price

        for test_strike in strikes:
            pain = 0
            for s in strikes:
                # Call buyer pain
                if test_strike.strike > s.strike:
                    pain += s.call_oi * (test_strike.strike - s.strike)
                # Put buyer pain
                if test_strike.strike < s.strike:
                    pain += s.put_oi * (s.strike - test_strike.strike)

            if pain < min_pain:
                min_pain = pain
                max_pain_strike = test_strike.strike

        return max_pain_strike

    def analyze_mean_reversion(self, candles: List[List], current_price: float) -> MeanReversionSignal:
        """Mean reversion analysis for directional context"""

        if not candles or len(candles) < 20:
            print(f"  Warning: Insufficient candles for mean reversion ({len(candles) if candles else 0})")
            return MeanReversionSignal(
                current_price=current_price,
                sma_20=current_price,
                sma_50=current_price,
                z_score=0,
                bollinger_position=0.5,
                signal="neutral",
                trend="sideways"
            )

        # Candles from Upstox are in reverse chronological order
        # Format: [timestamp, open, high, low, close, volume, oi]
        closes = []
        for c in candles:
            if isinstance(c, list) and len(c) >= 5:
                closes.append(c[4])  # Close price at index 4

        if len(closes) < 20:
            print(f"  Warning: Only {len(closes)} valid closes for mean reversion")
            return MeanReversionSignal(
                current_price=current_price,
                sma_20=current_price,
                sma_50=current_price,
                z_score=0,
                bollinger_position=0.5,
                signal="neutral",
                trend="sideways"
            )

        # Reverse to chronological order (oldest first)
        closes = closes[::-1]

        # Use current price as the latest data point
        closes.append(current_price)

        # Calculate SMAs (using most recent data)
        sma_20 = statistics.mean(closes[-20:])
        sma_50 = statistics.mean(closes[-50:]) if len(closes) >= 50 else statistics.mean(closes)

        # Calculate Z-score (how many std devs from 20-day mean)
        std_20 = statistics.stdev(closes[-20:])
        z_score = (current_price - sma_20) / std_20 if std_20 > 0 else 0

        # Bollinger position (0 = lower band, 1 = upper band)
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_range = bb_upper - bb_lower
        bollinger_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        bollinger_position = max(0, min(1, bollinger_position))

        # Determine signal
        if z_score >= Config.ZSCORE_OVERBOUGHT:
            signal = "overbought"
        elif z_score <= Config.ZSCORE_OVERSOLD:
            signal = "oversold"
        else:
            signal = "neutral"

        # Determine trend
        if current_price > sma_20 > sma_50:
            trend = "uptrend"
        elif current_price < sma_20 < sma_50:
            trend = "downtrend"
        else:
            trend = "sideways"

        print(f"  Mean Reversion: Price={current_price:.2f}, SMA20={sma_20:.2f}, SMA50={sma_50:.2f}, Z={z_score:.2f}")

        return MeanReversionSignal(
            current_price=round(current_price, 2),
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            z_score=round(z_score, 2),
            bollinger_position=round(bollinger_position, 2),
            signal=signal,
            trend=trend
        )

    def calculate_costs(self, premium: float, lot_size: int,
                        brokerage_per_order: float = 20) -> CostAnalysis:
        """Calculate all trading costs - India specific"""

        premium_value = premium * lot_size

        # STT on sell side
        stt = premium_value * Config.STT_RATE_OPTIONS_SELL

        # SEBI charges
        sebi = premium_value * Config.SEBI_CHARGES

        # Stamp duty on buy
        stamp = premium_value * Config.STAMP_DUTY

        # Brokerage (assuming flat fee model like Zerodha/Upstox)
        brokerage = brokerage_per_order * 2  # Buy + Sell

        # GST on brokerage
        gst = brokerage * Config.GST_RATE

        total_cost = stt + sebi + stamp + brokerage + gst

        # Impact on breakeven (as points)
        breakeven_impact = total_cost / lot_size

        return CostAnalysis(
            estimated_brokerage_per_lot=brokerage,
            stt_estimate=round(stt, 2),
            total_cost_per_lot=round(total_cost, 2),
            breakeven_points_impact=round(breakeven_impact, 2)
        )


# ============================================================================
# EXPIRY SELECTOR
# ============================================================================

class ExpirySelector:
    """Selects appropriate expiries for analysis"""

    @staticmethod
    def categorize_expiries(expiries_data: List[dict]) -> dict:
        """
        Categorize expiries into weekly and monthly
        Returns dict with 'weekly' and 'monthly' lists
        """
        today = datetime.now()

        weekly = []
        monthly = []

        for exp in expiries_data:
            exp_date_str = exp.get('expiry', '')
            if not exp_date_str:
                continue

            try:
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
            except:
                continue

            dte = (exp_date - today).days

            if dte < 0:  # Skip expired
                continue

            # Check if it's a monthly expiry (last Thursday of month)
            # Monthly expiries in India are typically the last Thursday
            is_monthly = ExpirySelector._is_monthly_expiry(exp_date)

            expiry_info = {
                'date': exp_date_str,
                'dte': dte,
                'is_monthly': is_monthly
            }

            if is_monthly:
                monthly.append(expiry_info)
            else:
                weekly.append(expiry_info)

        # Sort by DTE
        weekly = sorted(weekly, key=lambda x: x['dte'])
        monthly = sorted(monthly, key=lambda x: x['dte'])

        return {'weekly': weekly, 'monthly': monthly}

    @staticmethod
    def _is_monthly_expiry(exp_date: datetime) -> bool:
        """Check if date is a monthly expiry (last Thursday of month)"""
        # Find last Thursday of the month
        next_month = exp_date.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)

        # Find last Thursday
        days_until_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_until_thursday)

        return exp_date.date() == last_thursday.date()

    @staticmethod
    def select_expiries(categorized: dict, max_weekly: int = 4,
                        include_monthly: bool = True) -> List[dict]:
        """Select expiries for analysis"""
        selected = []

        # Add weekly expiries (up to max_weekly)
        for exp in categorized['weekly'][:max_weekly]:
            exp['type'] = 'weekly'
            selected.append(exp)

        # Add monthly expiry if requested and within reasonable range
        if include_monthly and categorized['monthly']:
            # Get first monthly that's at least 7 days out
            for monthly in categorized['monthly']:
                if monthly['dte'] >= 7:
                    monthly['type'] = 'monthly'
                    # Check if not already included
                    if monthly['date'] not in [s['date'] for s in selected]:
                        selected.append(monthly)
                    break

        return sorted(selected, key=lambda x: x['dte'])


# ============================================================================
# MAIN DATA COLLECTOR
# ============================================================================

class OptionsDataCollector:
    """Main class that orchestrates data collection and analysis"""

    def __init__(self, access_token: str):
        self.client = UpstoxClient(access_token)
        self.analyzer = OptionsAnalyzer(self.client)

    def collect_full_analysis(self, underlying: str = "NIFTY",
                              expiry_date: str = None,
                              brokerage_per_order: float = 20,
                              multi_expiry: bool = True) -> dict:
        """
        Collect and analyze all data for options trading decision

        Args:
            underlying: NIFTY or BANKNIFTY
            expiry_date: Specific expiry (if None, auto-selects)
            brokerage_per_order: Brokerage cost
            multi_expiry: If True, collect multiple expiries (weekly + monthly)

        Returns a comprehensive JSON structure for LLM analysis
        """

        # Determine instrument keys
        if underlying.upper() == "NIFTY":
            instrument_key = Config.NIFTY_INDEX
            lot_size = Config.NIFTY_LOT_SIZE
        elif underlying.upper() == "BANKNIFTY":
            instrument_key = Config.BANKNIFTY_INDEX
            lot_size = Config.BANKNIFTY_LOT_SIZE
        else:
            raise ValueError(f"Unsupported underlying: {underlying}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 60}")
        print(f"Options Data Collection - {underlying}")
        print(f"Timestamp: {timestamp}")
        print(f"{'=' * 60}")

        # Get all expiry dates
        print("\n1. Fetching available expiries...")
        expiries_response = self.client.get_option_expiries(instrument_key)

        if expiries_response.get('status') != 'success':
            print(f"   Error fetching expiries: {expiries_response}")
            return {"error": "Failed to fetch expiries"}

        # Categorize and select expiries
        all_expiries = expiries_response.get('data', [])
        categorized = ExpirySelector.categorize_expiries(all_expiries)

        print(f"   Found {len(categorized['weekly'])} weekly expiries")
        print(f"   Found {len(categorized['monthly'])} monthly expiries")

        if multi_expiry:
            selected_expiries = ExpirySelector.select_expiries(
                categorized,
                max_weekly=Config.MAX_WEEKLY_EXPIRIES,
                include_monthly=Config.INCLUDE_MONTHLY
            )
        elif expiry_date:
            selected_expiries = [{'date': expiry_date, 'dte': 0, 'type': 'specified'}]
        else:
            # Default: select optimal expiry (30-60 DTE)
            selected_expiries = ExpirySelector.select_expiries(
                categorized, max_weekly=1, include_monthly=True
            )

        print(f"   Selected {len(selected_expiries)} expiries for analysis:")
        for exp in selected_expiries:
            print(f"      - {exp['date']} ({exp['dte']} DTE, {exp.get('type', 'unknown')})")

        # Get spot price first (from first option chain)
        print("\n2. Fetching spot price...")
        first_chain = self.client.get_option_chain(instrument_key, selected_expiries[0]['date'])
        spot_price = 0
        if first_chain.get('status') == 'success' and first_chain.get('data'):
            spot_price = first_chain['data'][0].get('underlying_spot_price', 0)
        print(f"   Spot price: {spot_price}")

        # Get historical data for HV and mean reversion
        print("\n3. Fetching historical data...")
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=75)).strftime("%Y-%m-%d")

        historical = self.client.get_historical_candles(
            instrument_key, "day", to_date, from_date
        )

        candles = []
        if historical.get('status') == 'success':
            candles = historical.get('data', {}).get('candles', [])
            print(f"   Retrieved {len(candles)} daily candles")
        else:
            print(f"   Warning: Failed to get historical data: {historical.get('error', 'Unknown error')}")

        # Get VIX
        print("\n4. Fetching India VIX...")
        vix_data = self.client.get_market_quote([Config.INDIA_VIX])
        vix_value = 15.0  # Default
        if vix_data.get('status') == 'success':
            vix_info = vix_data.get('data', {}).get(Config.INDIA_VIX, {})
            vix_value = vix_info.get('last_price', 15.0)
        print(f"   VIX: {vix_value}")

        # Calculate mean reversion signals
        print("\n5. Calculating mean reversion signals...")
        mean_reversion = self.analyzer.analyze_mean_reversion(candles, spot_price)

        # Collect option chain data for each expiry
        print("\n6. Collecting option chains for each expiry...")
        expiry_analyses = []
        primary_volatility_analysis = None

        for exp in selected_expiries:
            print(f"\n   Processing {exp['date']} ({exp['dte']} DTE)...")

            chain_data = self.client.get_option_chain(instrument_key, exp['date'])

            if chain_data.get('status') != 'success':
                print(f"      Error: {chain_data.get('error', 'Unknown')}")
                continue

            # Parse with strike filtering
            strikes = self.analyzer.parse_option_chain(chain_data, spot_price, filter_strikes=True)
            print(f"      Parsed {len(strikes)} strikes (within ±{Config.STRIKE_RANGE_PERCENT}% of spot)")

            if not strikes:
                continue

            # Find ATM
            atm_strike = self.analyzer.find_atm_strike(strikes, spot_price)

            # Analyze theta
            theta_analysis = self.analyzer.analyze_theta(atm_strike, exp['date'], spot_price)

            # Analyze OI
            oi_analysis = self.analyzer.analyze_oi(strikes, spot_price)

            # Volatility analysis (use first expiry as primary)
            if primary_volatility_analysis is None:
                primary_volatility_analysis = self.analyzer.analyze_volatility(
                    strikes, spot_price, candles, vix_value
                )

            expiry_data = {
                'expiry_date': exp['date'],
                'days_to_expiry': exp['dte'],
                'expiry_type': exp.get('type', 'unknown'),
                'total_strikes': len(strikes),
                'atm_strike': atm_strike.strike if atm_strike else None,
                'atm_call_iv': atm_strike.call_iv if atm_strike else None,
                'atm_put_iv': atm_strike.put_iv if atm_strike else None,
                'atm_straddle_price': round(atm_strike.call_ltp + atm_strike.put_ltp, 2) if atm_strike else None,
                'theta_analysis': asdict(theta_analysis) if theta_analysis else None,
                'oi_analysis': asdict(oi_analysis),
                'strikes_data': [asdict(s) for s in strikes]
            }

            expiry_analyses.append(expiry_data)

        # Cost analysis for ATM straddle (using first expiry)
        if expiry_analyses and expiry_analyses[0].get('atm_straddle_price'):
            atm_premium = expiry_analyses[0]['atm_straddle_price']
            cost_analysis = self.analyzer.calculate_costs(atm_premium, lot_size, brokerage_per_order)
        else:
            cost_analysis = CostAnalysis(40, 0, 40, 0)

        # Build market context
        market_context = {
            'timestamp': timestamp,
            'underlying_symbol': underlying,
            'spot_price': spot_price,
            'futures_price': spot_price * 1.001,  # Approximation
            'futures_premium': spot_price * 0.001,
            'day_change_percent': 0,
            'day_high': candles[0][2] if candles and len(candles[0]) > 2 else spot_price,
            'day_low': candles[0][3] if candles and len(candles[0]) > 3 else spot_price,
            'day_range_percent': 0
        }

        # Determine primary expiry for signals (first weekly or monthly based on DTE)
        primary_expiry = expiry_analyses[0] if expiry_analyses else None

        # Build comprehensive output
        output = {
            "metadata": {
                "timestamp": timestamp,
                "data_source": "Upstox API",
                "underlying": underlying,
                "lot_size": lot_size,
                "strike_range_filter": f"±{Config.STRIKE_RANGE_PERCENT}% of spot",
                "analysis_framework": "Comprehensive Options Analysis v1.1"
            },
            "market_context": market_context,
            "core_analysis": {
                "volatility": asdict(primary_volatility_analysis) if primary_volatility_analysis else None,
                "mean_reversion": asdict(mean_reversion)
            },
            "expiries": expiry_analyses,
            "cost_analysis": asdict(cost_analysis),
            "framework_signals": self._generate_framework_signals(
                primary_volatility_analysis,
                expiry_analyses[0]['theta_analysis'] if expiry_analyses else None,
                expiry_analyses[0]['oi_analysis'] if expiry_analyses else None,
                mean_reversion
            ),
            "warnings": self._generate_warnings(
                primary_volatility_analysis,
                expiry_analyses[0] if expiry_analyses else None,
                cost_analysis
            )
        }

        print(f"\n{'=' * 60}")
        print("Data collection complete!")
        print(f"{'=' * 60}")

        return output

    def _generate_framework_signals(self, vol: VolatilityAnalysis,
                                    theta_dict: dict,
                                    oi_dict: dict,
                                    mr: MeanReversionSignal) -> dict:
        """Generate actionable signals based on framework"""

        signals = {
            "volatility_signal": "neutral",
            "time_decay_signal": "neutral",
            "directional_bias": "neutral",
            "overall_bias": "neutral",
            "recommended_approach": "wait"
        }

        if not vol:
            return signals

        # Volatility signal (Truth #1 & #2)
        if vol.regime in ["high_vol", "extreme"] or vol.iv_hv_ratio > 1.2:
            signals["volatility_signal"] = "sell_premium"
        elif vol.regime == "low_vol" and vol.iv_hv_ratio < 0.9:
            signals["volatility_signal"] = "buy_premium_cautiously"

        # Time decay signal (Truth #2)
        if theta_dict:
            if theta_dict.get('is_optimal_dte'):
                signals["time_decay_signal"] = "favorable_for_selling"
            elif theta_dict.get('days_to_expiry', 0) <= 7:
                signals["time_decay_signal"] = "high_gamma_risk"

        # Directional bias from mean reversion + OI
        oi_bias = oi_dict.get('oi_buildup_bias', 'neutral') if oi_dict else 'neutral'

        if mr.signal == "oversold" and oi_bias == "bullish":
            signals["directional_bias"] = "bullish"
        elif mr.signal == "overbought" and oi_bias == "bearish":
            signals["directional_bias"] = "bearish"
        else:
            signals["directional_bias"] = mr.trend

        # Overall bias
        if signals["volatility_signal"] == "sell_premium":
            if signals["directional_bias"] in ["neutral", "sideways"]:
                signals["overall_bias"] = "neutral_premium_selling"
                signals["recommended_approach"] = "iron_condor_or_strangle"
            elif signals["directional_bias"] in ["bullish", "uptrend"]:
                signals["overall_bias"] = "bullish_premium_selling"
                signals["recommended_approach"] = "put_credit_spread"
            else:
                signals["overall_bias"] = "bearish_premium_selling"
                signals["recommended_approach"] = "call_credit_spread"
        elif signals["volatility_signal"] == "buy_premium_cautiously":
            signals["overall_bias"] = "directional_play"
            signals["recommended_approach"] = "debit_spread_if_conviction"

        return signals

    def _generate_warnings(self, vol: VolatilityAnalysis,
                           primary_expiry: dict,
                           costs: CostAnalysis) -> List[str]:
        """Generate risk warnings"""
        warnings = []

        if vol and vol.regime == "extreme":
            warnings.append("EXTREME VOLATILITY: VIX elevated - increased tail risk")

        if primary_expiry:
            dte = primary_expiry.get('days_to_expiry', 0)
            if dte <= 7:
                warnings.append("HIGH GAMMA RISK: <7 DTE - rapid price swings possible")

        if costs and costs.breakeven_points_impact > 5:
            warnings.append(f"HIGH COST IMPACT: Costs add {costs.breakeven_points_impact} points to breakeven")

        if vol and vol.iv_skew == "put_premium":
            warnings.append("PUT SKEW ELEVATED: Market pricing downside risk higher")

        return warnings


# ============================================================================
# OUTPUT GENERATOR
# ============================================================================

def generate_json_output(data: dict, output_file: str = None) -> str:
    """Generate formatted JSON output"""
    json_str = json.dumps(data, indent=2, default=str)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(json_str)
        print(f"\nOutput saved to {output_file}")

    return json_str


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    import os

    # Read from environment variable (secure approach)
    ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "")

    # Check if token is set
    if not ACCESS_TOKEN:
        print("=" * 60)
        print("DEMO MODE - Using sample data structure")
        print("=" * 60)
        print("\nTo use live data, set your Upstox access token:")
        print("\n  Windows (CMD):    set UPSTOX_ACCESS_TOKEN=your_token_here")
        print("  Windows (PS):     $env:UPSTOX_ACCESS_TOKEN='your_token_here'")
        print("  Linux/Mac:        export UPSTOX_ACCESS_TOKEN=your_token_here")
        print("\nOr set it permanently in Windows Environment Variables.")
        print("=" * 60)

        # Generate sample output structure for demonstration
        sample_output = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "DEMO - Sample Data",
                "underlying": "NIFTY",
                "lot_size": 25,
                "analysis_framework": "Comprehensive Options Analysis v1.1"
            },
            "message": "Set UPSTOX_ACCESS_TOKEN environment variable for live data"
        }

        print(json.dumps(sample_output, indent=2))
        return sample_output

    # Live data collection
    collector = OptionsDataCollector(ACCESS_TOKEN)

    # Collect analysis for NIFTY with multiple expiries
    analysis = collector.collect_full_analysis(
        underlying="NIFTY",
        brokerage_per_order=20,
        multi_expiry=True  # Get weekly + monthly expiries
    )

    # Output JSON
    output = generate_json_output(analysis, "market_analysis.json")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options Trading Data Collector")
    parser.add_argument("--test", action="store_true", help="Test API connection only")
    parser.add_argument("--underlying", default="NIFTY", choices=["NIFTY", "BANKNIFTY"],
                        help="Underlying to analyze (default: NIFTY)")
    parser.add_argument("--expiry", default=None,
                        help="Specific expiry date (YYYY-MM-DD), auto-selects if not provided")
    parser.add_argument("--output", default="market_analysis.json", help="Output file name")
    parser.add_argument("--single-expiry", action="store_true", help="Only fetch single (optimal) expiry")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter strikes by range")

    args = parser.parse_args()

    if args.test:
        # Quick connection test
        import os

        token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        if not token:
            print("❌ UPSTOX_ACCESS_TOKEN not found in environment variables")
            print("\nSet it using:")
            print("  Windows: set UPSTOX_ACCESS_TOKEN=your_token")
            print("  Or add to Windows Environment Variables permanently")
        else:
            print("✅ Token found in environment variables")
            print(f"   Token starts with: {token[:10]}...")
            print("\nTesting API connection...")

            client = UpstoxClient(token)
            # Try to get option expiries as a simple test
            result = client.get_option_expiries(Config.NIFTY_INDEX)

            if result.get("status") == "success":
                print("✅ API connection successful!")
                expiries = result.get("data", [])
                if expiries:
                    print(f"   Found {len(expiries)} expiry dates")

                    # Categorize
                    categorized = ExpirySelector.categorize_expiries(expiries)
                    print(f"   Weekly expiries: {len(categorized['weekly'])}")
                    print(f"   Monthly expiries: {len(categorized['monthly'])}")

                    if categorized['weekly']:
                        print(f"   Next weekly: {categorized['weekly'][0]['date']}")
                    if categorized['monthly']:
                        print(f"   Next monthly: {categorized['monthly'][0]['date']}")
            else:
                print("❌ API connection failed!")
                print(f"   Error: {result.get('error', result)}")
    else:
        import os

        token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")

        if not token:
            main()  # Will show demo mode message
        else:
            # Run full analysis
            collector = OptionsDataCollector(token)

            result = collector.collect_full_analysis(
                underlying=args.underlying,
                expiry_date=args.expiry,
                multi_expiry=not args.single_expiry
            )

            # Save to file
            if result and "metadata" in result:
                output_file = args.output
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\n✅ Analysis saved to: {output_file}")