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
- Trading Zone: Actionable ±5 ATM strikes with bid/ask, liquidity, spread suggestions

Author: Options Trading System
Version: 2.1

Changes from v2.0:
- Added Trading Zone with ±5 strikes around ATM
- Added bid/ask spread analysis for liquidity assessment
- Added automatic spread suggestions (Bull Put, Bear Call, Iron Condor)
- Added expected move calculation from ATM straddle
- Added liquidity scoring and recommendations

Changes from v1.1:
- Added retry logic with exponential backoff for API calls
- Fetch actual Futures LTP instead of approximation
- Fail-fast on critical data errors (spot price, corrupted data)
- Warn and continue on non-critical errors with warnings in output
- Zero-division guards throughout
- Rate limiting between API calls
- IV Percentile approximation using VIX
- Strike range default changed to 10%
- Configurable API base URL (v2/v3 ready)
- Structured warnings and data quality flags
- Numeric safety checks for Greeks
"""

import requests
import json
import gzip
import io
import math
import time
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, asdict, field
import statistics
from enum import Enum


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class OptionsDataError(Exception):
    """Base exception for options data collector"""
    pass


class CriticalDataError(OptionsDataError):
    """Raised when critical data cannot be fetched and analysis cannot continue"""
    pass


class APIError(OptionsDataError):
    """Raised when API calls fail after retries"""
    pass


class DataQualityError(OptionsDataError):
    """Raised when data quality is too poor to produce reliable analysis"""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the data collector"""

    # Upstox API Base URL (configurable for v2/v3 migration)
    API_VERSION = "v2"
    BASE_URL = f"https://api.upstox.com/{API_VERSION}"

    # Instrument Keys
    NIFTY_INDEX = "NSE_INDEX|Nifty 50"
    BANKNIFTY_INDEX = "NSE_INDEX|Nifty Bank"
    INDIA_VIX = "NSE_INDEX|India VIX"

    # Futures Instrument Key Patterns (for fetching actual futures price)
    # Format: NSE_FO|NIFTY{YYMM}FUT
    #NIFTY_FUT_PREFIX = "NSE_FO|NIFTY"
    #BANKNIFTY_FUT_PREFIX = "NSE_FO|BANKNIFTY"

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
    # Changed from 20% to 10% - more focused on liquid strikes
    STRIKE_RANGE_PERCENT = 10  # ±10% of spot price

    # Expiry Selection
    MAX_WEEKLY_EXPIRIES = 4  # Number of weekly expiries to capture
    INCLUDE_MONTHLY = True  # Include monthly expiry

    # API Reliability Settings
    MAX_RETRIES = 3  # Number of retry attempts for failed API calls
    RETRY_BASE_DELAY = 1.0  # Base delay in seconds (will be exponentially increased)
    RATE_LIMIT_DELAY = 0.3  # Delay between consecutive API calls (seconds)

    # VIX Range for IV Percentile Approximation
    # Historical VIX typically ranges from 10 (calm) to 40+ (crisis)
    VIX_HISTORICAL_LOW = 10.0
    VIX_HISTORICAL_HIGH = 35.0
    VIX_TYPICAL_MEDIAN = 15.0

    # Trading Zone Configuration
    TRADING_ZONE_STRIKES_ABOVE = 5  # Number of strikes above ATM
    TRADING_ZONE_STRIKES_BELOW = 5  # Number of strikes below ATM
    BID_ASK_SPREAD_GOOD_THRESHOLD = 1.0  # Below 1% = good liquidity
    BID_ASK_SPREAD_POOR_THRESHOLD = 3.0  # Above 3% = poor liquidity


# ============================================================================
# DATA QUALITY TRACKING
# ============================================================================

class WarningLevel(Enum):
    """Warning severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DataWarning:
    """Structured warning for data quality issues"""
    level: str
    category: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp
        }


class WarningCollector:
    """Collects warnings throughout the analysis process"""

    def __init__(self):
        self.warnings: List[DataWarning] = []

    def add(self, level: WarningLevel, category: str, message: str):
        """Add a warning"""
        self.warnings.append(DataWarning(
            level=level.value,
            category=category,
            message=message
        ))
        # Also print for immediate feedback
        prefix = "⚠️" if level == WarningLevel.WARNING else "ℹ️" if level == WarningLevel.INFO else "🚨"
        print(f"   {prefix} [{category}] {message}")

    def get_all(self) -> List[dict]:
        """Get all warnings as dicts"""
        return [w.to_dict() for w in self.warnings]

    def has_critical(self) -> bool:
        """Check if any critical warnings exist"""
        return any(w.level == WarningLevel.CRITICAL.value for w in self.warnings)

    def count_by_level(self) -> dict:
        """Count warnings by level"""
        counts = {"info": 0, "warning": 0, "critical": 0}
        for w in self.warnings:
            counts[w.level] = counts.get(w.level, 0) + 1
        return counts


# ============================================================================
# RETRY DECORATOR
# ============================================================================

def retry_with_backoff(
        max_retries: int = Config.MAX_RETRIES,
        base_delay: float = Config.RETRY_BASE_DELAY,
        exceptions: tuple = (requests.exceptions.RequestException,)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        exceptions: Tuple of exceptions to catch and retry on
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"      Retry {attempt + 1}/{max_retries} after {delay}s due to: {str(e)[:50]}...")
                        time.sleep(delay)
                    else:
                        print(f"      All {max_retries} retries exhausted for {func.__name__}")

            # All retries exhausted
            raise APIError(f"API call failed after {max_retries} retries: {last_exception}")

        return wrapper

    return decorator


# ============================================================================
# SAFE MATH UTILITIES
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator"""
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Safe mean calculation that handles empty lists and None values"""
    filtered = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        filtered.append(v)
    if not filtered:
        return default
    return statistics.mean(filtered)


def safe_stdev(values: List[float], default: float = 0.0) -> float:
    """Safe standard deviation that handles edge cases"""
    filtered = [v for v in values if v is not None and (not isinstance(v, float) or not math.isnan(v))]
    if len(filtered) < 2:
        return default
    return statistics.stdev(filtered)


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
    # Data quality flags
    has_valid_greeks: bool = True


@dataclass
class VolatilityAnalysis:
    """Volatility analysis results"""
    current_iv_atm: float
    iv_percentile: float  # Approximated from VIX
    iv_rank: float  # Approximated from VIX
    historical_volatility_20d: float
    iv_hv_ratio: float  # IV/HV - above 1 means IV is elevated
    iv_skew: str  # "call_premium", "put_premium", "neutral"
    vix_current: float
    vix_percentile: float  # Approximated from historical range
    regime: str  # "low_vol", "normal", "high_vol", "extreme"
    data_quality: str  # "good", "degraded", "poor"


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
    data_quality: str  # "good", "degraded", "insufficient"


@dataclass
class CostAnalysis:
    """Trading cost estimates"""
    estimated_brokerage_per_lot: float
    stt_estimate: float
    total_cost_per_lot: float
    breakeven_points_impact: float  # How much costs affect breakeven


@dataclass
class FuturesData:
    """Futures price data for accurate basis calculation"""
    futures_price: float
    spot_price: float
    basis: float  # Futures - Spot
    basis_percent: float  # Basis as % of spot
    is_premium: bool  # True if futures > spot (contango)
    is_actual: bool  # True if fetched from API, False if approximated
    expiry_date: str


@dataclass
class StrikeDetail:
    """Detailed strike information for trading zone"""
    strike: float
    moneyness: str  # "ITM", "ATM", "OTM"
    distance_from_atm: float  # In points
    distance_percent: float  # As percentage of spot
    ce_ltp: float
    ce_bid: float
    ce_ask: float
    ce_bid_ask_spread_pct: float
    ce_iv: float
    ce_delta: float
    ce_theta: float
    ce_oi: int
    ce_volume: int
    ce_liquidity: str  # "good", "moderate", "poor"
    pe_ltp: float
    pe_bid: float
    pe_ask: float
    pe_bid_ask_spread_pct: float
    pe_iv: float
    pe_delta: float
    pe_theta: float
    pe_oi: int
    pe_volume: int
    pe_liquidity: str


@dataclass
class SpreadSuggestion:
    """Suggested spread strikes based on OI and liquidity"""
    strategy: str
    sell_strike: float
    sell_premium: float
    buy_strike: float
    buy_premium: float
    net_credit: float
    max_risk: float
    risk_reward_ratio: float
    liquidity_score: str
    rationale: str


@dataclass
class TradingZone:
    """Focused view of tradeable strikes around ATM"""
    reference_spot: float
    atm_strike: float
    atm_straddle_price: float
    atm_straddle_bid: float  # Conservative entry
    atm_straddle_ask: float  # Aggressive entry
    expected_move: float  # ATM straddle price as expected move
    expected_move_percent: float
    strikes: List[dict]  # List of StrikeDetail as dicts
    liquidity_summary: dict
    spread_suggestions: List[dict]  # List of SpreadSuggestion as dicts


@dataclass
class MarketContext:
    """Overall market context"""
    timestamp: str
    underlying_symbol: str
    spot_price: float
    futures_data: dict  # FuturesData as dict
    day_change_percent: float
    day_high: float
    day_low: float
    day_range_percent: float


# ============================================================================
# UPSTOX API CLIENT
# ============================================================================

class UpstoxClient:
    """Client for interacting with Upstox API"""

    def __init__(self, access_token: str, raise_on_error: bool = False):
        """
        Initialize Upstox client.

        Args:
            access_token: Valid Upstox API access token
            raise_on_error: If True, raise exceptions on API errors instead of returning error dict
        """
        self.access_token = access_token
        self.raise_on_error = raise_on_error
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < Config.RATE_LIMIT_DELAY:
            time.sleep(Config.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    @retry_with_backoff(max_retries=Config.MAX_RETRIES)
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling and retry logic"""
        self._rate_limit()

        url = f"{Config.BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            # Handle rate limiting specifically
            if response.status_code == 429:
                raise requests.exceptions.RequestException("Rate limit exceeded (429)")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise requests.exceptions.RequestException("Request timed out")
        except requests.exceptions.RequestException as e:
            if self.raise_on_error:
                raise APIError(f"API request failed: {e}")
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

    def get_futures_instrument_key(self, underlying: str) -> Optional[str]:
        """
        Get current month futures instrument key by downloading instrument file.
        """
        try:
            print(f"      Downloading NSE instruments file...")
            url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
                instruments = json.loads(f.read().decode('utf-8'))

            now = datetime.now()
            underlying_name = "NIFTY" if underlying.upper() == "NIFTY" else "BANKNIFTY"

            futures_contracts = []
            for inst in instruments:
                if (inst.get('segment') == 'NSE_FO' and
                        inst.get('instrument_type') == 'FUT' and
                        inst.get('underlying_symbol') == underlying_name):

                    expiry_ms = inst.get('expiry', 0)
                    if expiry_ms:
                        expiry_date = datetime.fromtimestamp(expiry_ms / 1000)
                        if expiry_date > now:
                            futures_contracts.append({
                                'instrument_key': inst.get('instrument_key'),
                                'expiry': expiry_date,
                                'trading_symbol': inst.get('trading_symbol')
                            })

            if futures_contracts:
                futures_contracts.sort(key=lambda x: x['expiry'])
                nearest = futures_contracts[0]
                print(f"      Found: {nearest['trading_symbol']} -> {nearest['instrument_key']}")
                return nearest['instrument_key']

            return None

        except Exception as e:
            print(f"      Error fetching instruments file: {e}")
            return None

    def get_futures_quote(self, underlying: str) -> dict:
        """
        Get current month futures quote for accurate basis calculation.
        """
        fut_key = self.get_futures_instrument_key(underlying)

        if fut_key:
            return self.get_market_quote([fut_key])
        else:
            return {"status": "error", "error": f"Could not find futures instrument key for {underlying}"}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

class OptionsAnalyzer:
    """Analyzes options data based on the Comprehensive Framework"""

    def __init__(self, client: UpstoxClient, warnings: WarningCollector):
        self.client = client
        self.warnings = warnings

    def parse_option_chain(self, chain_data: dict, spot_price: float,
                           filter_strikes: bool = True) -> List[OptionStrike]:
        """Parse raw option chain data into structured format with optional filtering"""
        strikes = []

        if chain_data.get('status') != 'success':
            self.warnings.add(
                WarningLevel.WARNING,
                "option_chain",
                f"Option chain fetch failed: {chain_data.get('error', 'Unknown error')}"
            )
            return strikes

        # Calculate strike range if filtering
        if filter_strikes and spot_price > 0:
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

                # Check if Greeks are valid (non-zero for liquid options)
                call_iv = call_greeks.get('iv', 0) or 0
                put_iv = put_greeks.get('iv', 0) or 0
                has_valid_greeks = (call_iv > 0 or put_iv > 0)

                strike = OptionStrike(
                    strike=strike_price,
                    call_ltp=call_market.get('ltp', 0) or 0,
                    call_iv=call_iv,
                    call_delta=call_greeks.get('delta', 0) or 0,
                    call_theta=call_greeks.get('theta', 0) or 0,
                    call_gamma=call_greeks.get('gamma', 0) or 0,
                    call_vega=call_greeks.get('vega', 0) or 0,
                    call_oi=call_market.get('oi', 0) or 0,
                    call_volume=call_market.get('volume', 0) or 0,
                    call_bid=call_market.get('bid_price', 0) or 0,
                    call_ask=call_market.get('ask_price', 0) or 0,
                    call_pop=call_greeks.get('pop', 0) or 0,
                    put_ltp=put_market.get('ltp', 0) or 0,
                    put_iv=put_iv,
                    put_delta=put_greeks.get('delta', 0) or 0,
                    put_theta=put_greeks.get('theta', 0) or 0,
                    put_gamma=put_greeks.get('gamma', 0) or 0,
                    put_vega=put_greeks.get('vega', 0) or 0,
                    put_oi=put_market.get('oi', 0) or 0,
                    put_volume=put_market.get('volume', 0) or 0,
                    put_bid=put_market.get('bid_price', 0) or 0,
                    put_ask=put_market.get('ask_price', 0) or 0,
                    put_pop=put_greeks.get('pop', 0) or 0,
                    pcr=item.get('pcr', 0) or 0,
                    has_valid_greeks=has_valid_greeks
                )
                strikes.append(strike)
            except Exception as e:
                self.warnings.add(
                    WarningLevel.INFO,
                    "parsing",
                    f"Error parsing strike {item.get('strike_price', 'unknown')}: {e}"
                )
                continue

        return sorted(strikes, key=lambda x: x.strike)

    def find_atm_strike(self, strikes: List[OptionStrike], spot_price: float) -> Optional[OptionStrike]:
        """Find the At-The-Money strike"""
        if not strikes:
            return None
        return min(strikes, key=lambda x: abs(x.strike - spot_price))

    def calculate_historical_volatility(self, candles: List[List]) -> Tuple[float, str]:
        """
        Calculate 20-day historical volatility from daily candles.

        Returns:
            Tuple of (volatility_value, data_quality)
        """
        if not candles:
            self.warnings.add(
                WarningLevel.WARNING,
                "historical_volatility",
                "No candle data available for HV calculation"
            )
            return 0.0, "insufficient"

        if len(candles) < 15:
            self.warnings.add(
                WarningLevel.WARNING,
                "historical_volatility",
                f"Insufficient candles for HV calculation ({len(candles)} candles, need 15+)"
            )
            return 0.0, "insufficient"

        # Candles are typically in reverse chronological order from Upstox
        # Format: [timestamp, open, high, low, close, volume, oi]
        recent_candles = candles[:min(len(candles), 25)]

        # Extract closing prices (index 4 in candle array)
        closes = []
        for c in recent_candles:
            if isinstance(c, list) and len(c) >= 5:
                close_price = c[4]
                if close_price is not None and close_price > 0:
                    closes.append(close_price)

        if len(closes) < 10:
            self.warnings.add(
                WarningLevel.WARNING,
                "historical_volatility",
                f"Only {len(closes)} valid closes found, need 10+"
            )
            return 0.0, "insufficient"

        # Reverse to chronological order for proper return calculation
        closes = closes[::-1]

        # Calculate log returns
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                returns.append(math.log(closes[i] / closes[i - 1]))

        if len(returns) < 5:
            self.warnings.add(
                WarningLevel.WARNING,
                "historical_volatility",
                f"Only {len(returns)} returns calculated, need 5+"
            )
            return 0.0, "degraded"

        # Calculate annualized volatility
        std_dev = safe_stdev(returns, 0.0)
        if std_dev == 0:
            return 0.0, "degraded"

        annualized_vol = std_dev * math.sqrt(252) * 100  # As percentage

        data_quality = "good" if len(returns) >= 15 else "degraded"

        return round(annualized_vol, 2), data_quality

    def approximate_iv_percentile_from_vix(self, vix_value: float) -> Tuple[float, float]:
        """
        Approximate IV percentile and rank using VIX as a proxy.

        This is an approximation since we don't have historical IV data.
        VIX typically ranges from 10 (extremely calm) to 35+ (crisis).

        Returns:
            Tuple of (iv_percentile, iv_rank)
        """
        vix_low = Config.VIX_HISTORICAL_LOW
        vix_high = Config.VIX_HISTORICAL_HIGH

        # Clamp VIX to reasonable range
        clamped_vix = max(vix_low, min(vix_high, vix_value))

        # IV Rank: (Current - Min) / (Max - Min)
        iv_rank = ((clamped_vix - vix_low) / (vix_high - vix_low)) * 100

        # IV Percentile: Approximate using a sigmoid-like distribution
        # Most of the time VIX is 12-18, so we weight accordingly
        # This is a rough approximation
        if vix_value <= 12:
            iv_percentile = 10 + (vix_value - vix_low) * 5
        elif vix_value <= 15:
            iv_percentile = 20 + (vix_value - 12) * 10
        elif vix_value <= 18:
            iv_percentile = 50 + (vix_value - 15) * 8
        elif vix_value <= 22:
            iv_percentile = 74 + (vix_value - 18) * 4
        elif vix_value <= 28:
            iv_percentile = 90 + (vix_value - 22) * 1.5
        else:
            iv_percentile = min(99, 95 + (vix_value - 28) * 0.5)

        return round(iv_percentile, 1), round(iv_rank, 1)

    def analyze_volatility(self, strikes: List[OptionStrike], spot_price: float,
                           historical_candles: List, vix_value: float) -> Optional[VolatilityAnalysis]:
        """Comprehensive volatility analysis - CORE FACTOR #1 & #2"""

        atm = self.find_atm_strike(strikes, spot_price)
        if not atm:
            self.warnings.add(
                WarningLevel.WARNING,
                "volatility",
                "No ATM strike found for volatility analysis"
            )
            return None

        # ATM IV (average of call and put, with fallback)
        if atm.call_iv > 0 and atm.put_iv > 0:
            current_iv_atm = (atm.call_iv + atm.put_iv) / 2
        elif atm.call_iv > 0:
            current_iv_atm = atm.call_iv
        elif atm.put_iv > 0:
            current_iv_atm = atm.put_iv
        else:
            self.warnings.add(
                WarningLevel.WARNING,
                "volatility",
                "ATM IV is zero - Greeks may be unavailable for this strike"
            )
            current_iv_atm = 0

        # Historical Volatility
        hv_20d, hv_quality = self.calculate_historical_volatility(historical_candles)

        # IV/HV Ratio - key metric from framework
        iv_hv_ratio = safe_divide(current_iv_atm, hv_20d, 1.0)

        # IV Skew analysis
        otm_calls = [s for s in strikes if s.strike > spot_price * 1.02 and s.call_iv > 0]
        otm_puts = [s for s in strikes if s.strike < spot_price * 0.98 and s.put_iv > 0]

        avg_call_iv = safe_mean([s.call_iv for s in otm_calls], 0)
        avg_put_iv = safe_mean([s.put_iv for s in otm_puts], 0)

        if avg_put_iv > 0 and avg_call_iv > 0:
            if avg_put_iv > avg_call_iv * 1.1:
                iv_skew = "put_premium"  # Fear in market
            elif avg_call_iv > avg_put_iv * 1.1:
                iv_skew = "call_premium"  # Greed/euphoria
            else:
                iv_skew = "neutral"
        else:
            iv_skew = "unknown"
            self.warnings.add(
                WarningLevel.INFO,
                "volatility",
                "Insufficient OTM data for skew analysis"
            )

        # Determine volatility regime
        if vix_value < 13:
            regime = "low_vol"
        elif vix_value < 18:
            regime = "normal"
        elif vix_value < 25:
            regime = "high_vol"
        else:
            regime = "extreme"

        # Approximate IV percentile using VIX
        iv_percentile, iv_rank = self.approximate_iv_percentile_from_vix(vix_value)

        # Determine data quality
        if current_iv_atm == 0 or hv_quality == "insufficient":
            data_quality = "poor"
        elif hv_quality == "degraded" or iv_skew == "unknown":
            data_quality = "degraded"
        else:
            data_quality = "good"

        return VolatilityAnalysis(
            current_iv_atm=round(current_iv_atm, 2),
            iv_percentile=iv_percentile,
            iv_rank=iv_rank,
            historical_volatility_20d=hv_20d,
            iv_hv_ratio=round(iv_hv_ratio, 2),
            iv_skew=iv_skew,
            vix_current=vix_value,
            vix_percentile=iv_percentile,  # Same approximation
            regime=regime,
            data_quality=data_quality
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

        # ATM theta per day (average of call and put, handle None/0)
        call_theta = abs(atm_strike.call_theta) if atm_strike.call_theta else 0
        put_theta = abs(atm_strike.put_theta) if atm_strike.put_theta else 0

        if call_theta > 0 and put_theta > 0:
            atm_theta = (call_theta + put_theta) / 2
        else:
            atm_theta = call_theta or put_theta

        # Theta capture potential (time value as % of premium)
        atm_premium = (atm_strike.call_ltp + atm_strike.put_ltp) / 2
        intrinsic_call = max(0, spot_price - atm_strike.strike)
        intrinsic_put = max(0, atm_strike.strike - spot_price)
        time_value = atm_premium - (intrinsic_call + intrinsic_put) / 2
        theta_capture = safe_divide(time_value, atm_premium, 0) * 100

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

        # Safe division for PCR
        pcr_oi = safe_divide(total_put_oi, total_call_oi, 1.0)
        pcr_volume = safe_divide(total_put_vol, total_call_vol, 1.0)

        # Max Pain calculation (strike where total option buyer loss is maximum)
        max_pain = self._calculate_max_pain(strikes, spot_price)

        # Find significant OI strikes (within reasonable range)
        relevant_strikes = [s for s in strikes if abs(s.strike - spot_price) / spot_price < 0.10]

        if relevant_strikes:
            avg_call_oi = safe_divide(
                sum(s.call_oi for s in relevant_strikes),
                len(relevant_strikes),
                1
            )
            avg_put_oi = safe_divide(
                sum(s.put_oi for s in relevant_strikes),
                len(relevant_strikes),
                1
            )
        else:
            avg_call_oi = safe_divide(total_call_oi, len(strikes), 1) if strikes else 1
            avg_put_oi = safe_divide(total_put_oi, len(strikes), 1) if strikes else 1

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
            pcr_oi=round(pcr_oi, 2),
            pcr_volume=round(pcr_volume, 2),
            max_pain=max_pain,
            significant_call_strikes=significant_calls,
            significant_put_strikes=significant_puts,
            oi_buildup_bias=oi_bias
        )

    def _calculate_max_pain(self, strikes: List[OptionStrike], spot_price: float) -> float:
        """Calculate max pain strike"""
        if not strikes:
            return spot_price

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

        # Default response for insufficient data
        default_response = MeanReversionSignal(
            current_price=current_price,
            sma_20=current_price,
            sma_50=current_price,
            z_score=0,
            bollinger_position=0.5,
            signal="neutral",
            trend="sideways",
            data_quality="insufficient"
        )

        if not candles:
            self.warnings.add(
                WarningLevel.WARNING,
                "mean_reversion",
                "No candle data available for mean reversion analysis"
            )
            return default_response

        if len(candles) < 20:
            self.warnings.add(
                WarningLevel.WARNING,
                "mean_reversion",
                f"Insufficient candles for mean reversion ({len(candles)}, need 20+)"
            )
            return default_response

        # Candles from Upstox are in reverse chronological order
        # Format: [timestamp, open, high, low, close, volume, oi]
        closes = []
        for c in candles:
            if isinstance(c, list) and len(c) >= 5:
                close_price = c[4]
                if close_price is not None and close_price > 0:
                    closes.append(close_price)

        if len(closes) < 20:
            self.warnings.add(
                WarningLevel.WARNING,
                "mean_reversion",
                f"Only {len(closes)} valid closes for mean reversion"
            )
            return default_response

        # Reverse to chronological order (oldest first)
        closes = closes[::-1]

        # Use current price as the latest data point
        closes.append(current_price)

        # Calculate SMAs (using most recent data)
        sma_20 = safe_mean(closes[-20:], current_price)
        sma_50 = safe_mean(closes[-50:], sma_20) if len(closes) >= 50 else safe_mean(closes, sma_20)

        # Calculate Z-score (how many std devs from 20-day mean)
        std_20 = safe_stdev(closes[-20:], 1)  # Default to 1 to avoid division by zero
        z_score = safe_divide(current_price - sma_20, std_20, 0)

        # Bollinger position (0 = lower band, 1 = upper band)
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_range = bb_upper - bb_lower
        bollinger_position = safe_divide(current_price - bb_lower, bb_range, 0.5)
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

        data_quality = "good" if len(closes) >= 50 else "degraded"

        return MeanReversionSignal(
            current_price=round(current_price, 2),
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            z_score=round(z_score, 2),
            bollinger_position=round(bollinger_position, 2),
            signal=signal,
            trend=trend,
            data_quality=data_quality
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
        breakeven_impact = safe_divide(total_cost, lot_size, 0)

        return CostAnalysis(
            estimated_brokerage_per_lot=brokerage,
            stt_estimate=round(stt, 2),
            total_cost_per_lot=round(total_cost, 2),
            breakeven_points_impact=round(breakeven_impact, 2)
        )

    def fetch_futures_data(self, underlying: str, spot_price: float) -> FuturesData:
        """
        Fetch actual futures price and calculate basis.

        Args:
            underlying: "NIFTY" or "BANKNIFTY"
            spot_price: Current spot price

        Returns:
            FuturesData with actual or approximated futures info
        """
        now = datetime.now()

        # Try to fetch actual futures quote
        fut_response = self.client.get_futures_quote(underlying)

        if fut_response.get('status') == 'success' and fut_response.get('data'):
            # Extract futures price from response
            data = fut_response.get('data', {})

            # The response structure varies - handle both formats
            if isinstance(data, dict):
                # Try to get the first key's data
                for key, value in data.items():
                    if isinstance(value, dict) and 'last_price' in value:
                        futures_price = value.get('last_price', 0)
                        if futures_price > 0:
                            basis = futures_price - spot_price
                            basis_percent = safe_divide(basis, spot_price, 0) * 100

                            return FuturesData(
                                futures_price=round(futures_price, 2),
                                spot_price=round(spot_price, 2),
                                basis=round(basis, 2),
                                basis_percent=round(basis_percent, 4),
                                is_premium=basis > 0,
                                is_actual=True,
                                expiry_date=f"{now.strftime('%Y-%m')}-last-thursday"
                            )

        # Fallback: Approximate futures price
        self.warnings.add(
            WarningLevel.WARNING,
            "futures",
            "Could not fetch actual futures price - using approximation based on typical cost of carry"
        )

        # Approximate using typical cost of carry (risk-free rate ~6.5% p.a.)
        # For near month, assume ~20 days to expiry on average
        days_to_expiry = 20
        annual_rate = 0.065
        cost_of_carry = spot_price * annual_rate * (days_to_expiry / 365)
        futures_price = spot_price + cost_of_carry

        basis = futures_price - spot_price
        basis_percent = safe_divide(basis, spot_price, 0) * 100

        return FuturesData(
            futures_price=round(futures_price, 2),
            spot_price=round(spot_price, 2),
            basis=round(basis, 2),
            basis_percent=round(basis_percent, 4),
            is_premium=basis > 0,
            is_actual=False,
            expiry_date=f"{now.strftime('%Y-%m')}-approximated"
        )

    def _assess_liquidity(self, bid: float, ask: float, ltp: float) -> Tuple[float, str]:
        """
        Assess liquidity based on bid-ask spread.

        Returns:
            Tuple of (spread_percent, liquidity_label)
        """
        if bid <= 0 or ask <= 0:
            return 100.0, "no_market"

        spread = ask - bid
        mid_price = (bid + ask) / 2
        spread_pct = safe_divide(spread, mid_price, 100) * 100

        if spread_pct <= Config.BID_ASK_SPREAD_GOOD_THRESHOLD:
            liquidity = "good"
        elif spread_pct <= Config.BID_ASK_SPREAD_POOR_THRESHOLD:
            liquidity = "moderate"
        else:
            liquidity = "poor"

        return round(spread_pct, 2), liquidity

    def _determine_moneyness(self, strike: float, spot_price: float, option_type: str = "CE") -> str:
        """Determine if strike is ITM, ATM, or OTM"""
        diff_pct = abs(strike - spot_price) / spot_price * 100

        # Within 0.5% of spot = ATM
        if diff_pct <= 0.5:
            return "ATM"

        if option_type == "CE":
            return "ITM" if strike < spot_price else "OTM"
        else:  # PE
            return "ITM" if strike > spot_price else "OTM"

    def build_trading_zone(self, strikes: List[OptionStrike], spot_price: float,
                           atm_strike: OptionStrike, lot_size: int) -> dict:
        """
        Build a focused trading zone with ±5 strikes around ATM.

        This provides LLMs with actionable data for strategy recommendations:
        - Actual bid/ask premiums for spread construction
        - Liquidity assessment for entry/exit feasibility
        - Suggested spreads based on OI and liquidity

        Args:
            strikes: List of all parsed strikes
            spot_price: Current spot price
            atm_strike: The ATM strike object
            lot_size: Lot size for P&L calculations

        Returns:
            dict representation of TradingZone
        """
        if not strikes or not atm_strike:
            return None

        # Sort strikes by strike price
        sorted_strikes = sorted(strikes, key=lambda x: x.strike)

        # Find ATM index
        atm_idx = None
        for i, s in enumerate(sorted_strikes):
            if s.strike == atm_strike.strike:
                atm_idx = i
                break

        if atm_idx is None:
            self.warnings.add(
                WarningLevel.WARNING,
                "trading_zone",
                "Could not locate ATM strike in sorted list"
            )
            return None

        # Select strikes: 5 below + ATM + 5 above = 11 strikes
        start_idx = max(0, atm_idx - Config.TRADING_ZONE_STRIKES_BELOW)
        end_idx = min(len(sorted_strikes), atm_idx + Config.TRADING_ZONE_STRIKES_ABOVE + 1)

        zone_strikes = sorted_strikes[start_idx:end_idx]

        # Build detailed strike information
        strike_details = []
        liquidity_scores = {"good": 0, "moderate": 0, "poor": 0, "no_market": 0}

        for s in zone_strikes:
            distance = s.strike - atm_strike.strike
            distance_pct = safe_divide(distance, spot_price, 0) * 100

            # Assess liquidity for CE and PE
            ce_spread_pct, ce_liquidity = self._assess_liquidity(s.call_bid, s.call_ask, s.call_ltp)
            pe_spread_pct, pe_liquidity = self._assess_liquidity(s.put_bid, s.put_ask, s.put_ltp)

            # Track liquidity distribution
            liquidity_scores[ce_liquidity] = liquidity_scores.get(ce_liquidity, 0) + 1
            liquidity_scores[pe_liquidity] = liquidity_scores.get(pe_liquidity, 0) + 1

            # Determine moneyness (use CE perspective for the strike itself)
            if abs(distance) < 1:  # Effectively ATM
                moneyness = "ATM"
            elif s.strike < spot_price:
                moneyness = "ITM_CE / OTM_PE"
            else:
                moneyness = "OTM_CE / ITM_PE"

            detail = StrikeDetail(
                strike=s.strike,
                moneyness=moneyness,
                distance_from_atm=distance,
                distance_percent=round(distance_pct, 2),
                ce_ltp=s.call_ltp,
                ce_bid=s.call_bid,
                ce_ask=s.call_ask,
                ce_bid_ask_spread_pct=ce_spread_pct,
                ce_iv=s.call_iv,
                ce_delta=round(s.call_delta, 3),
                ce_theta=round(s.call_theta, 2),
                ce_oi=s.call_oi,
                ce_volume=s.call_volume,
                ce_liquidity=ce_liquidity,
                pe_ltp=s.put_ltp,
                pe_bid=s.put_bid,
                pe_ask=s.put_ask,
                pe_bid_ask_spread_pct=pe_spread_pct,
                pe_iv=s.put_iv,
                pe_delta=round(s.put_delta, 3),
                pe_theta=round(s.put_theta, 2),
                pe_oi=s.put_oi,
                pe_volume=s.put_volume,
                pe_liquidity=pe_liquidity
            )
            strike_details.append(asdict(detail))

        # Calculate ATM straddle prices
        atm_straddle_ltp = atm_strike.call_ltp + atm_strike.put_ltp
        atm_straddle_bid = atm_strike.call_bid + atm_strike.put_bid
        atm_straddle_ask = atm_strike.call_ask + atm_strike.put_ask

        # Expected move (straddle price represents market's expected move)
        expected_move = atm_straddle_ltp
        expected_move_pct = safe_divide(expected_move, spot_price, 0) * 100

        # Generate spread suggestions
        spread_suggestions = self._generate_spread_suggestions(
            zone_strikes, spot_price, atm_strike, lot_size
        )

        # Calculate overall liquidity score
        total_assessments = sum(liquidity_scores.values())
        good_pct = safe_divide(liquidity_scores["good"], total_assessments, 0) * 100

        if good_pct >= 70:
            overall_liquidity = "good"
        elif good_pct >= 40:
            overall_liquidity = "moderate"
        else:
            overall_liquidity = "poor"

        liquidity_summary = {
            "overall": overall_liquidity,
            "good_count": liquidity_scores["good"],
            "moderate_count": liquidity_scores["moderate"],
            "poor_count": liquidity_scores["poor"],
            "no_market_count": liquidity_scores["no_market"],
            "good_percentage": round(good_pct, 1),
            "recommendation": self._liquidity_recommendation(overall_liquidity)
        }

        trading_zone = TradingZone(
            reference_spot=round(spot_price, 2),
            atm_strike=atm_strike.strike,
            atm_straddle_price=round(atm_straddle_ltp, 2),
            atm_straddle_bid=round(atm_straddle_bid, 2),
            atm_straddle_ask=round(atm_straddle_ask, 2),
            expected_move=round(expected_move, 2),
            expected_move_percent=round(expected_move_pct, 2),
            strikes=strike_details,
            liquidity_summary=liquidity_summary,
            spread_suggestions=[asdict(s) for s in spread_suggestions]
        )

        return asdict(trading_zone)

    def _liquidity_recommendation(self, liquidity: str) -> str:
        """Generate liquidity-based trading recommendation"""
        if liquidity == "good":
            return "Liquid market - market orders acceptable for small sizes"
        elif liquidity == "moderate":
            return "Use limit orders - may need to work orders for better fills"
        else:
            return "Illiquid - use limit orders only, consider wider strikes or different expiry"

    def _generate_spread_suggestions(self, zone_strikes: List[OptionStrike],
                                     spot_price: float, atm_strike: OptionStrike,
                                     lot_size: int) -> List[SpreadSuggestion]:
        """
        Generate actionable spread suggestions based on OI, liquidity, and premiums.

        Suggests:
        1. Bull Put Spread (bullish credit spread)
        2. Bear Call Spread (bearish credit spread)
        3. Iron Condor strikes (neutral)
        """
        suggestions = []

        # Separate strikes above and below ATM
        below_atm = [s for s in zone_strikes if s.strike < atm_strike.strike]
        above_atm = [s for s in zone_strikes if s.strike > atm_strike.strike]

        # Sort for proper ordering
        below_atm = sorted(below_atm, key=lambda x: x.strike, reverse=True)  # Highest first
        above_atm = sorted(above_atm, key=lambda x: x.strike)  # Lowest first

        # 1. Bull Put Spread (Sell higher strike put, buy lower strike put)
        if len(below_atm) >= 2:
            # Find best strikes based on OI and liquidity
            sell_put_candidates = [s for s in below_atm if s.put_bid > 0]

            if len(sell_put_candidates) >= 2:
                # Sell the first OTM put with good liquidity
                sell_strike = sell_put_candidates[0]
                # Buy one strike below for protection
                buy_strike = sell_put_candidates[1]

                # Calculate spread metrics
                net_credit = sell_strike.put_bid - buy_strike.put_ask
                spread_width = sell_strike.strike - buy_strike.strike
                max_risk = spread_width - net_credit

                if net_credit > 0 and max_risk > 0:
                    risk_reward = safe_divide(net_credit, max_risk, 0)

                    # Assess combined liquidity
                    _, sell_liq = self._assess_liquidity(sell_strike.put_bid, sell_strike.put_ask, sell_strike.put_ltp)
                    _, buy_liq = self._assess_liquidity(buy_strike.put_bid, buy_strike.put_ask, buy_strike.put_ltp)
                    combined_liq = "good" if sell_liq == "good" and buy_liq == "good" else \
                        "poor" if sell_liq == "poor" or buy_liq == "poor" else "moderate"

                    suggestions.append(SpreadSuggestion(
                        strategy="bull_put_spread",
                        sell_strike=sell_strike.strike,
                        sell_premium=round(sell_strike.put_bid, 2),
                        buy_strike=buy_strike.strike,
                        buy_premium=round(buy_strike.put_ask, 2),
                        net_credit=round(net_credit, 2),
                        max_risk=round(max_risk * lot_size, 2),
                        risk_reward_ratio=round(risk_reward, 2),
                        liquidity_score=combined_liq,
                        rationale=f"Sell {sell_strike.strike} PE @ ₹{sell_strike.put_bid:.0f}, "
                                  f"Buy {buy_strike.strike} PE @ ₹{buy_strike.put_ask:.0f}. "
                                  f"Max profit ₹{net_credit * lot_size:.0f}/lot if NIFTY stays above {sell_strike.strike}"
                    ))

        # 2. Bear Call Spread (Sell lower strike call, buy higher strike call)
        if len(above_atm) >= 2:
            sell_call_candidates = [s for s in above_atm if s.call_bid > 0]

            if len(sell_call_candidates) >= 2:
                sell_strike = sell_call_candidates[0]
                buy_strike = sell_call_candidates[1]

                net_credit = sell_strike.call_bid - buy_strike.call_ask
                spread_width = buy_strike.strike - sell_strike.strike
                max_risk = spread_width - net_credit

                if net_credit > 0 and max_risk > 0:
                    risk_reward = safe_divide(net_credit, max_risk, 0)

                    _, sell_liq = self._assess_liquidity(sell_strike.call_bid, sell_strike.call_ask,
                                                         sell_strike.call_ltp)
                    _, buy_liq = self._assess_liquidity(buy_strike.call_bid, buy_strike.call_ask, buy_strike.call_ltp)
                    combined_liq = "good" if sell_liq == "good" and buy_liq == "good" else \
                        "poor" if sell_liq == "poor" or buy_liq == "poor" else "moderate"

                    suggestions.append(SpreadSuggestion(
                        strategy="bear_call_spread",
                        sell_strike=sell_strike.strike,
                        sell_premium=round(sell_strike.call_bid, 2),
                        buy_strike=buy_strike.strike,
                        buy_premium=round(buy_strike.call_ask, 2),
                        net_credit=round(net_credit, 2),
                        max_risk=round(max_risk * lot_size, 2),
                        risk_reward_ratio=round(risk_reward, 2),
                        liquidity_score=combined_liq,
                        rationale=f"Sell {sell_strike.strike} CE @ ₹{sell_strike.call_bid:.0f}, "
                                  f"Buy {buy_strike.strike} CE @ ₹{buy_strike.call_ask:.0f}. "
                                  f"Max profit ₹{net_credit * lot_size:.0f}/lot if NIFTY stays below {sell_strike.strike}"
                    ))

        # 3. Iron Condor (combine put spread + call spread)
        if len(below_atm) >= 2 and len(above_atm) >= 2:
            # Use wider strikes for iron condor (2nd OTM on each side)
            put_sell_candidates = [s for s in below_atm if s.put_bid > 0]
            call_sell_candidates = [s for s in above_atm if s.call_bid > 0]

            if len(put_sell_candidates) >= 2 and len(call_sell_candidates) >= 2:
                # Put side
                put_sell = put_sell_candidates[0]  # Sell closer to ATM
                put_buy = put_sell_candidates[-1]  # Buy furthest

                # Call side
                call_sell = call_sell_candidates[0]  # Sell closer to ATM
                call_buy = call_sell_candidates[-1]  # Buy furthest

                # Calculate total credit
                put_credit = put_sell.put_bid - put_buy.put_ask
                call_credit = call_sell.call_bid - call_buy.call_ask
                total_credit = put_credit + call_credit

                # Max risk is the wider spread width minus credit
                put_width = put_sell.strike - put_buy.strike
                call_width = call_buy.strike - call_sell.strike
                max_width = max(put_width, call_width)
                max_risk = max_width - total_credit

                if total_credit > 0 and max_risk > 0:
                    risk_reward = safe_divide(total_credit, max_risk, 0)

                    suggestions.append(SpreadSuggestion(
                        strategy="iron_condor",
                        sell_strike=put_sell.strike,  # Lower sell (put)
                        sell_premium=round(total_credit, 2),  # Combined credit
                        buy_strike=call_sell.strike,  # Upper sell (call)
                        buy_premium=0,  # Not applicable for IC summary
                        net_credit=round(total_credit, 2),
                        max_risk=round(max_risk * lot_size, 2),
                        risk_reward_ratio=round(risk_reward, 2),
                        liquidity_score="moderate",  # Conservative estimate for 4-leg
                        rationale=f"Iron Condor: Sell {put_sell.strike}/{call_sell.strike} strangle, "
                                  f"Buy {put_buy.strike}/{call_buy.strike} wings. "
                                  f"Profit zone: {put_sell.strike} to {call_sell.strike}. "
                                  f"Max profit ₹{total_credit * lot_size:.0f}/lot"
                    ))

        return suggestions


# ============================================================================
# EXPIRY SELECTOR
# ============================================================================

class ExpirySelector:
    """
    Selects appropriate expiries for analysis based on NSE rules.

    NSE Expiry Rules:
    - NIFTY: Weekly expiry every Thursday, Monthly expiry last Thursday of month
    - BANKNIFTY: Weekly expiry every Thursday, Monthly expiry last Thursday of month
    - If Thursday is a holiday, expiry moves to previous trading day (usually Wednesday)

    Note: All expiries are on Thursday (or previous day if holiday)
    """

    # Known holidays that affect expiry (add more as needed)
    # Format: 'YYYY-MM-DD'
    KNOWN_HOLIDAYS_2024_2025 = {
        '2024-01-26',  # Republic Day
        '2024-03-08',  # Maha Shivaratri
        '2024-03-25',  # Holi
        '2024-03-29',  # Good Friday
        '2024-04-11',  # Id-Ul-Fitr (tentative)
        '2024-04-14',  # Dr. Ambedkar Jayanti
        '2024-04-17',  # Ram Navami
        '2024-04-21',  # Mahavir Jayanti
        '2024-05-23',  # Buddha Purnima
        '2024-06-17',  # Eid ul-Adha
        '2024-07-17',  # Muharram
        '2024-08-15',  # Independence Day
        '2024-10-02',  # Gandhi Jayanti
        '2024-10-31',  # Diwali (Laxmi Puja)
        '2024-11-01',  # Diwali (Balipratipada)
        '2024-11-15',  # Guru Nanak Jayanti
        '2024-12-25',  # Christmas
        '2025-01-26',  # Republic Day
        '2025-02-26',  # Maha Shivaratri
        '2025-03-14',  # Holi
        '2025-03-31',  # Id-Ul-Fitr (tentative)
        '2025-04-10',  # Mahavir Jayanti
        '2025-04-14',  # Dr. Ambedkar Jayanti
        '2025-04-18',  # Good Friday
        '2025-05-12',  # Buddha Purnima
        '2025-08-15',  # Independence Day
        '2025-08-27',  # Janmashtami
        '2025-10-02',  # Gandhi Jayanti / Dussehra
        '2025-10-20',  # Diwali (tentative)
        '2025-10-21',  # Diwali Balipratipada
        '2025-11-05',  # Guru Nanak Jayanti
        '2025-12-25',  # Christmas
    }

    @staticmethod
    def get_last_thursday_of_month(year: int, month: int) -> datetime:
        """
        Get the last Thursday of a given month.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            datetime of last Thursday
        """
        # Start from last day of month
        if month == 12:
            next_month_first = datetime(year + 1, 1, 1)
        else:
            next_month_first = datetime(year, month + 1, 1)

        last_day = next_month_first - timedelta(days=1)

        # Find last Thursday (weekday 3 = Thursday)
        days_since_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_since_thursday)

        return last_thursday

    @staticmethod
    def get_all_thursdays_in_month(year: int, month: int) -> List[datetime]:
        """
        Get all Thursdays in a given month.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            List of datetime objects for all Thursdays
        """
        thursdays = []

        # Start from first day of month
        current = datetime(year, month, 1)

        # Find first Thursday
        days_until_thursday = (3 - current.weekday()) % 7
        first_thursday = current + timedelta(days=days_until_thursday)

        # Collect all Thursdays
        thursday = first_thursday
        while thursday.month == month:
            thursdays.append(thursday)
            thursday += timedelta(days=7)

        return thursdays

    @staticmethod
    def is_holiday(date: datetime) -> bool:
        """Check if a date is a known market holiday"""
        date_str = date.strftime('%Y-%m-%d')
        return date_str in ExpirySelector.KNOWN_HOLIDAYS_2024_2025

    @staticmethod
    def get_actual_expiry_date(thursday: datetime) -> datetime:
        """
        Get actual expiry date considering holidays.
        If Thursday is a holiday, expiry moves to Wednesday (or earlier).

        Args:
            thursday: The Thursday that should be expiry

        Returns:
            Actual expiry date
        """
        if ExpirySelector.is_holiday(thursday):
            # Move to Wednesday
            actual = thursday - timedelta(days=1)
            # If Wednesday is also a holiday, move to Tuesday
            if ExpirySelector.is_holiday(actual):
                actual = actual - timedelta(days=1)
            return actual
        return thursday

    @staticmethod
    def _is_monthly_expiry(exp_date: datetime) -> bool:
        """
        Check if a given date is a monthly expiry.

        Monthly expiry = Last Thursday of the month (or previous day if holiday)

        Args:
            exp_date: The expiry date to check

        Returns:
            True if this is a monthly expiry
        """
        last_thursday = ExpirySelector.get_last_thursday_of_month(exp_date.year, exp_date.month)
        actual_monthly_expiry = ExpirySelector.get_actual_expiry_date(last_thursday)

        return exp_date.date() == actual_monthly_expiry.date()

    @staticmethod
    def generate_expected_expiries(underlying: str, num_weeks: int = 8) -> dict:
        """
        Generate expected expiry dates based on NSE rules.

        This is useful for validation or when API returns incomplete data.

        Args:
            underlying: "NIFTY" or "BANKNIFTY"
            num_weeks: Number of weeks to generate

        Returns:
            dict with 'weekly' and 'monthly' expiry lists
        """
        today = datetime.now()
        weekly = []
        monthly = []

        # Generate expiries for next num_weeks weeks
        for week_offset in range(num_weeks):
            # Find next Thursday
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0 and today.hour >= 15:  # If today is Thursday after 3:30 PM
                days_until_thursday = 7

            thursday = today + timedelta(days=days_until_thursday + (week_offset * 7))
            actual_expiry = ExpirySelector.get_actual_expiry_date(thursday)

            # Skip if in the past
            if actual_expiry.date() < today.date():
                continue

            dte = (actual_expiry - today).days
            is_monthly = ExpirySelector._is_monthly_expiry(actual_expiry)

            expiry_info = {
                'date': actual_expiry.strftime('%Y-%m-%d'),
                'dte': dte,
                'is_monthly': is_monthly,
                'day_of_week': actual_expiry.strftime('%A')
            }

            if is_monthly:
                monthly.append(expiry_info)
            else:
                weekly.append(expiry_info)

        return {'weekly': weekly, 'monthly': monthly}

    @staticmethod
    def categorize_expiries(expiries_data: List[dict]) -> dict:
        """
        Categorize expiries from API data into weekly and monthly.

        Uses NSE rules to properly identify monthly expiries.

        Args:
            expiries_data: List of expiry dicts from Upstox API

        Returns:
            dict with 'weekly' and 'monthly' lists
        """
        today = datetime.now()

        weekly = []
        monthly = []

        seen_dates = set()

        for exp in expiries_data:
            exp_date_str = exp.get('expiry', '')
            if not exp_date_str:
                continue
            
            # Deduplicate
            if exp_date_str in seen_dates:
                continue
            seen_dates.add(exp_date_str)

            try:
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
            except ValueError:
                continue

            dte = (exp_date - today).days

            if dte < 0:  # Skip expired
                continue

            # Check if it's a monthly expiry using proper NSE logic
            is_monthly = ExpirySelector._is_monthly_expiry(exp_date)

            expiry_info = {
                'date': exp_date_str,
                'dte': dte,
                'is_monthly': is_monthly,
                'day_of_week': exp_date.strftime('%A')
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
    def select_expiries(categorized: dict, max_weekly: int = 4,
                        include_monthly: bool = True) -> List[dict]:
        """
        Select expiries for analysis.

        Args:
            categorized: Output from categorize_expiries()
            max_weekly: Maximum number of weekly expiries to include
            include_monthly: Whether to include monthly expiry

        Returns:
            List of selected expiry dicts
        """
        selected = []

        # Add weekly expiries (up to max_weekly)
        for exp in categorized['weekly'][:max_weekly]:
            exp['type'] = 'weekly'
            selected.append(exp)

        # Add monthly expiry if requested
        if include_monthly and categorized['monthly']:
            # Get first monthly that's at least 7 days out
            for monthly in categorized['monthly']:
                if monthly['dte'] >= 7:
                    monthly['type'] = 'monthly'
                    # Check if not already included (monthly might also appear in weekly list)
                    if monthly['date'] not in [s['date'] for s in selected]:
                        selected.append(monthly)
                    break

        return sorted(selected, key=lambda x: x['dte'])

    @staticmethod
    def get_current_week_expiry(underlying: str = "NIFTY") -> dict:
        """
        Get the current week's expiry date.

        Args:
            underlying: "NIFTY" or "BANKNIFTY"

        Returns:
            dict with expiry info
        """
        today = datetime.now()

        # Find this week's Thursday
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:  # Thursday after market close
            days_until_thursday = 7  # Move to next Thursday

        thursday = today + timedelta(days=days_until_thursday)
        actual_expiry = ExpirySelector.get_actual_expiry_date(thursday)

        dte = (actual_expiry - today).days
        is_monthly = ExpirySelector._is_monthly_expiry(actual_expiry)

        return {
            'date': actual_expiry.strftime('%Y-%m-%d'),
            'dte': dte,
            'is_monthly': is_monthly,
            'type': 'monthly' if is_monthly else 'weekly',
            'day_of_week': actual_expiry.strftime('%A')
        }

    @staticmethod
    def get_next_monthly_expiry(underlying: str = "NIFTY") -> dict:
        """
        Get the next monthly expiry date.

        Args:
            underlying: "NIFTY" or "BANKNIFTY"

        Returns:
            dict with expiry info
        """
        today = datetime.now()

        # Check current month's last Thursday
        last_thursday = ExpirySelector.get_last_thursday_of_month(today.year, today.month)
        actual_expiry = ExpirySelector.get_actual_expiry_date(last_thursday)

        # If already passed, move to next month
        if actual_expiry.date() <= today.date():
            if today.month == 12:
                last_thursday = ExpirySelector.get_last_thursday_of_month(today.year + 1, 1)
            else:
                last_thursday = ExpirySelector.get_last_thursday_of_month(today.year, today.month + 1)
            actual_expiry = ExpirySelector.get_actual_expiry_date(last_thursday)

        dte = (actual_expiry - today).days

        return {
            'date': actual_expiry.strftime('%Y-%m-%d'),
            'dte': dte,
            'is_monthly': True,
            'type': 'monthly',
            'day_of_week': actual_expiry.strftime('%A')
        }


# ============================================================================
# MAIN DATA COLLECTOR
# ============================================================================

class OptionsDataCollector:
    """Main class that orchestrates data collection and analysis"""

    def __init__(self, access_token: str):
        self.warnings = WarningCollector()
        self.client = UpstoxClient(access_token)
        self.analyzer = OptionsAnalyzer(self.client, self.warnings)

    def _fetch_spot_price(self, instrument_key: str, expiries: List[dict]) -> float:
        """
        Fetch spot price with fallback mechanisms.

        Raises:
            CriticalDataError: If spot price cannot be determined
        """
        spot_price = 0

        # Method 1: Try to get from option chain
        if expiries:
            first_chain = self.client.get_option_chain(instrument_key, expiries[0]['date'])
            if first_chain.get('status') == 'success' and first_chain.get('data'):
                spot_price = first_chain['data'][0].get('underlying_spot_price', 0)

        # Method 2: Fallback to market quote
        if spot_price <= 0:
            self.warnings.add(
                WarningLevel.INFO,
                "spot_price",
                "Trying market quote as fallback for spot price"
            )
            quote_response = self.client.get_market_quote([instrument_key])
            if quote_response.get('status') == 'success' and quote_response.get('data'):
                for key, value in quote_response.get('data', {}).items():
                    if isinstance(value, dict):
                        spot_price = value.get('last_price', 0)
                        if spot_price > 0:
                            break

        # Critical failure if still no spot price
        if spot_price <= 0:
            raise CriticalDataError(
                f"CRITICAL: Could not fetch spot price for {instrument_key}. "
                "Analysis cannot proceed with zero/invalid spot price."
            )

        return spot_price

    def _fetch_vix(self) -> Tuple[float, bool]:
        """
        Fetch VIX value with multiple fallback mechanisms.

        Returns:
            Tuple of (vix_value, is_reliable)
            - is_reliable: True if from primary source, False if estimated
        """
        vix_value = 0
        is_reliable = True

        # Method 1: Try Upstox API (primary source)
        print("      Trying Upstox API...")
        vix_data = self.client.get_market_quote([Config.INDIA_VIX])

        if vix_data.get('status') == 'success':
            vix_response_key = Config.INDIA_VIX.replace("|", ":")
            vix_info = vix_data.get('data', {}).get(vix_response_key, {})
            vix_value = vix_info.get('last_price', 0)
            if vix_value > 0:
                print(f"      ✓ VIX from Upstox: {vix_value}")
                return vix_value, True

        # Method 2: Try to get VIX from NSE website (fallback)
        print("      Upstox VIX failed, trying NSE website fallback...")
        self.warnings.add(
            WarningLevel.WARNING,
            "vix",
            "Upstox VIX fetch failed - attempting NSE website fallback"
        )

        try:
            vix_value = self._fetch_vix_from_nse()
            if vix_value > 0:
                print(f"      ✓ VIX from NSE website: {vix_value}")
                return vix_value, True
        except Exception as e:
            print(f"      NSE fallback failed: {e}")

        # Method 3: Estimate VIX based on market conditions (last resort)
        print("      All VIX sources failed, using estimation...")
        self.warnings.add(
            WarningLevel.WARNING,
            "vix",
            "Could not fetch VIX from any source - using historical estimate. "
            "Volatility regime analysis may be inaccurate!"
        )

        vix_value = self._estimate_vix_from_options()
        is_reliable = False

        if vix_value <= 0:
            # Final fallback: use historical median
            vix_value = Config.VIX_TYPICAL_MEDIAN  # 15.0
            self.warnings.add(
                WarningLevel.CRITICAL,
                "vix",
                f"VIX estimation failed - using historical median ({vix_value}). "
                "VOLATILITY ANALYSIS UNRELIABLE - Consider NO TRADE!"
            )

        print(f"      ⚠ VIX estimated: {vix_value} (UNRELIABLE)")
        return vix_value, is_reliable

    def _fetch_vix_from_nse(self) -> float:
        """
        Fetch VIX from NSE India website as fallback.

        Returns:
            VIX value or 0 if failed
        """
        import urllib.request
        import re

        try:
            # NSE India VIX page
            url = "https://www.nseindia.com/api/allIndices"

            # Create request with headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            req = urllib.request.Request(url, headers=headers)

            # Set timeout
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

                # Find India VIX in the response
                for index in data.get('data', []):
                    if 'VIX' in index.get('index', '').upper():
                        vix_value = index.get('last', 0)
                        if vix_value > 0:
                            return float(vix_value)

            return 0

        except Exception as e:
            print(f"      NSE fetch error: {e}")
            return 0

    def _estimate_vix_from_options(self) -> float:
        """
        Estimate VIX from ATM option IV if available.
        This is a rough approximation using ATM straddle IV.

        Returns:
            Estimated VIX or 0 if cannot estimate
        """
        # This will be populated if we already have option chain data
        # For now, return 0 to trigger the historical median fallback
        # In a more sophisticated implementation, we could use the ATM IV
        # from already-fetched option chains
        return 0

    def collect_full_analysis(self, underlying: str = "NIFTY",
                              expiry_date: str = None,
                              brokerage_per_order: float = 20,
                              multi_expiry: bool = True,
                              strike_range_percent: float = None) -> dict:
        """
        Collect and analyze all data for options trading decision

        Args:
            underlying: NIFTY or BANKNIFTY
            expiry_date: Specific expiry (if None, auto-selects)
            brokerage_per_order: Brokerage cost
            multi_expiry: If True, collect multiple expiries (weekly + monthly)
            strike_range_percent: Override default strike range filter

        Returns a comprehensive JSON structure for LLM analysis
        """

        # Override strike range if provided
        if strike_range_percent is not None:
            Config.STRIKE_RANGE_PERCENT = strike_range_percent

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
        print(f"Options Data Collection - {underlying} (v2.0)")
        print(f"Timestamp: {timestamp}")
        print(f"Strike Range: ±{Config.STRIKE_RANGE_PERCENT}% of spot")
        print(f"{'=' * 60}")

        try:
            # Get all expiry dates
            print("\n1. Fetching available expiries...")
            expiries_response = self.client.get_option_expiries(instrument_key)

            if expiries_response.get('status') != 'success':
                raise CriticalDataError(
                    f"Could not fetch expiry dates: {expiries_response.get('error', 'Unknown')}"
                )

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
                selected_expiries = ExpirySelector.select_expiries(
                    categorized, max_weekly=1, include_monthly=True
                )

            if not selected_expiries:
                raise CriticalDataError("No valid expiry dates found for analysis")

            print(f"   Selected {len(selected_expiries)} expiries for analysis:")
            for exp in selected_expiries:
                print(f"      - {exp['date']} ({exp['dte']} DTE, {exp.get('type', 'unknown')})")

            # Get spot price (CRITICAL - fail if unavailable)
            print("\n2. Fetching spot price...")
            spot_price = self._fetch_spot_price(instrument_key, selected_expiries)
            print(f"   Spot price: {spot_price}")

            # Get VIX (with fallbacks - warn if unreliable)
            print("\n3. Fetching India VIX...")
            vix_value, vix_is_reliable = self._fetch_vix()
            vix_status = "✓ Reliable" if vix_is_reliable else "⚠ ESTIMATED - USE CAUTION"
            print(f"   VIX: {vix_value} ({vix_status})")

            # Get Futures data (actual price if possible)
            print("\n4. Fetching futures data...")
            futures_data = self.analyzer.fetch_futures_data(underlying, spot_price)
            actual_label = "✓ Actual" if futures_data.is_actual else "⚠ Approximated"
            print(f"   Futures: {futures_data.futures_price} ({actual_label})")
            print(f"   Basis: {futures_data.basis} ({futures_data.basis_percent:.2f}%)")

            # Get historical data for HV and mean reversion
            print("\n5. Fetching historical data...")
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
                self.warnings.add(
                    WarningLevel.WARNING,
                    "historical_data",
                    f"Failed to get historical data: {historical.get('error', 'Unknown')}"
                )

            # Calculate day's change if we have candles
            day_change_percent = 0
            day_high = spot_price
            day_low = spot_price
            if candles and len(candles) > 1 and len(candles[0]) >= 5:
                prev_close = candles[1][4] if len(candles[1]) >= 5 else spot_price
                day_change_percent = safe_divide(spot_price - prev_close, prev_close, 0) * 100
                day_high = candles[0][2] if candles[0][2] else spot_price
                day_low = candles[0][3] if candles[0][3] else spot_price

            # Calculate mean reversion signals
            print("\n6. Calculating mean reversion signals...")
            mean_reversion = self.analyzer.analyze_mean_reversion(candles, spot_price)

            # Collect option chain data for each expiry
            print("\n7. Collecting option chains for each expiry...")
            expiry_analyses = []
            primary_volatility_analysis = None

            for exp in selected_expiries:
                print(f"\n   Processing {exp['date']} ({exp['dte']} DTE)...")

                chain_data = self.client.get_option_chain(instrument_key, exp['date'])

                if chain_data.get('status') != 'success':
                    self.warnings.add(
                        WarningLevel.WARNING,
                        "option_chain",
                        f"Could not fetch chain for {exp['date']}: {chain_data.get('error', 'Unknown')}"
                    )
                    continue

                # Parse with strike filtering
                strikes = self.analyzer.parse_option_chain(chain_data, spot_price, filter_strikes=True)
                print(f"      Parsed {len(strikes)} strikes (within ±{Config.STRIKE_RANGE_PERCENT}% of spot)")

                if not strikes:
                    self.warnings.add(
                        WarningLevel.WARNING,
                        "option_chain",
                        f"No valid strikes found for {exp['date']}"
                    )
                    continue

                # Count strikes with valid Greeks
                valid_greeks_count = sum(1 for s in strikes if s.has_valid_greeks)
                if valid_greeks_count < len(strikes) * 0.5:
                    self.warnings.add(
                        WarningLevel.WARNING,
                        "greeks",
                        f"Only {valid_greeks_count}/{len(strikes)} strikes have valid Greeks for {exp['date']}"
                    )

                # Find ATM
                atm_strike = self.analyzer.find_atm_strike(strikes, spot_price)

                # Analyze theta
                theta_analysis = self.analyzer.analyze_theta(atm_strike, exp['date'], spot_price)

                # Analyze OI
                oi_analysis = self.analyzer.analyze_oi(strikes, spot_price)

                # Build trading zone (±5 strikes around ATM with actionable data)
                trading_zone = self.analyzer.build_trading_zone(strikes, spot_price, atm_strike, lot_size)
                if trading_zone:
                    print(f"      Built trading zone: {len(trading_zone.get('strikes', []))} strikes, "
                          f"{len(trading_zone.get('spread_suggestions', []))} spread suggestions")

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
                    'strikes_with_valid_greeks': valid_greeks_count,
                    'atm_strike': atm_strike.strike if atm_strike else None,
                    'atm_call_iv': atm_strike.call_iv if atm_strike else None,
                    'atm_put_iv': atm_strike.put_iv if atm_strike else None,
                    'atm_straddle_price': round(atm_strike.call_ltp + atm_strike.put_ltp, 2) if atm_strike else None,
                    'trading_zone': trading_zone,  # NEW: Actionable trading data for LLMs
                    'theta_analysis': asdict(theta_analysis) if theta_analysis else None,
                    'oi_analysis': asdict(oi_analysis),
                    'strikes_data': [asdict(s) for s in strikes]
                }

                expiry_analyses.append(expiry_data)

            if not expiry_analyses:
                raise CriticalDataError("No valid expiry data could be collected")

            # Cost analysis for ATM straddle (using first expiry)
            if expiry_analyses and expiry_analyses[0].get('atm_straddle_price'):
                atm_premium = expiry_analyses[0]['atm_straddle_price']
                cost_analysis = self.analyzer.calculate_costs(atm_premium, lot_size, brokerage_per_order)
            else:
                cost_analysis = CostAnalysis(40, 0, 40, 0)
                self.warnings.add(
                    WarningLevel.INFO,
                    "costs",
                    "Using default cost estimates as ATM straddle price unavailable"
                )

            # Build market context
            day_range_percent = safe_divide(day_high - day_low, spot_price, 0) * 100

            market_context = {
                'timestamp': timestamp,
                'underlying_symbol': underlying,
                'spot_price': spot_price,
                'futures_data': asdict(futures_data),
                'vix': {
                    'value': vix_value,
                    'is_reliable': vix_is_reliable,
                    'source': 'upstox' if vix_is_reliable else 'estimated'
                },
                'day_change_percent': round(day_change_percent, 2),
                'day_high': day_high,
                'day_low': day_low,
                'day_range_percent': round(day_range_percent, 2)
            }

            # Build comprehensive output
            output = {
                "metadata": {
                    "timestamp": timestamp,
                    "data_source": "Upstox API",
                    "api_version": Config.API_VERSION,
                    "underlying": underlying,
                    "lot_size": lot_size,
                    "strike_range_filter": f"±{Config.STRIKE_RANGE_PERCENT}% of spot",
                    "analysis_framework": "Comprehensive Options Analysis v2.1",
                    "vix_reliable": vix_is_reliable
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
                "data_quality": {
                    "vix_reliable": vix_is_reliable,
                    "warnings": self.warnings.get_all(),
                    "warning_counts": self.warnings.count_by_level(),
                    "overall_quality": self._assess_overall_quality(
                        primary_volatility_analysis,
                        mean_reversion,
                        expiry_analyses
                    )
                },
                "risk_warnings": self._generate_warnings(
                    primary_volatility_analysis,
                    expiry_analyses[0] if expiry_analyses else None,
                    cost_analysis,
                    futures_data,
                    vix_is_reliable
                )
            }

            print(f"\n{'=' * 60}")
            print("Data collection complete!")
            warning_counts = self.warnings.count_by_level()
            print(
                f"Warnings: {warning_counts['critical']} critical, {warning_counts['warning']} warnings, {warning_counts['info']} info")
            print(f"{'=' * 60}")

            return output

        except CriticalDataError as e:
            print(f"\n🚨 CRITICAL ERROR: {e}")
            return {
                "status": "error",
                "error_type": "critical",
                "error_message": str(e),
                "timestamp": timestamp,
                "warnings": self.warnings.get_all()
            }

        except APIError as e:
            print(f"\n❌ API ERROR: {e}")
            return {
                "status": "error",
                "error_type": "api",
                "error_message": str(e),
                "timestamp": timestamp,
                "warnings": self.warnings.get_all()
            }

        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error_type": "unexpected",
                "error_message": str(e),
                "timestamp": timestamp,
                "warnings": self.warnings.get_all()
            }

    def _assess_overall_quality(self, volatility: Optional[VolatilityAnalysis],
                                mean_reversion: MeanReversionSignal,
                                expiries: List[dict]) -> str:
        """Assess overall data quality"""
        issues = 0

        if volatility is None or volatility.data_quality == "poor":
            issues += 2
        elif volatility.data_quality == "degraded":
            issues += 1

        if mean_reversion.data_quality == "insufficient":
            issues += 2
        elif mean_reversion.data_quality == "degraded":
            issues += 1

        if len(expiries) < 2:
            issues += 1

        if self.warnings.has_critical():
            issues += 3

        if issues == 0:
            return "good"
        elif issues <= 2:
            return "acceptable"
        elif issues <= 4:
            return "degraded"
        else:
            return "poor"

    def _generate_framework_signals(self, vol: Optional[VolatilityAnalysis],
                                    theta_dict: Optional[dict],
                                    oi_dict: Optional[dict],
                                    mr: MeanReversionSignal) -> dict:
        """Generate actionable signals based on framework"""

        signals = {
            "volatility_signal": "neutral",
            "time_decay_signal": "neutral",
            "directional_bias": "neutral",
            "overall_bias": "neutral",
            "recommended_approach": "wait",
            "confidence": "low"
        }

        if not vol:
            signals["confidence"] = "very_low"
            return signals

        confidence_score = 0

        # Volatility signal (Truth #1 & #2)
        if vol.regime in ["high_vol", "extreme"] or vol.iv_hv_ratio > 1.2:
            signals["volatility_signal"] = "sell_premium"
            confidence_score += 1
        elif vol.regime == "low_vol" and vol.iv_hv_ratio < 0.9:
            signals["volatility_signal"] = "buy_premium_cautiously"
            confidence_score += 1

        # Time decay signal (Truth #2)
        if theta_dict:
            if theta_dict.get('is_optimal_dte'):
                signals["time_decay_signal"] = "favorable_for_selling"
                confidence_score += 1
            elif theta_dict.get('days_to_expiry', 0) <= 7:
                signals["time_decay_signal"] = "high_gamma_risk"

        # Directional bias from mean reversion + OI
        oi_bias = oi_dict.get('oi_buildup_bias', 'neutral') if oi_dict else 'neutral'

        if mr.signal == "oversold" and oi_bias == "bullish":
            signals["directional_bias"] = "bullish"
            confidence_score += 1
        elif mr.signal == "overbought" and oi_bias == "bearish":
            signals["directional_bias"] = "bearish"
            confidence_score += 1
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

        # Set confidence level
        if vol.data_quality == "good" and mr.data_quality == "good" and confidence_score >= 3:
            signals["confidence"] = "high"
        elif confidence_score >= 2:
            signals["confidence"] = "medium"
        else:
            signals["confidence"] = "low"

        return signals

    def _generate_warnings(self, vol: Optional[VolatilityAnalysis],
                           primary_expiry: Optional[dict],
                           costs: CostAnalysis,
                           futures: FuturesData,
                           vix_is_reliable: bool = True) -> List[str]:
        """Generate risk warnings"""
        warnings = []

        # VIX reliability warning (highest priority)
        if not vix_is_reliable:
            warnings.append(
                "🚨 VIX DATA UNRELIABLE: Using estimated VIX - volatility regime analysis may be inaccurate. Consider NO TRADE or reduced position size!")

        if vol and vol.regime == "extreme":
            warnings.append("🚨 EXTREME VOLATILITY: VIX elevated significantly - increased tail risk")

        if vol and vol.data_quality in ["poor", "degraded"]:
            warnings.append(f"⚠️ VOLATILITY DATA QUALITY: {vol.data_quality.upper()} - signals may be unreliable")

        if primary_expiry:
            dte = primary_expiry.get('days_to_expiry', 0)
            if dte <= 7:
                warnings.append("⚠️ HIGH GAMMA RISK: <7 DTE - rapid price swings possible")

        if costs and costs.breakeven_points_impact > 5:
            warnings.append(f"💰 HIGH COST IMPACT: Costs add {costs.breakeven_points_impact} points to breakeven")

        if vol and vol.iv_skew == "put_premium":
            warnings.append("📉 PUT SKEW ELEVATED: Market pricing downside risk higher")

        if not futures.is_actual:
            warnings.append("⚠️ FUTURES APPROXIMATED: Actual futures price unavailable - basis may be inaccurate")

        if futures.is_actual and abs(futures.basis_percent) > 0.5:
            direction = "premium" if futures.is_premium else "discount"
            warnings.append(f"📊 SIGNIFICANT BASIS: Futures trading at {abs(futures.basis_percent):.2f}% {direction}")

        return warnings


# ============================================================================
# OUTPUT GENERATOR
# ============================================================================

def get_output_directory(base_dir: str = "analysis_output") -> str:
    """
    Create and return a date-based output directory.

    Structure: base_dir/YYYY-MM-DD/

    Args:
        base_dir: Base directory name (default: "analysis_output")

    Returns:
        Path to the date-based directory
    """
    import os

    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")

    # Create directory path
    output_dir = os.path.join(base_dir, today)

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def generate_json_output(data: dict, output_file: str = None, use_date_folder: bool = True) -> Tuple[str, str]:
    """
    Generate formatted JSON output with optional date-based folder organization.

    Args:
        data: Dictionary to save as JSON
        output_file: Filename (not path) for the output
        use_date_folder: If True, save to analysis_output/YYYY-MM-DD/

    Returns:
        Tuple of (json_string, full_file_path)
    """
    import os

    json_str = json.dumps(data, indent=2, default=str)

    if output_file:
        if use_date_folder:
            output_dir = get_output_directory()
            full_path = os.path.join(output_dir, output_file)
        else:
            full_path = output_file

        with open(full_path, 'w') as f:
            f.write(json_str)
        print(f"\nOutput saved to {full_path}")
        return json_str, full_path

    return json_str, None


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
                "analysis_framework": "Comprehensive Options Analysis v2.1"
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

    # Output JSON to date-based folder
    timestamp = datetime.now().strftime("%H%M")
    output_file = f"market_data_NIFTY_{timestamp}.json"
    json_str, file_path = generate_json_output(analysis, output_file, use_date_folder=True)

    print(f"\n📁 Analysis folder: {get_output_directory()}")
    print("   Save your LLM analysis files to the same folder for record keeping.")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options Trading Data Collector v2.0")
    parser.add_argument("--test", action="store_true", help="Test API connection only")
    parser.add_argument("--underlying", default="NIFTY", choices=["NIFTY", "BANKNIFTY"],
                        help="Underlying to analyze (default: NIFTY)")
    parser.add_argument("--expiry", default=None,
                        help="Specific expiry date (YYYY-MM-DD), auto-selects if not provided")
    parser.add_argument("--output", default="market_analysis.json", help="Output file name")
    parser.add_argument("--single-expiry", action="store_true", help="Only fetch single (optimal) expiry")
    parser.add_argument("--strike-range", type=float, default=None,
                        help=f"Strike range as %% of spot (default: {Config.STRIKE_RANGE_PERCENT}%%)")

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

            # Test 1: Get option expiries
            print("\n--- Test 1: Option Expiries ---")
            result = client.get_option_expiries(Config.NIFTY_INDEX)

            if result.get("status") == "success":
                print("✅ Option expiries fetch successful!")
                expiries = result.get("data", [])
                if expiries:
                    print(f"   Found {len(expiries)} expiry dates")
                    categorized = ExpirySelector.categorize_expiries(expiries)
                    print(f"   Weekly expiries: {len(categorized['weekly'])}")
                    print(f"   Monthly expiries: {len(categorized['monthly'])}")
            else:
                print(f"❌ Option expiries fetch failed: {result.get('error', result)}")

            # Test 2: Get VIX
            print("\n--- Test 2: India VIX ---")
            vix_result = client.get_market_quote([Config.INDIA_VIX])
            if vix_result.get("status") == "success":
                vix_data = vix_result.get("data", {}).get(Config.INDIA_VIX, {})
                vix_value = vix_data.get("last_price", 0)
                print(f"✅ VIX fetch successful: {vix_value}")
            else:
                print(f"❌ VIX fetch failed: {vix_result.get('error', vix_result)}")

            # Test 3: Get Futures
            print("\n--- Test 3: Futures Quote ---")
            fut_result = client.get_futures_quote("NIFTY")
            if fut_result.get("status") == "success" and fut_result.get("data"):
                print("✅ Futures quote fetch successful!")
                for key, value in fut_result.get("data", {}).items():
                    if isinstance(value, dict) and "last_price" in value:
                        print(f"   {key}: {value.get('last_price')}")
            else:
                print(f"⚠️ Futures quote fetch returned: {fut_result}")

            print("\n--- Connection Test Complete ---")

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
                multi_expiry=not args.single_expiry,
                strike_range_percent=args.strike_range
            )

            # Save to date-based folder
            if result and result.get("status") != "error":
                # Create timestamped filename
                timestamp = datetime.now().strftime("%H%M")

                # Use provided output name or generate one
                if args.output == "market_analysis.json":
                    output_file = f"market_data_{args.underlying}_{timestamp}.json"
                else:
                    output_file = args.output

                # Save to date-based folder
                output_dir = get_output_directory()
                full_path = os.path.join(output_dir, output_file)

                with open(full_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                print(f"\n✅ Analysis saved to: {full_path}")
                print(f"\n📁 Analysis folder: {output_dir}")
                print("   Save your LLM analysis files to the same folder for record keeping:")
                print(f"   - Claude analysis: {output_dir}/llm_claude_analysis.json")
                print(f"   - GPT-4 analysis:  {output_dir}/llm_gpt4_analysis.json")
                print(f"   - Judge verdict:   {output_dir}/llm_judge_verdict.json")
            else:
                print(f"\n❌ Analysis failed. Check errors above.")
                # Still save error output for debugging
                output_dir = get_output_directory()
                error_path = os.path.join(output_dir, "error_output.json")
                with open(error_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"   Error details saved to: {error_path}")
