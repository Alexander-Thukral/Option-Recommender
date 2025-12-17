# 🎯 Multi-LLM Options Trading Decision System

## A Dispassionate, Bias-Countering Framework for Indian Options Markets

---

## 📋 Executive Summary

This system implements a **4-stage, multi-LLM consensus framework** for options trading decisions in Indian markets (NSE/BSE). It is designed to:

1. **Counter emotional biases** that cause 93% of retail traders to lose money
2. **Focus on core factors** that actually matter (Volatility, Theta, Risk Management)
3. **Ignore noise** (complex indicators, hot tips, price predictions)
4. **Provide specific, actionable recommendations** with exact strikes, position sizing, and exit rules

The system is based on the **Comprehensive Analysis Framework** which identifies the 5 fundamental truths about options trading success.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OPTIONS TRADING DECISION SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: DATA COLLECTION (Python + Upstox API)                          │   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │   │
│  │  │ Option Chain │  │  Historical  │  │  India VIX   │  │   Market    │  │   │
│  │  │  + Greeks    │  │   Candles    │  │    Data      │  │   Quotes    │  │   │
│  │  │  + OI + IV   │  │  (60 days)   │  │              │  │             │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │   │
│  │                              │                                           │   │
│  │                              ▼                                           │   │
│  │              ┌──────────────────────────────┐                           │   │
│  │              │   Mathematical Analysis       │                           │   │
│  │              │   • IV vs HV calculation      │                           │   │
│  │              │   • Mean Reversion signals    │                           │   │
│  │              │   • OI Analysis & Max Pain    │                           │   │
│  │              │   • Cost calculations         │                           │   │
│  │              └──────────────────────────────┘                           │   │
│  │                              │                                           │   │
│  │                              ▼                                           │   │
│  │              ┌──────────────────────────────┐                           │   │
│  │              │     OUTPUT: market_data.json  │                           │   │
│  │              └──────────────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2 & 3: PARALLEL LLM ANALYSIS                                      │   │
│  │                                                                          │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │   │
│  │   │   CLAUDE    │    │   GPT-4     │    │ PERPLEXITY  │                 │   │
│  │   │             │    │             │    │             │                 │   │
│  │   │ + Web Search│    │ + Web Search│    │ + Web Search│                 │   │
│  │   │ + Framework │    │ + Framework │    │ + Framework │                 │   │
│  │   │ + Bias Check│    │ + Bias Check│    │ + Bias Check│                 │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘                 │   │
│  │          │                  │                  │                         │   │
│  │          ▼                  ▼                  ▼                         │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │   │
│  │   │ llm1.json   │    │ llm2.json   │    │ llm3.json   │                 │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: JUDGE LLM (Final Arbitration)                                  │   │
│  │                                                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  │                        JUDGE LLM                                  │   │   │
│  │  │                                                                   │   │   │
│  │  │  • Score each LLM analysis (8 criteria)                          │   │   │
│  │  │  • Identify consensus and conflicts                              │   │   │
│  │  │  • Resolve conflicts (conservative bias)                         │   │   │
│  │  │  • Apply Devil's Advocate thinking                               │   │   │
│  │  │  • Generate final recommendation                                 │   │   │
│  │  └──────────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                           │   │
│  │                              ▼                                           │   │
│  │              ┌──────────────────────────────┐                           │   │
│  │              │  FINAL RECOMMENDATION        │                           │   │
│  │              │  • Exact strategy & strikes  │                           │   │
│  │              │  • Position sizing           │                           │   │
│  │              │  • Stop-loss & profit target │                           │   │
│  │              │  • Scenario analysis         │                           │   │
│  │              │  • Confidence level          │                           │   │
│  │              └──────────────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 File Structure

```
options_trading_system/
├── README.md                           # This file
├── data_collector.py                   # Stage 1: Python data collection
├── prompts/
│   ├── stage2_analysis_prompt.md       # Stage 2: Individual LLM prompts
│   └── stage4_judge_prompt.md          # Stage 4: Judge LLM prompt
├── outputs/                            # Generated outputs
│   ├── market_data.json                # Stage 1 output
│   ├── llm1_response.json              # Stage 2 output (Claude)
│   ├── llm2_response.json              # Stage 2 output (GPT-4)
│   ├── llm3_response.json              # Stage 2 output (Perplexity)
│   └── final_recommendation.json       # Stage 4 output
└── config/
    └── settings.json                   # Configuration settings
```

---

## 🚀 Quick Start Guide

### Prerequisites

1. **Upstox Account** with API access
2. **LLM Access**: At least 2 of:
   - Claude (Anthropic) - via claude.ai or API
   - GPT-4 (OpenAI) - via ChatGPT Plus or API
   - Perplexity Pro - via perplexity.ai
3. **Python 3.8+** with `requests` library

### Step-by-Step Usage

#### Step 1: Configure Upstox API

```python
# In data_collector.py, set your access token
ACCESS_TOKEN = "your_upstox_access_token_here"
```

To get your access token:
1. Go to https://developer.upstox.com
2. Create an app and get your API credentials
3. Complete the OAuth flow to get an access token

#### Step 2: Run Data Collection

```bash
cd options_trading_system
python data_collector.py
```

This generates `market_data.json` with all the analyzed data.

#### Step 3: Copy Prompt to Multiple LLMs

1. Open `prompts/stage2_analysis_prompt.md`
2. Copy the **System Prompt** to each LLM
3. Fill in the **User Prompt Template** with:
   - Your capital and risk parameters
   - The `market_data.json` content
4. Send to 2-3 different LLMs

#### Step 4: Collect LLM Responses

Save each LLM's response as:
- `outputs/llm1_response.json` (Claude)
- `outputs/llm2_response.json` (GPT-4)
- `outputs/llm3_response.json` (Perplexity)

#### Step 5: Run Judge Analysis

1. Open `prompts/stage4_judge_prompt.md`
2. Copy the prompt to your preferred "Judge" LLM (Claude recommended)
3. Include all LLM responses in the prompt
4. Get the final synthesized recommendation

---

## 📊 Data Available from Upstox API

### ✅ Available Data

| Data Point | Endpoint | Notes |
|------------|----------|-------|
| Option Chain | `/v2/option/chain` | Full chain with all strikes |
| Greeks (Δ, Θ, Γ, ν, ρ) | Included in chain | Real-time calculations |
| Implied Volatility | Included in chain | Per strike |
| Open Interest + Change | Included | prev_oi available |
| Volume | Included | Real-time |
| LTP, Bid/Ask | Included | With quantities |
| PCR | Included | Per strike |
| Probability of Profit | `pop` field | Recently added |
| Historical Candles | `/v2/historical-candle` | Up to 10 years daily |
| Intraday Data | `/v2/historical-candle/intraday` | 1-min, 30-min |
| India VIX | Via market quotes | `NSE_INDEX\|India VIX` |

### ⚠️ Not Available (LLM Web Search Required)

| Data Point | Alternative Source |
|------------|-------------------|
| GIFT Nifty | LLM web search (NSE IX, financial sites) |
| Global Markets | LLM web search (US futures, European indices) |
| FII/DII Activity | LLM web search (NSE, financial news) |
| News & Events | LLM web search |
| Sentiment | LLM analysis of news |

---

## 🎯 Core Framework Principles

### The 5 Truths (from Comprehensive Analysis)

1. **Math Is Against Option Buyers**
   - 80-90% expire worthless
   - Theta decay is CERTAIN
   - Sellers have 2/3 probability of profit

2. **Volatility Is the True Underlying**
   - IV percentile > 70 → Sell premium
   - IV percentile < 30 → Buying less risky
   - Never buy before events (IV crush)

3. **Risk Management Is THE Differentiator**
   - Max 1-2% capital per trade
   - Keep 20-40% in cash
   - Position sizing > Strategy selection

4. **Retail Traders Are the Product**
   - Competing against algos and institutions
   - Edge is patience, discipline, simplicity
   - Not information or speed

5. **Emotional Discipline Is THE Skill**
   - System counters biases
   - NO_TRADE is valid recommendation
   - Rules beat emotions

### What Doesn't Matter

- Complex technical indicators (RSI, MACD, etc.)
- Hot tips and stock picks
- Precise price targets
- Frequent trading
- Exotic multi-leg strategies

---

## 💰 Cost Considerations (India-Specific)

| Cost Component | Rate | Notes |
|----------------|------|-------|
| STT (Sell) | 0.0625% | On premium |
| STT (Exercise) | 0.125% | On FULL CONTRACT VALUE! |
| SEBI Charges | ₹10/crore | Negligible |
| Stamp Duty | 0.003% | On buy side |
| GST | 18% | On brokerage |
| Brokerage | ₹20-40 | Per order (flat fee) |
| Slippage | 0.5-1% | Estimate |

**⚠️ CRITICAL WARNING**: If ITM options expire/exercise without squaring off, STT is charged on FULL contract value, not premium. This can devastate profits!

---

## 🔄 Workflow Summary

```
1. MORNING (Pre-Market)
   └── Run data_collector.py
   └── Review market_data.json

2. ANALYSIS (9:00-9:30 AM)
   └── Submit to 2-3 LLMs with your parameters
   └── Wait for responses (5-10 min each)

3. SYNTHESIS (9:30-10:00 AM)
   └── Submit all responses to Judge LLM
   └── Review final recommendation

4. EXECUTION (If trade recommended)
   └── Execute as per recommendation
   └── Set alerts for stop-loss triggers

5. MONITORING (Throughout day)
   └── Check position vs monitoring checklist
   └── Adjust if triggers hit

6. EOD REVIEW
   └── Log trade outcome
   └── Update feedback loop
```

---

## 📈 Example Output Structure

### Stage 1 Output (market_data.json)

```json
{
  "metadata": {
    "timestamp": "2024-12-15 09:15:00",
    "underlying": "NIFTY",
    "expiry_date": "2024-12-26"
  },
  "core_analysis": {
    "volatility": {
      "current_iv_atm": 13.5,
      "historical_volatility_20d": 12.8,
      "iv_hv_ratio": 1.05,
      "regime": "normal"
    },
    "theta": {
      "days_to_expiry": 11,
      "theta_regime": "moderate",
      "is_optimal_dte": false
    },
    "mean_reversion": {
      "z_score": 0.45,
      "signal": "neutral",
      "trend": "uptrend"
    }
  },
  "framework_signals": {
    "volatility_signal": "neutral",
    "recommended_approach": "wait_for_better_setup"
  }
}
```

### Final Recommendation Example

```json
{
  "final_recommendation": {
    "action": "TRADE",
    "confidence_level": 7,
    "strategy_name": "put_credit_spread",
    "exact_trade": {
      "leg_1": {
        "action": "SELL",
        "option_type": "PE",
        "strike": 24000,
        "lots": 2,
        "expected_premium": 85
      },
      "leg_2": {
        "action": "BUY",
        "option_type": "PE",
        "strike": 23900,
        "lots": 2,
        "expected_premium": 55
      }
    },
    "trade_metrics": {
      "net_credit": 30,
      "max_profit": 1500,
      "max_loss": 3500,
      "probability_of_profit": 68
    },
    "risk_management": {
      "stop_loss": "Close if spread widens to 60 (2x entry)",
      "profit_target": "Close at 50% profit (spread at 15)"
    }
  }
}
```

---

## ⚠️ Disclaimers

1. **No Financial Advice**: This system provides analysis, not financial advice
2. **No Guarantees**: Past performance doesn't guarantee future results
3. **Risk of Loss**: Options trading involves significant risk of loss
4. **Verify Data**: Always verify critical data points manually
5. **Test First**: Paper trade before using with real money
6. **Your Responsibility**: Final trading decisions are yours alone

---

## 🔧 Customization

### Adjusting Risk Parameters

In `data_collector.py`, modify the `Config` class:

```python
class Config:
    IV_HIGH_PERCENTILE = 70  # Adjust for your risk tolerance
    IV_LOW_PERCENTILE = 30
    OPTIMAL_DTE_MIN = 30     # Change based on strategy preference
    OPTIMAL_DTE_MAX = 60
```

### Adding New Underlyings

Add instrument keys to the Config:

```python
FINNIFTY_INDEX = "NSE_INDEX|Nifty Fin Service"
MIDCPNIFTY_INDEX = "NSE_INDEX|NIFTY MID SELECT"
```

### Adjusting Cost Parameters

Update for your broker:

```python
STT_RATE_OPTIONS_SELL = 0.000625
# Brokerage varies by broker - update accordingly
```

---

## 📞 Support & Feedback

For improvements to this system:
1. Review trade outcomes and identify patterns
2. Update prompts based on what's working
3. Add new data sources as they become available
4. Refine bias correction based on your tendencies

Remember: **The goal is not to win every trade, but to survive long enough for probability to work in your favor.**

---

*Built on the Comprehensive Analysis Framework for Options Trading in India*
