# 📊 Options Strategy Quick Reference
## Based on the Comprehensive Analysis Framework

---

## Strategy Selection Matrix

Use this matrix to select the appropriate strategy based on market conditions:

| Volatility Regime | Directional Bias | Recommended Strategy | Why |
|-------------------|------------------|---------------------|-----|
| **High IV (>70th %ile)** | Bullish | Put Credit Spread | Collect elevated premium, defined risk |
| **High IV (>70th %ile)** | Bearish | Call Credit Spread | Collect elevated premium, defined risk |
| **High IV (>70th %ile)** | Neutral | Iron Condor / Short Strangle (with protection) | Maximize premium collection |
| **Normal IV (30-70th %ile)** | Bullish | Bull Call Spread (if IV low side) or Put Spread | Lower cost directional play |
| **Normal IV (30-70th %ile)** | Bearish | Bear Put Spread (if IV low side) or Call Spread | Lower cost directional play |
| **Normal IV (30-70th %ile)** | Neutral | **WAIT** or small Iron Condor | Not optimal for any strategy |
| **Low IV (<30th %ile)** | Strong Bullish | Long Call / Call Debit Spread | Cheap premium, low theta decay |
| **Low IV (<30th %ile)** | Strong Bearish | Long Put / Put Debit Spread | Cheap premium, low theta decay |
| **Low IV (<30th %ile)** | Neutral | Calendar Spread / Long Straddle | Bet on volatility expansion |

---

## Preferred Strategies (Framework-Aligned)

### 1. Put Credit Spread (Bull Put Spread)
**When to Use**: High IV + Bullish/Neutral bias + Strong support below

```
SELL: OTM Put (e.g., Delta -0.30)
BUY:  Further OTM Put (e.g., Delta -0.15)
```

**Ideal Setup**:
- IV Percentile > 70
- Strong put OI support at sold strike
- PCR > 1.0 (bullish sentiment)
- Max Pain at or above current price

**Risk Management**:
- Stop Loss: Close if spread value doubles
- Profit Target: Close at 50-65% of max profit
- Time Exit: Close 7-10 days before expiry

**Framework Alignment**: ✅ Sells premium, positive theta, defined risk

---

### 2. Call Credit Spread (Bear Call Spread)
**When to Use**: High IV + Bearish/Neutral bias + Strong resistance above

```
SELL: OTM Call (e.g., Delta 0.30)
BUY:  Further OTM Call (e.g., Delta 0.15)
```

**Ideal Setup**:
- IV Percentile > 70
- Strong call OI resistance at sold strike
- PCR < 1.0 (bearish sentiment)
- Max Pain at or below current price

**Risk Management**:
- Stop Loss: Close if spread value doubles
- Profit Target: Close at 50-65% of max profit
- Time Exit: Close 7-10 days before expiry

**Framework Alignment**: ✅ Sells premium, positive theta, defined risk

---

### 3. Iron Condor
**When to Use**: High IV + Neutral bias + Range-bound expectation

```
SELL: OTM Call (e.g., Delta 0.20)
BUY:  Further OTM Call (e.g., Delta 0.10)
SELL: OTM Put (e.g., Delta -0.20)
BUY:  Further OTM Put (e.g., Delta -0.10)
```

**Ideal Setup**:
- IV Percentile > 70 (ideally > 80)
- Low VIX (< 15) suggesting range-bound market
- No major events in holding period
- Clear support and resistance from OI analysis

**Risk Management**:
- Stop Loss: Close if either spread value doubles
- Profit Target: Close at 40-50% of max profit
- Adjustment: Roll untested side when one side threatened
- Time Exit: Close 10-14 days before expiry

**Framework Alignment**: ✅ Maximum theta capture, but watch for complexity costs

---

### 4. Debit Spread (When IV is Low)
**When to Use**: Low IV + Strong directional conviction + Event expected

```
Bull Call Spread:
BUY:  ATM or slightly ITM Call
SELL: OTM Call (to reduce cost)

Bear Put Spread:
BUY:  ATM or slightly ITM Put
SELL: OTM Put (to reduce cost)
```

**Ideal Setup**:
- IV Percentile < 30
- Strong directional signal (not just "feeling")
- Expected catalyst to move price
- 30-45 DTE minimum

**Risk Management**:
- Stop Loss: Close if loses 50% of debit paid
- Profit Target: Close at 70-100% of max profit
- Time Exit: Close if no movement by halfway point

**Framework Alignment**: ⚠️ Acceptable only in low IV environments

---

## Strategies to AVOID (Per Framework)

### ❌ Naked Short Options
- **Why**: Unlimited risk, not suitable for retail
- **Instead**: Always use spreads for defined risk

### ❌ Weekly Options Buying
- **Why**: Maximum theta decay, 80%+ expire worthless
- **Framework Truth**: "80-90% of options expire worthless"

### ❌ Pre-Event Option Buying
- **Why**: IV crush will hurt even if direction is correct
- **Framework Truth**: "IV crush after events is predictable"

### ❌ Complex Multi-Leg Strategies (Butterflies, Calendars for beginners)
- **Why**: More legs = more costs, more execution risk, harder to manage
- **Framework Truth**: "Simple strategies executed well beat complex ones"

### ❌ Frequent Trading / Scalping Options
- **Why**: Costs compound, emotional decisions increase
- **Framework Truth**: "Average holding time <30 minutes = speculation"

---

## Position Sizing Rules

### The 1-2% Rule
```
Max Risk Per Trade = Capital × 1% (conservative) to 2% (moderate)

Example:
Capital = ₹10,00,000
Max Risk = ₹10,000 to ₹20,000

If Max Loss on Spread = ₹5,000 per lot
Max Lots = ₹20,000 / ₹5,000 = 4 lots maximum
```

### The 30-40% Cash Rule
```
Always keep 30-40% of capital in cash

Example:
Capital = ₹10,00,000
Max Deployment = ₹6,00,000 to ₹7,00,000
Cash Reserve = ₹3,00,000 to ₹4,00,000
```

### Position Sizing Formula
```python
def calculate_position_size(capital, max_risk_percent, max_loss_per_lot):
    max_risk_amount = capital * (max_risk_percent / 100)
    max_lots = int(max_risk_amount / max_loss_per_lot)
    return max(1, max_lots)  # At least 1 lot
```

---

## Strike Selection Guidelines

### For Credit Spreads (Selling Premium)

| Risk Tolerance | Sold Strike Delta | Width | Probability of Profit |
|---------------|-------------------|-------|----------------------|
| Conservative | 0.15-0.20 | 50-100 points | ~80-85% |
| Moderate | 0.25-0.30 | 100-150 points | ~70-75% |
| Aggressive | 0.35-0.40 | 150-200 points | ~60-65% |

### For Debit Spreads (Buying Premium)

| Risk Tolerance | Bought Strike | Sold Strike | Cost vs Outright |
|---------------|---------------|-------------|------------------|
| Conservative | ATM | ATM + 100 | 60-70% savings |
| Moderate | Slightly ITM | ATM | 40-50% savings |
| Aggressive | ITM | Slightly OTM | 30-40% savings |

### OI-Based Strike Selection
```
Sell Puts at: Strike with highest Put OI below current price (support)
Sell Calls at: Strike with highest Call OI above current price (resistance)
```

---

## DTE (Days to Expiry) Guidelines

| DTE Range | Theta Character | Best For | Avoid |
|-----------|-----------------|----------|-------|
| **0-7 DTE** | Accelerating rapidly | Nothing (too risky) | All strategies |
| **7-21 DTE** | Fast decay | Quick credit spreads | Debit spreads |
| **21-45 DTE** | Moderate decay | Most strategies | Long-dated plays |
| **45-60 DTE** | Theta sweet spot | Credit spreads, condors | Short-term directional |
| **60+ DTE** | Slow decay | LEAPS, long-term | Short premium |

**Framework Recommendation**: 30-60 DTE for credit strategies (theta sweet spot)

---

## Quick Decision Flowchart

```
START
  │
  ▼
┌─────────────────────────────────────┐
│ Is there a major event in next 7   │
│ days? (RBI, Budget, F&O Expiry,    │
│ Global event)                       │
└─────────────────────────────────────┘
  │
  │ YES → WAIT or very conservative trade
  │
  ▼ NO
┌─────────────────────────────────────┐
│ Check IV Percentile                 │
└─────────────────────────────────────┘
  │
  ├── > 70% ────────────────┐
  │                         ▼
  │              ┌──────────────────────┐
  │              │ SELL PREMIUM         │
  │              │ Credit Spreads/Condor│
  │              └──────────────────────┘
  │
  ├── 30-70% ───────────────┐
  │                         ▼
  │              ┌──────────────────────┐
  │              │ WAIT for better setup│
  │              │ or small position    │
  │              └──────────────────────┘
  │
  └── < 30% ────────────────┐
                            ▼
                 ┌──────────────────────┐
                 │ BUY PREMIUM          │
                 │ (only with conviction)│
                 │ Debit Spreads        │
                 └──────────────────────┘
```

---

## Cost Impact Calculator

```python
def calculate_total_costs(premium, lot_size, num_lots, brokerage=20):
    """
    Calculate all trading costs for options trade
    Returns total cost and cost as percentage of premium
    """
    premium_value = premium * lot_size * num_lots
    
    # Costs
    stt_sell = premium_value * 0.000625  # 0.0625% on sell
    sebi = premium_value * 0.000001     # ₹10 per crore
    stamp = premium_value * 0.00003     # 0.003% on buy
    brokerage_total = brokerage * 2 * num_lots  # Buy + Sell
    gst = brokerage_total * 0.18        # 18% GST on brokerage
    
    total_cost = stt_sell + sebi + stamp + brokerage_total + gst
    cost_percent = (total_cost / premium_value) * 100
    
    return {
        'total_cost': round(total_cost, 2),
        'cost_per_lot': round(total_cost / num_lots, 2),
        'cost_percent': round(cost_percent, 2),
        'breakeven_impact_points': round(total_cost / (lot_size * num_lots), 2)
    }

# Example
costs = calculate_total_costs(premium=100, lot_size=25, num_lots=2)
# {'total_cost': 53.44, 'cost_per_lot': 26.72, 'cost_percent': 1.07, 'breakeven_impact_points': 1.07}
```

---

## Red Flags Checklist

Before entering ANY trade, check these:

- [ ] Is position size within 1-2% risk limit?
- [ ] Is there a major event in holding period?
- [ ] Is IV in favorable zone for the strategy?
- [ ] Are you revenge trading after a loss?
- [ ] Does this feel like a "sure thing"? (Red flag!)
- [ ] Have you factored in costs?
- [ ] Is the risk-reward ratio acceptable (>1:1)?
- [ ] Do you have a clear exit plan (stop loss + profit target)?
- [ ] Can you afford to lose the max loss amount?
- [ ] Are you okay with this trade if you don't look at it for 2 days?

**If ANY answer is NO or uncertain, reconsider the trade.**

---

## Framework Mantras

1. **"I trade volatility, not direction"**
2. **"Time is my friend or enemy - I choose which side"**
3. **"Position size matters more than being right"**
4. **"Simple strategies executed well beat complex ones"**
5. **"No trade is a valid trade"**
6. **"I can't predict price, but I can manage risk"**
7. **"Costs compound; discipline compounds faster"**

---

*Reference Guide based on the Comprehensive Analysis Framework for Options Trading*
