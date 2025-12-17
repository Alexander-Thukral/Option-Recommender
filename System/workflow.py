#!/usr/bin/env python3
"""
Options Trading System - Workflow Orchestrator
===============================================
This script helps orchestrate the complete 4-stage workflow.

Usage:
    python workflow.py --stage 1     # Run data collection only
    python workflow.py --stage 2     # Generate prompts for LLMs
    python workflow.py --stage 4     # Generate judge prompt
    python workflow.py --full        # Full workflow guidance
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
OUTPUT_DIR = BASE_DIR / "outputs"
PROMPTS_DIR = BASE_DIR.parent  # Prompts are in the root directory


def load_config():
    """Load configuration from settings.json"""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def stage1_data_collection():
    """Stage 1: Run data collection"""
    print("\n" + "="*60)
    print("STAGE 1: DATA COLLECTION")
    print("="*60)
    
    config = load_config()
    
    if config['api_config']['upstox_access_token'] == "YOUR_UPSTOX_ACCESS_TOKEN_HERE":
        print("\n⚠️  Upstox access token not configured!")
        print("\nTo configure:")
        print("1. Go to https://developer.upstox.com")
        print("2. Create an app and get API credentials")
        print("3. Complete OAuth flow to get access token")
        print("4. Update config/settings.json with your token")
        print("\nRunning in DEMO mode with sample data...\n")
    
    # Import and run the data collector
    print("Running data_collector.py...")
    os.system(f"python {BASE_DIR}/data_collector.py")
    
    print("\n✅ Stage 1 complete!")
    print(f"Output saved to: {OUTPUT_DIR}/market_data.json")


def stage2_generate_prompt():
    """Stage 2: Generate LLM prompt with market data"""
    print("\n" + "="*60)
    print("STAGE 2: GENERATE LLM ANALYSIS PROMPT")
    print("="*60)
    
    config = load_config()
    trader = config['trader_profile']
    
    # Load market data if available
    market_data_path = OUTPUT_DIR / "market_data.json"
    if market_data_path.exists():
        with open(market_data_path) as f:
            market_data = json.load(f)
    else:
        print("\n⚠️  No market_data.json found. Run Stage 1 first.")
        print("Using placeholder for demonstration...")
        market_data = {"_placeholder": "Run Stage 1 to generate actual data"}
    
    # Load prompt template
    prompt_template_path = PROMPTS_DIR / "stage2_llm_analysis_prompt.md"
    with open(prompt_template_path) as f:
        prompt_template = f.read()
    
    # Generate filled prompt
    filled_prompt = f"""
## TRADING ANALYSIS REQUEST

### TRADER PROFILE
- **Available Capital**: ₹{trader['capital']:,}
- **Maximum Risk Per Trade**: {trader['max_risk_per_trade_percent']}%
- **Maximum Capital Deployment**: {trader['max_capital_deployment_percent']}%
- **Risk Tolerance**: {trader['risk_tolerance']}
- **Trading Experience**: {trader['experience_level']}
- **Preferred Holding Period**: {trader['preferred_holding_period']}
- **Broker**: {trader['broker_name']} (Brokerage: ₹{trader['brokerage_per_order']} per order)

### CURRENT POSITIONS
None (or update if you have existing positions)

### TRADER'S INITIAL VIEW (for bias detection)
[Optional: Add your view here for the LLM to check for confirmation bias]

### MARKET DATA (JSON from Python script)
```json
{json.dumps(market_data, indent=2)}
```

### SPECIFIC QUESTIONS
[Add any specific aspects you want analyzed]
"""
    
    # Save the filled prompt
    output_path = OUTPUT_DIR / f"stage2_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    full_prompt = f"""
================================================================================
SYSTEM PROMPT (Copy this to the LLM first)
================================================================================

{prompt_template.split('```')[1] if '```' in prompt_template else 'See prompts/stage2_analysis_prompt.md for system prompt'}

================================================================================
USER PROMPT (Send this after the system prompt)
================================================================================

{filled_prompt}
"""
    
    with open(output_path, 'w') as f:
        f.write(full_prompt)
    
    print(f"\n✅ Prompt generated!")
    print(f"Saved to: {output_path}")
    print("\n📋 Next steps:")
    print("1. Open the generated prompt file")
    print("2. Copy the SYSTEM PROMPT to Claude/GPT-4/Perplexity")
    print("3. Send the USER PROMPT as your message")
    print("4. Save each LLM's response as llm1_response.json, llm2_response.json, etc.")
    print("5. Run Stage 4 for final synthesis")


def stage4_generate_judge_prompt():
    """Stage 4: Generate judge prompt with all LLM responses"""
    print("\n" + "="*60)
    print("STAGE 4: GENERATE JUDGE PROMPT")
    print("="*60)
    
    config = load_config()
    trader = config['trader_profile']
    
    # Check for LLM responses
    llm_responses = {}
    for i in range(1, 4):
        response_path = OUTPUT_DIR / f"llm{i}_response.json"
        if response_path.exists():
            with open(response_path) as f:
                try:
                    llm_responses[f"LLM_{i}"] = json.load(f)
                except json.JSONDecodeError:
                    # If not valid JSON, read as text
                    with open(response_path) as f2:
                        llm_responses[f"LLM_{i}"] = f2.read()
    
    if len(llm_responses) < 2:
        print("\n⚠️  Need at least 2 LLM responses for synthesis!")
        print(f"Found: {len(llm_responses)} responses")
        print("\nPlease save LLM responses as:")
        print("  - outputs/llm1_response.json")
        print("  - outputs/llm2_response.json")
        print("  - outputs/llm3_response.json (optional)")
        return
    
    # Load market data
    market_data_path = OUTPUT_DIR / "market_data.json"
    if market_data_path.exists():
        with open(market_data_path) as f:
            market_data = json.load(f)
    else:
        market_data = {"_note": "Original market data not found"}
    
    # Generate judge prompt
    judge_prompt = f"""
## FINAL ARBITRATION REQUEST

### TRADER PROFILE
- **Available Capital**: ₹{trader['capital']:,}
- **Maximum Risk Per Trade**: {trader['max_risk_per_trade_percent']}%
- **Risk Tolerance**: {trader['risk_tolerance']}
- **Trading Experience**: {trader['experience_level']}

"""
    
    for llm_name, response in llm_responses.items():
        judge_prompt += f"""
### LLM ANALYSIS: {llm_name}
```json
{json.dumps(response, indent=2) if isinstance(response, dict) else response}
```

"""
    
    judge_prompt += f"""
### ORIGINAL MARKET DATA (for reference)
```json
{json.dumps(market_data, indent=2)}
```

---

Please synthesize these analyses following the Judge LLM framework in your system prompt.
"""
    
    # Save the judge prompt
    output_path = OUTPUT_DIR / f"stage4_judge_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SYSTEM PROMPT: See stage3_judge_prompt.md\n")
        f.write("="*80 + "\n\n")
        f.write("USER PROMPT:\n")
        f.write("="*80 + "\n")
        f.write(judge_prompt)
    
    print(f"\n✅ Judge prompt generated!")
    print(f"Saved to: {output_path}")
    print("\n📋 Next steps:")
    print("1. Copy system prompt from prompts/stage4_judge_prompt.md")
    print("2. Send the generated user prompt")
    print("3. Review the final synthesized recommendation")


def full_workflow_guide():
    """Display complete workflow guide"""
    print("""
================================================================================
🎯 OPTIONS TRADING DECISION SYSTEM - COMPLETE WORKFLOW
================================================================================

This system uses a 4-stage, multi-LLM approach to generate dispassionate,
bias-countered options trading recommendations.

STAGE 1: DATA COLLECTION (Automated)
─────────────────────────────────────
• Run: python workflow.py --stage 1
• This collects market data from Upstox API
• Performs mathematical analysis (IV, HV, Greeks, OI, Mean Reversion)
• Output: outputs/market_data.json

STAGE 2: INDIVIDUAL LLM ANALYSIS (Manual)
─────────────────────────────────────────
• Run: python workflow.py --stage 2
• This generates a prompt pre-filled with your data
• Send this prompt to 2-3 LLMs:
  - Claude (claude.ai)
  - GPT-4 (chatgpt.com)
  - Perplexity (perplexity.ai)
• Save responses as:
  - outputs/llm1_response.json
  - outputs/llm2_response.json
  - outputs/llm3_response.json

STAGE 3: Response Collection (Manual)
─────────────────────────────────────
• Copy each LLM's JSON response
• Save to the outputs folder
• Ensure valid JSON format

STAGE 4: JUDGE SYNTHESIS (Semi-Automated)
─────────────────────────────────────────
• Run: python workflow.py --stage 4
• This generates a prompt for the Judge LLM
• The Judge LLM (recommend Claude) synthesizes all responses
• Provides final recommendation with:
  - Exact strategy and strikes
  - Position sizing
  - Stop-loss and profit targets
  - Scenario analysis
  - Confidence level

EXECUTION (Manual)
──────────────────
• Review the final recommendation
• If TRADE: Execute as specified
• If NO_TRADE: Wait for better setup
• Set alerts for stop-loss triggers

POST-TRADE REVIEW (Manual)
──────────────────────────
• Log trade outcome
• Compare to recommendation
• Update feedback loop

================================================================================
⚙️  CONFIGURATION
================================================================================

Before starting, configure your settings in config/settings.json:
• Upstox API token (required for live data)
• Your capital and risk parameters
• Brokerage and cost settings

================================================================================
🚦 QUICK START
================================================================================

1. Configure settings:    Edit config/settings.json
2. Collect data:          python workflow.py --stage 1
3. Generate LLM prompt:   python workflow.py --stage 2
4. Send to LLMs:          Manual - use generated prompt
5. Save responses:        Save to outputs/llm1_response.json, etc.
6. Generate judge prompt: python workflow.py --stage 4
7. Get final rec:         Send judge prompt to Claude

================================================================================
📁 FILE STRUCTURE
================================================================================

options_trading_system/
├── workflow.py              ← This script
├── data_collector.py        ← Stage 1 data collection
├── README.md                ← Full documentation
├── strategy_reference.md    ← Strategy quick reference
├── config/
│   └── settings.json        ← Your configuration
├── prompts/
│   ├── stage2_analysis_prompt.md  ← LLM analysis prompt
│   └── stage4_judge_prompt.md     ← Judge synthesis prompt
└── outputs/
    ├── market_data.json     ← Stage 1 output
    ├── llm1_response.json   ← Claude response
    ├── llm2_response.json   ← GPT-4 response
    └── llm3_response.json   ← Perplexity response

================================================================================
""")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        full_workflow_guide()
        return
    
    arg = sys.argv[1].lower()
    
    if arg == "--stage" and len(sys.argv) > 2:
        stage = sys.argv[2]
        if stage == "1":
            stage1_data_collection()
        elif stage == "2":
            stage2_generate_prompt()
        elif stage == "4":
            stage4_generate_judge_prompt()
        else:
            print(f"Unknown stage: {stage}")
            print("Valid stages: 1, 2, 4")
    elif arg == "--full":
        full_workflow_guide()
    elif arg == "--help" or arg == "-h":
        print(__doc__)
    else:
        print(f"Unknown argument: {arg}")
        print("Use --help for usage information")


if __name__ == "__main__":
    main()
