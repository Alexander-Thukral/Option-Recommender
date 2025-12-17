import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Constants
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Try looking in parent directory if not found
        parent_path = Path(prompt_path).parent.parent / Path(prompt_path).name
        if parent_path.exists():
            with open(parent_path, 'r', encoding='utf-8') as f:
                return f.read()
        raise

def call_claude(api_key: str, system_prompt: str, user_prompt: str, model: str = "claude-3-5-sonnet-20240620") -> Dict[str, Any]:
    """Call Claude API"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": model,
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }
    
    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return {
            "status": "success",
            "content": result['content'][0]['text'],
            "raw_response": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_response": response.text if 'response' in locals() else str(e)
        }

def call_openai(api_key: str, system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Call OpenAI API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return {
            "status": "success",
            "content": result['choices'][0]['message']['content'],
            "raw_response": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "raw_response": response.text if 'response' in locals() else str(e)
        }

def run_analysis(provider: str, api_key: str, market_data: Dict[str, Any], prompt_file: str) -> Dict[str, Any]:
    """Run the full analysis flow"""
    
    # 1. Load Prompt
    try:
        full_prompt_text = load_prompt_template(prompt_file)
    except Exception as e:
        return {"status": "error", "error_message": f"Failed to load prompt: {e}"}
    
    # 2. Split System/User prompts (assuming standard format or just use whole as system/user)
    # The prompts in this project seem to be markdown files. 
    # We'll treat the file content as the "System Instructions" and append the data as the "User Message".
    # OR if the prompt file has a specific structure, we parse it.
    # Looking at stage2_llm_analysis_prompt.md, it contains "SYSTEM INSTRUCTIONS".
    
    system_prompt = full_prompt_text
    
    # Prepare User Prompt with Data
    user_prompt = f"""
    Here is the Market Data for analysis:
    ```json
    {json.dumps(market_data, indent=2)}
    ```
    
    Please provide your analysis based on the System Instructions.
    """
    
    # 3. Call API
    if provider.lower() == "claude":
        return call_claude(api_key, system_prompt, user_prompt)
    elif provider.lower() == "openai":
        return call_openai(api_key, system_prompt, user_prompt)
    else:
        return {"status": "error", "error_message": f"Unknown provider: {provider}"}
