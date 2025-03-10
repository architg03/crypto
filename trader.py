import os
import time
import requests
import pandas as pd
import krakenex
from pykrakenapi import KrakenAPI
from dotenv import load_dotenv
import torch


# Hugging Face model imports
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Load API keys from .env file (or set them here)
load_dotenv()


CRYPTO_PANIC_API_KEY = 'YOUR_KEY'
KRAKEN_API_KEY = 'YOUR_KEY'
KRAKEN_PRIVATE_KEY = 'YOUR_KEY'

HUGGINGFACE_API_KEY = 'YOUR_KEY'
HUGGINGFACE_API_KEY_WIN = 'YOUR_KEY'

# Initialize Kraken API
kraken_api = krakenex.API(KRAKEN_API_KEY, KRAKEN_PRIVATE_KEY)
kraken = KrakenAPI(kraken_api)

# Hugging Face model configuration
PROMPT_TEMPLATE = "[INST]{prompt_body}[/INST]"
MAX_LENGTH = 32768  # Maximum length for the model generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")      # Change to "cuda" if using GPU and available

# Model identifiers
model_id = "agarkovv/CryptoTrader-LM"
base_model_id = "mistralai/Ministral-8B-Instruct-2410"

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = model.to(DEVICE)
model.eval()

def fetch_crypto_news():
    """Fetch news from CryptoPanic with rate-limit handling."""
    url = f'https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&public=true'
    for attempt in range(5):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = 2 ** attempt
            print(f"Rate limited. Retrying in {wait} seconds...")
            time.sleep(wait)
        else:
            print(f"Failed to fetch news. Status Code: {response.status_code}")
            return None

def fetch_market_data():
    """Fetch current market prices for BTC and ETH from Kraken."""
    pairs = ['XXBTZUSD', 'XETHZUSD']  # Kraken pairs for BTC and ETH
    try:
        ticker = kraken.get_ticker_information(pairs)
        prices = {p: float(ticker[p]['c'][0][0]) for p in pairs}
        return prices
    except Exception as e:
        print(f"Market data fetch failed: {e}")
        return None

def build_prompt(news_data, prices):
    """
    Build a prompt for the model including news headlines and current prices.
    The prompt asks the model to return trading decisions for BTC and ETH.
    """
    headlines = []
    if news_data:
        for article in news_data.get('results', []):
            title = article.get('title', '')
            published = article.get('created_at', '')
            url = article.get('url', '')
            headlines.append(f"Title: {title} (Published: {published}, URL: {url})")
    # Use a subset of headlines for brevity (e.g., the first 5)
    news_text = "\n".join(headlines[:5]) if headlines else "No recent news available."

    prompt_body = f"""
Given the following cryptocurrency news and current market prices, please provide a trading decision for Bitcoin (BTC) and Ethereum (ETH). The decision should be one of: buy, sell, or hold.

News Articles:
{news_text}

Current Prices:
BTC: {prices.get('XXBTZUSD', 'N/A')} USD
ETH: {prices.get('XETHZUSD', 'N/A')} USD

Please output your answer in the following format:
BTC: <decision>, ETH: <decision>
"""
    return PROMPT_TEMPLATE.format(prompt_body=prompt_body)

def get_trading_decision(prompt):
    """Query the CryptoTrader-LM model with the given prompt and return its output."""
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=False, max_length=MAX_LENGTH, truncation=True
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    res = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=100,  # Limit generation to a reasonable number of tokens
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    return output

def parse_decision(output):
    """
    Parse the model output to extract trading decisions.
    Expected output format: "BTC: <decision>, ETH: <decision>"
    """
    decisions = {}
    try:
        # Split the output by comma and parse each part
        parts = output.split(',')
        for part in parts:
            if ':' in part:
                coin, decision = part.split(':', 1)
                coin = coin.strip().upper()
                decision = decision.strip().lower()
                decisions[coin] = decision
    except Exception as e:
        print(f"Error parsing decision: {e}")
    return decisions

    """
    Execute a market order for the given coin using Kraken's batch order API.
    :param coin: "BTC" or "ETH"
    :param action: "buy" or "sell"
    :param volume: Trade volume (e.g., in BTC or ETH units)
    """
    # Map coin symbol to Kraken trading pair
    pair_mapping = {
        'BTC': 'XXBTZUSD',
        'ETH': 'XETHZUSD'
    }
    pair = pair_mapping.get(coin)
    if not pair:
        print(f"Unsupported coin: {coin}")
        return

    order = {
        "pair": pair,
        "type": action,         # "buy" or "sell" 
        "ordertype": "market",
        "volume": volume
    }
    try:
        response = kraken.add_standard_order_batch([order])
        print(f"Executed {action} order for {coin}: {response}")
        return response
    except Exception as e:
        print(f"Trade execution failed for {coin}: {e}")
        return None

def main():
    while True:
        print("Fetching crypto news...")
        news_data = fetch_crypto_news()
        print("Fetching market data...")
        prices = fetch_market_data()
        if news_data and prices:
            prompt = build_prompt(news_data, prices)
            print("Generated prompt for model:")
            print(prompt)
            decision_output = get_trading_decision(prompt)
            print("Model output:")
            print(decision_output)
            decisions = parse_decision(decision_output)
            print("Parsed decisions:")
            print(decisions)

            # Execute trades based on the model's decision.
            # Example volumes: 0.01 BTC and 0.1 ETH (adjust as needed)
            for coin, decision in decisions.items():
                if decision in ['buy', 'sell']:
                    volume = 0.01 if coin == 'BTC' else 0.1
                    execute_trade(coin, decision, volume)
                else:
                    print(f"No trade executed for {coin} (decision: {decision})")
        else:
            print("Missing data, skipping iteration.")
        # Wait for a minute before the next iteration (adjust as needed)
        time.sleep(60)

if __name__ == '__main__':
    main()
