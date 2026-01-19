import httpx
import json

def inspect_gamma():
    print("Fetching markets from Gamma...")
    url = "https://gamma-api.polymarket.com/markets"
    params = {"limit": 1, "active": "true", "closed": "false"}
    
    try:
        res = httpx.get(url, params=params)
        if res.status_code != 200:
            print(f"Failed to fetch markets: {res.status_code}")
            return
            
        markets = res.json()
        if not markets:
            print("No markets found.")
            return
            
        market = markets[0]
        market_id = market.get("id")
        print(f"Inspecting Market ID: {market_id}")
        print("Market Keys:", list(market.keys()))
        
        # Check for history or prices in the market object
        if "history" in market:
            print("Found 'history' key in market object.")
        if "prices" in market:
            print("Found 'prices' key in market object.")
            
        # Try fetching specific history endpoint
        # Common patterns: /markets/{id}/history, /history?market_id={id}
        history_url = f"https://gamma-api.polymarket.com/markets/{market_id}/history"
        print(f"Trying history endpoint: {history_url}")
        hist_res = httpx.get(history_url)
        if hist_res.status_code == 200:
            print("Success! Found history endpoint.")
            print("History data sample:", str(hist_res.json())[:200])
        else:
            print(f"History endpoint failed: {hist_res.status_code}")
            
            # Try CLOB history URL pattern just in case
            # The clob_token_ids are needed for CLOB API usually
            clob_token_ids = json.loads(market.get("clobTokenIds", "[]"))
            if clob_token_ids:
                token_id = clob_token_ids[0]
                clob_url = f"https://clob.polymarket.com/prices-history?interval=1d&market={token_id}&fidelity=1000"
                print(f"Trying CLOB history endpoint: {clob_url}")
                clob_res = httpx.get(clob_url)
                if clob_res.status_code == 200:
                    print("Success! Found CLOB history.")
                    data = clob_res.json()
                    print("CLOB History sample:", str(data)[:200])
                else:
                    print(f"CLOB endpoint failed: {clob_res.status_code}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_gamma()
