import httpx
import json

url = "https://gamma-api.polymarket.com/markets"
params = {"active": "true", "closed": "false", "featured": "true", "limit": 10}
res = httpx.get(url, params=params)
if res.status_code == 200:
    for m in res.json():
        question = m.get("question")
        volume = m.get("volumeNum", 0)
        print(f"Question: {question}")
        print(f"Volume: ${volume:,.2f}")
        outcomes = m.get("outcomes")
        prices = m.get("outcomePrices")
        if outcomes and prices:
            try:
                o_list = json.loads(outcomes)
                p_list = json.loads(prices)
                price_str = ", ".join([f"{o}: {p}" for o, p in zip(o_list, p_list)])
                print(f"Prices: {price_str}")
            except Exception as e:
                print(f"Prices: {prices}")
        print("-" * 20)
