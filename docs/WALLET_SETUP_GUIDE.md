# Wallet Setup Guide for Polymarket Trading

## Problem: Invalid Private Key Format

Your `.env` currently has a 44-character key with hyphens, but Polymarket requires a **64-character hexadecimal Ethereum private key**.

## How to Get Your Private Key

### Option 1: MetaMask (Recommended)
1. Open MetaMask browser extension
2. Click the three dots menu → Account details
3. Click "Show private key"
4. Enter your MetaMask password
5. Copy the 64-character hex string (without the "0x" prefix)

### Option 2: From Existing Wallet Software
- **Trust Wallet**: Settings → [Wallet] → Show Secret Phrase → Export Private Key
- **Coinbase Wallet**: Settings → Security & Privacy → Show Private Key
- **Hardware Wallet**: Not recommended - keep keys on hardware for security

### Option 3: Generate New Wallet (Testing Only)
```python
from eth_account import Account
import secrets

# Generate new private key
private_key = secrets.token_hex(32)  # 64 hex chars
account = Account.from_key(private_key)

print(f"Private Key: {private_key}")
print(f"Address: {account.address}")
print("\n⚠️  SAVE THIS PRIVATE KEY SECURELY!")
print("⚠️  Send USDC to the address above to fund trading")
```

## Update Your .env File

Replace the current key in your `.env`:

```bash
# WRONG FORMAT (44 chars with hyphens):
# POLYGON_WALLET_PRIVATE_KEY=G3jwS6V5-r...

# CORRECT FORMAT (64 hex chars, no 0x prefix):
POLYGON_WALLET_PRIVATE_KEY=abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
```

## Security Best Practices

1. **Never commit .env files** - Already in .gitignore ✅
2. **Use file-based keys for production**:
   ```bash
   # Store key in a PEM file
   POLYGON_WALLET_KEY_FILE=/path/to/secure/key.pem
   ```
3. **Test with small amounts first** - Start with $5-10 USDC
4. **Use a dedicated trading wallet** - Don't use your main wallet with large holdings

## Verification Script

Run this to verify your key format:

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('POLYGON_WALLET_PRIVATE_KEY')
if key and len(key) == 64 and all(c in '0123456789abcdefABCDEF' for c in key):
    print('✅ Private key format is VALID')
    
    from eth_account import Account
    account = Account.from_key(key)
    print(f'✅ Wallet address: {account.address}')
else:
    print('❌ Private key format is INVALID')
    print(f'   Length: {len(key) if key else 0} (expected: 64)')
    print('   Expected: 64 hexadecimal characters (0-9, a-f)')
"
```

## Next Steps After Fixing

1. **Fund your wallet** - Send USDC (Polygon network) to your address
2. **Test connection**:
   ```bash
   python -c "from polymarket_agents.connectors.polymarket import Polymarket; p=Polymarket(); print(f'Balance: ${p.get_usdc_balance():.2f}')"
   ```
3. **Start trading** - You'll be ready to place bets!

## What if I don't have a Polygon wallet?

You'll need to:
1. Create a MetaMask wallet (or similar)
2. Get USDC on Polygon network
   - Bridge from Ethereum using https://wallet.polygon.technology/
   - Buy directly on an exchange that supports Polygon (e.g., Binance, Crypto.com)
   - Use a fiat on-ramp like Transak or MoonPay

## Support

- Polymarket Help: https://polymarket.com/help
- Polygon Bridge: https://wallet.polygon.technology/
- MetaMask Setup: https://metamask.io/
