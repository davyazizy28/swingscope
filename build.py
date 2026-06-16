#!/usr/bin/env python3
"""
SwingScope data builder
------------------------
Fetches free daily OHLC + recent headlines for your watchlist, scores headline
sentiment, and writes a single data.json that the SwingScope PWA reads.

No API keys required:
  - US stocks + metal miners/ETFs  -> Stooq daily CSV (robust, CI-friendly)
  - Indonesian (.JK) stocks         -> Yahoo Finance chart JSON (keyless, delayed)
  - Headlines                       -> Google News RSS (per ticker)
  - Sentiment                       -> VADER + a small finance-term overlay

Run locally:  python build.py
In CI:        the GitHub Action runs this every ~30 min and commits data.json.
"""

import json, time, math, calendar, urllib.parse
from io import StringIO
from pathlib import Path

import requests
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------------------------
# Watchlist. ("SYMBOL", "us" | "jk")
#   "us" -> fetched from Stooq (use the symbol as it trades in the US)
#   "jk" -> fetched from Yahoo with the full .JK suffix
# Add or remove names here; the app shows whatever this produces.
# ----------------------------------------------------------------------
TICKERS = [
    ("NVDA", "us"), ("AAPL", "us"), ("MSFT", "us"),
    ("NEM",  "us"), ("GOLD", "us"), ("GLD",  "us"), ("SLV", "us"),
    ("ANTM.JK", "jk"), ("MDKA.JK", "jk"), ("BBCA.JK", "jk"),
]

BARS = 260          # daily bars to keep (enough for the 200-SMA + history)
MAX_HEADLINES = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SwingScope/1.0; +https://github.com)"}

# ---- Sentiment: VADER + finance-specific lexicon overlay -------------
analyzer = SentimentIntensityAnalyzer()
FIN_LEXICON = {
    # positive
    "beats": 2.0, "beat": 2.0, "upgrade": 2.5, "upgraded": 2.5, "upgrades": 2.5,
    "raises": 2.0, "raised": 2.0, "buyback": 1.5, "surges": 2.2, "jumps": 2.0,
    "rally": 1.5, "rallies": 1.5, "tops": 1.5, "outperform": 2.0, "record": 1.2,
    "approval": 1.5, "expansion": 1.0, "soars": 2.4, "bullish": 1.8,
    # negative
    "miss": -2.0, "misses": -2.0, "downgrade": -2.5, "downgraded": -2.5,
    "cuts": -2.0, "cut": -1.5, "probe": -2.0, "lawsuit": -2.0, "fraud": -3.0,
    "plunges": -2.5, "slumps": -2.0, "warning": -1.5, "outflows": -1.5,
    "selloff": -2.0, "sell-off": -2.0, "halt": -1.5, "recall": -1.5,
    "default": -2.5, "bearish": -1.8, "slides": -1.6,
}
analyzer.lexicon.update(FIN_LEXICON)

# Bahasa Indonesia finance terms — VADER is English-only, so without this the
# IDX (.JK) headlines (fetched in Indonesian) would all score ~0. This is a
# keyword overlay: decent for headline tone, not a full Indonesian NLP model.
ID_LEXICON = {
    # positive
    "naik": 1.5, "menguat": 1.8, "melonjak": 2.2, "melesat": 2.2, "untung": 1.5,
    "laba": 1.2, "cuan": 1.8, "tumbuh": 1.2, "rekor": 1.2, "positif": 1.5,
    "dividen": 1.0, "ekspansi": 1.0, "rebound": 1.6, "optimistis": 1.4, "lompat": 1.8,
    # negative
    "melemah": -1.8, "anjlok": -2.5, "turun": -1.5, "merosot": -2.0, "jeblok": -2.0,
    "ambruk": -2.5, "tertekan": -1.6, "tekanan": -1.2, "koreksi": -1.2, "negatif": -1.5,
    "lesu": -1.4, "rugi": -2.0, "gugatan": -2.0, "denda": -1.5, "ambles": -2.2,
}
analyzer.lexicon.update(ID_LEXICON)

def score_headline(title: str) -> float:
    """Compound sentiment in [-1, 1]."""
    return round(analyzer.polarity_scores(title)["compound"], 3)

# ---- Price sources ---------------------------------------------------
def fetch_stooq(sym: str):
    """Daily OHLCV from Stooq. US symbols use the .us suffix."""
    s = sym.lower()
    if not s.endswith(".us"):
        s = s + ".us"
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    txt = r.text.strip()
    if (not txt) or txt.lower().startswith("<") or "no data" in txt.lower():
        raise ValueError(f"stooq returned no data for {sym}")
    df = pd.read_csv(StringIO(txt)).dropna()
    df = df.tail(BARS)
    bars = []
    for _, row in df.iterrows():
        ts = int(pd.Timestamp(row["Date"]).timestamp() * 1000)
        vol = row.get("Volume", 0)
        bars.append({
            "t": ts,
            "o": float(row["Open"]), "h": float(row["High"]),
            "l": float(row["Low"]),  "c": float(row["Close"]),
            "v": int(vol) if not (isinstance(vol, float) and math.isnan(vol)) else 0,
        })
    if not bars:
        raise ValueError(f"stooq parsed empty for {sym}")
    return bars

def fetch_yahoo(sym: str):
    """Daily OHLCV from Yahoo's chart JSON endpoint (keyless)."""
    q = urllib.parse.quote(sym)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{q}"
           f"?range=1y&interval=1d")
    r = requests.get(url, headers=HEADERS, timeout=25)
    if r.status_code == 429:
        raise RuntimeError(f"yahoo rate-limited (429) for {sym}")
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    q0 = res["indicators"]["quote"][0]
    bars = []
    for i, t in enumerate(ts):
        o, h, l, c = q0["open"][i], q0["high"][i], q0["low"][i], q0["close"][i]
        v = q0["volume"][i]
        if None in (o, h, l, c):
            continue
        bars.append({"t": int(t * 1000), "o": float(o), "h": float(h),
                     "l": float(l), "c": float(c), "v": int(v or 0)})
    if not bars:
        raise ValueError(f"yahoo parsed empty for {sym}")
    return bars[-BARS:]

def fetch_prices(sym: str, market: str):
    """Stooq first for US (CI-friendly); Yahoo for .JK and as the US fallback."""
    if market == "us":
        try:
            return fetch_stooq(sym)
        except Exception as e:
            print(f"    stooq failed ({e}); falling back to Yahoo")
            return fetch_yahoo(sym)
    return fetch_yahoo(sym)

# ---- News -----------------------------------------------------------
def fetch_news(sym: str, market: str):
    base = sym.replace(".JK", "")
    if market == "jk":
        query = urllib.parse.quote(f"{base} saham")
        url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
    else:
        query = urllib.parse.quote(f"{base} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url, request_headers=HEADERS)
    items = []
    for e in feed.entries[:MAX_HEADLINES]:
        title = getattr(e, "title", "").strip()
        if not title:
            continue
        pp = getattr(e, "published_parsed", None)
        ts = int(calendar.timegm(pp) * 1000) if pp else int(time.time() * 1000)  # RSS times are UTC
        items.append({"t": ts, "title": title, "s": score_headline(title)})
    items.sort(key=lambda x: x["t"], reverse=True)
    return items

# ---- Main -----------------------------------------------------------
def main():
    out = Path("data.json")
    prev = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text()).get("tickers", {})
        except Exception:
            prev = {}

    tickers = {}
    for sym, market in TICKERS:
        print(f"Fetching {sym} [{market}] ...")
        entry = {}
        # prices (keep last good data if the source fails this run)
        try:
            entry["ohlc"] = fetch_prices(sym, market)
            print(f"    {len(entry['ohlc'])} bars")
        except Exception as e:
            print(f"    price fetch FAILED: {e}")
            if prev.get(sym, {}).get("ohlc"):
                entry["ohlc"] = prev[sym]["ohlc"]
                print("    -> kept previous prices")
            else:
                print("    -> no prices available, skipping ticker")
                continue
        # headlines (non-fatal)
        try:
            entry["headlines"] = fetch_news(sym, market)
            print(f"    {len(entry['headlines'])} headlines")
        except Exception as e:
            print(f"    news fetch failed: {e}")
            entry["headlines"] = prev.get(sym, {}).get("headlines", [])
        tickers[sym] = entry
        time.sleep(1.2)   # be gentle on the free sources

    data = {"generatedAt": int(time.time() * 1000), "tickers": tickers}
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"\nWrote {out} — {len(tickers)} tickers, {out.stat().st_size//1024} KB")

if __name__ == "__main__":
    main()
