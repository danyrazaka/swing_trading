import pandas as pd
from src.data_collector import get_historical_data

# ==============================================================================
# VOTRE WATCHLIST PERSONNALISÉE POUR LE SWING TRADING
# C'est ici que vous contrôlez l'univers d'analyse du programme.
# Ajoutez ou supprimez des tickers de cette liste pour l'adapter à votre stratégie.
# Astuce : Cherchez sur Google "Nom de l'action Yahoo Finance" pour trouver le bon ticker.
# ==============================================================================
XTB_ASSETS = [
    # --- Actions US (NASDAQ & NYSE) ---
    "NVDA", "AMD", "TSLA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "INTC", "QCOM", "JPM", "BAC", "PFE", "JNJ", "DIS", "NKE",

    # --- Actions Européennes (Euronext, Xetra) ---
    "MC.PA", "TTE.PA", "AIR.PA", "OR.PA", "BNP.PA", "SAP.DE", "VOW3.DE", "SIE.DE", "MBG.DE", "BAYN.DE", "ASML.AS", "INGA.AS", "SHELL.AS",

    # --- Indices Mondiaux (via les ETFs les plus liquides et les bons tickers) ---
    "SPY",      # S&P 500 (USA)
    "QQQ",      # Nasdaq 100 (USA Tech)
    "IWM",      # Russell 2000 (USA Petites entreprises)
    "EWW",      # Indice du Mexique
    "EWZ",      # Indice du Brésil
    "^GDAXI",   # MODIFIÉ : Le vrai ticker pour l'indice DAX
    "^FCHI",    # MODIFIÉ : Le vrai ticker pour l'indice CAC 40

    # --- Matières Premières (via les Tickers Futures) ---
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F",

    # --- Cryptomonnaies ---
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "LINK-USD",
]

# ==============================================================================

def find_swing_trade_candidates(asset_list, top_n=5):
    """
    Scanne une liste d'actifs pour trouver ceux avec le plus fort momentum haussier.
    Un bon candidat est dans une tendance haussière, n'est pas encore sur-acheté, et montre un intérêt récent.
    """
    candidates = []
    
    # On enlève les doublons au cas où
    unique_assets = sorted(list(set(asset_list)))
    print(f"Début du scan sur votre watchlist de {len(unique_assets)} actifs...")
    
    for asset in unique_assets:
        print(f"Analyse de {asset}...")
        data = get_historical_data(asset, start_date="2024-01-01") 
        if data is None or len(data) < 60:
            continue

        # --- CRITÈRES DU SCREENER DE MOMENTUM ---
        is_uptrend = data['Close'].iloc[-1] > data['SMA_50'].iloc[-1]
        is_short_term_momentum = data['SMA_10'].iloc[-1] > data['SMA_30'].iloc[-1]
        is_not_overbought = data['RSI_14'].iloc[-1] < 70
        recent_volume = data['Volume'].iloc[-2:].mean()
        average_volume = data['Volume'].iloc[-22:-2].mean()
        has_volume_interest = recent_volume > average_volume * 1.2
        
        if is_uptrend and is_short_term_momentum and is_not_overbought and has_volume_interest:
            momentum_score = data['RSI_14'].iloc[-1]
            candidates.append({'asset': asset, 'score': momentum_score})
            print(f"-> Opportunité détectée pour {asset} (Score de Momentum: {momentum_score:.2f})")

    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    return [c['asset'] for c in sorted_candidates[:top_n]]