from stable_baselines3 import PPO # On s'assure d'importer le bon modèle
import numpy as np
from src.data_collector import get_historical_data
from src.rl_trader import StockTradingEnv

def get_advice(ticker, investment_amount): # investment_amount n'est plus vraiment utilisé mais on le garde pour la cohérence
    """
    Génère un conseil d'investissement basé sur le modèle entraîné.
    """
    # Nettoyer le nom du ticker pour correspondre au nom du fichier modèle
    model_filename = ticker.replace('=', '').replace('^', '')
    model_path = f"data/trained_models/{model_filename}_model.zip"
    
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        return f"Aucun modèle entraîné trouvé pour {ticker} à '{model_path}'.\nVeuillez d'abord l'entraîner avec la commande 'train'."

    # On utilise les données les plus récentes pour la prédiction
    data = get_historical_data(ticker, start_date="2023-01-01")
    if data is None or data.empty:
        return f"Impossible de récupérer les données récentes pour {ticker}."

    # Créer un environnement 'test' pour obtenir une prédiction sur l'état actuel
    env = StockTradingEnv(data.reset_index(drop=True))
    
    # --- LA CORRECTION EST ICI ---
    # L'ancienne ligne était : obs = env.reset()
    # La fonction reset() de Gymnasium retourne maintenant un tuple (observation, info).
    # Nous devons "déballer" ce tuple pour ne récupérer que l'observation.
    obs, info = env.reset()
    # --- FIN DE LA CORRECTION ---

    # On passe maintenant uniquement l'observation (obs) au modèle, comme il s'y attend.
    action, _states = model.predict(obs, deterministic=True)

    advice = ""
    current_price = data.iloc[-1]['Close']

    if action == 2: # Acheter
        advice = (f"Conseil pour {ticker} : **ACHETER**\n"
                  f"  - Le modèle détecte un signal d'achat fort.\n"
                  f"  - Prix actuel : {current_price:.2f}")
    elif action == 0: # Vendre
        advice = (f"Conseil pour {ticker} : **VENDRE**\n"
                  f"  - Le modèle suggère de sortir de position ou de prendre des bénéfices.\n"
                  f"  - Prix actuel : {current_price:.2f}")
    else: # Conserver (Hold)
        advice = (f"Conseil pour {ticker} : **CONSERVER**\n"
                  f"  - Pas de signal clair. La patience est recommandée.\n"
                  f"  - Prix actuel : {current_price:.2f}")

    return advice