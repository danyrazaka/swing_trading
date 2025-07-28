import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import date

def get_historical_data(ticker, start_date="2020-01-01", end_date=None):
    """
    Récupère les données historiques et les enrichit avec des indicateurs techniques.
    Cette version est corrigée pour être compatible avec les versions récentes de yfinance.
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    try:
        # On utilise l'appel direct et robuste à yfinance.download
        # Le paramètre group_by='ticker' est souvent utile pour la cohérence
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            group_by='ticker'
        )
        
        if data.empty:
            print(f"Aucune donnée retournée pour {ticker}.")
            return None

        # --- GESTION ROBUSTE DES COLONNES MULTI-INDEX ---
        # Si yfinance retourne des colonnes à plusieurs niveaux, on les aplatit.
        if isinstance(data.columns, pd.MultiIndex):
            # Le niveau supérieur est souvent le ticker, on le supprime.
            data.columns = data.columns.droplevel(0)
        # --- FIN DE LA CORRECTION ---

        # On peut maintenant appliquer les indicateurs techniques en toute sécurité
        data.ta.sma(length=10, append=True)
        data.ta.sma(length=30, append=True)
        data.ta.sma(length=50, append=True)
        data.ta.rsi(length=14, append=True)
        
        # On supprime toutes les lignes contenant des valeurs non valides (NaN)
        # qui apparaissent au début du calcul des indicateurs.
        data.dropna(inplace=True)
        
        return data
        
    except Exception as e:
        # Afficher l'erreur complète pour un meilleur diagnostic
        print(f"Erreur critique lors du traitement de {ticker}: {e}")
        return None
