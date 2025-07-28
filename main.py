import argparse
import os
import pandas as pd
from src.investment_selector import find_swing_trade_candidates, XTB_ASSETS
from src.data_collector import get_historical_data
from src.rl_trader import train_model
from src.trading_advisor import get_advice

# Le fichier où sont stockés les meilleurs candidats du jour
CANDIDATES_FILE = "daily_candidates.txt"

def main():
    parser = argparse.ArgumentParser(description="Assistant de Swing Trading avec IA pour XTB.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Commandes disponibles")

    # Commande SCAN : trouve les opportunités du jour
    subparsers.add_parser('scan', help="Scanne le marché pour trouver les 5 meilleures opportunités du jour.")

    # Commande TRAIN : entraîne les modèles
    parser_train = subparsers.add_parser('train', help="Entraîne les modèles IA. Par défaut, entraîne les candidats du jour.")
    parser_train.add_argument('--all', action='store_true', help="Option pour forcer l'entraînement sur TOUS les actifs de la watchlist (long).")

    # Commande ADVISE : donne des conseils
    parser_advise = subparsers.add_parser('advise', help="Donne un conseil (ACHAT/VENTE/CONSERVER) pour les candidats du jour.")
    parser_advise.add_argument('--asset', type=str, help="Option pour obtenir un conseil sur un seul actif spécifique.")

    args = parser.parse_args()

    # Création des répertoires si nécessaire
    if not os.path.exists('data/trained_models'):
        os.makedirs('data/trained_models')

    # --- LOGIQUE DES COMMANDES ---

    if args.command == 'scan':
        print("Lancement du scan de momentum pour trouver les opportunités du jour...")
        candidates = find_swing_trade_candidates(XTB_ASSETS, top_n=5)
        
        if not candidates:
            print("\nAucune opportunité claire trouvée aujourd'hui selon les critères de momentum.")
            # On vide le fichier des candidats s'il n'y en a pas
            open(CANDIDATES_FILE, 'w').close()
        else:
            print("\n--- Top 5 des Opportunités de Swing Trading pour Aujourd'hui ---")
            for asset in candidates:
                print(f"- {asset}")
            
            with open(CANDIDATES_FILE, 'w') as f:
                for asset in candidates:
                    f.write(f"{asset}\n")
            print(f"\nListe sauvegardée dans '{CANDIDATES_FILE}'. Vous pouvez maintenant entraîner ces modèles avec 'train' ou obtenir des conseils avec 'advise'.")

    elif args.command == 'train':
        assets_to_train = []
        if args.all:
            # Mode "entraînement complet"
            assets_to_train = XTB_ASSETS
            print("Lancement de l'entraînement pour TOUS les actifs de la watchlist. Cela peut être très long...")
        else:
            # Mode par défaut : on entraîne seulement les candidats du jour
            print("Lancement de l'entraînement pour les candidats du jour...")
            try:
                with open(CANDIDATES_FILE, 'r') as f:
                    assets_to_train = [line.strip() for line in f.readlines() if line.strip()]
                if not assets_to_train:
                    print(f"Le fichier '{CANDIDATES_FILE}' est vide. Lancez d'abord la commande 'scan' pour trouver les candidats.")
                    return
            except FileNotFoundError:
                print(f"Fichier '{CANDIDATES_FILE}' non trouvé. Lancez d'abord la commande 'scan'.")
                return
        
        for ticker in assets_to_train:
            print(f"\n--- Préparation de l'entraînement pour {ticker} ---")
            data = get_historical_data(ticker, start_date="2021-01-01")
            if data is not None and not data.empty:
                train_model(ticker, data)
            else:
                print(f"Données insuffisantes pour {ticker}, entraînement sauté.")
        print("\nEntraînement terminé.")

    elif args.command == 'advise':
        assets_to_advise = []
        if args.asset:
            assets_to_advise.append(args.asset)
            print(f"Génération d'un conseil pour l'actif spécifique : {args.asset}")
        else:
            print("Génération de conseils pour la liste des candidats du jour...")
            try:
                with open(CANDIDATES_FILE, 'r') as f:
                    assets_to_advise = [line.strip() for line in f.readlines() if line.strip()]
                if not assets_to_advise:
                    print(f"Le fichier des candidats est vide. Lancez d'abord 'scan'.")
                    return
            except FileNotFoundError:
                print(f"Fichier '{CANDIDATES_FILE}' non trouvé. Lancez la commande 'scan' pour générer la liste du jour.")
                return

        print("\n--- Conseils de l'IA ---")
        for ticker in assets_to_advise:
            advice = get_advice(ticker, investment_amount=1000)
            print(f"\n{advice}")

if __name__ == "__main__":
    main()
