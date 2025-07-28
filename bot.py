import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# On importe les fonctions existantes de notre projet
from src.investment_selector import find_swing_trade_candidates, XTB_ASSETS
from src.data_collector import get_historical_data
from src.rl_trader import train_model
from src.trading_advisor import get_advice

# --- Configuration ---
# Obtenez ce token en parlant à "BotFather" sur Telegram
TELEGRAM_TOKEN = "VOTRE_TOKEN_TELEGRAM_ICI" 
CANDIDATES_FILE = "daily_candidates.txt"

# Configuration du logging pour voir les erreurs
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# --- Fonctions des Commandes du Bot ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Message d'accueil quand on lance le bot."""
    await update.message.reply_text(
        "Bonjour ! Je suis votre assistant de trading IA.\n"
        "Commandes disponibles :\n"
        "/scan - Trouve les 5 meilleures opportunités du jour.\n"
        "/train - Entraîne les modèles pour les candidats trouvés.\n"
        "/advise - Donne un conseil pour chaque candidat."
    )

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lance le scan du marché."""
    await update.message.reply_text("Reçu ! Lancement du scan de momentum... Cela peut prendre quelques minutes.")
    
    candidates = find_swing_trade_candidates(XTB_ASSETS, top_n=5)
    
    if not candidates:
        await update.message.reply_text("Aucune opportunité claire trouvée aujourd'hui.")
        open(CANDIDATES_FILE, 'w').close()
        return

    response = "--- Top 5 des Opportunités du Jour ---\n"
    response += "\n".join([f"- {asset}" for asset in candidates])
    
    with open(CANDIDATES_FILE, 'w') as f:
        for asset in candidates:
            f.write(f"{asset}\n")
            
    await update.message.reply_text(response)
    await update.message.reply_text("Prêt à entraîner les modèles avec /train")

async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entraîne les modèles pour les candidats du jour."""
    await update.message.reply_text("Reçu ! Lancement de l'entraînement pour les candidats du jour...")
    
    try:
        with open(CANDIDATES_FILE, 'r') as f:
            assets_to_train = [line.strip() for line in f.readlines() if line.strip()]
        if not assets_to_train:
            await update.message.reply_text(f"Aucun candidat à entraîner. Lancez d'abord /scan.")
            return
    except FileNotFoundError:
        await update.message.reply_text(f"Fichier des candidats non trouvé. Lancez d'abord /scan.")
        return
        
    for ticker in assets_to_train:
        await update.message.reply_text(f"Entraînement pour {ticker} en cours...")
        data = get_historical_data(ticker, start_date="2021-01-01")
        if data is not None and not data.empty:
            train_model(ticker, data)
            await update.message.reply_text(f"✅ Modèle pour {ticker} entraîné avec succès.")
        else:
            await update.message.reply_text(f"❌ Données insuffisantes pour {ticker}, entraînement sauté.")
            
    await update.message.reply_text("Entraînement terminé ! Prêt pour les conseils avec /advise")

async def advise_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Donne des conseils pour les candidats du jour."""
    await update.message.reply_text("Reçu ! Analyse des conseils de l'IA...")
    
    try:
        with open(CANDIDATES_FILE, 'r') as f:
            assets_to_advise = [line.strip() for line in f.readlines() if line.strip()]
        if not assets_to_advise:
            await update.message.reply_text(f"Aucun candidat à analyser. Lancez d'abord /scan.")
            return
    except FileNotFoundError:
        await update.message.reply_text(f"Fichier des candidats non trouvé. Lancez d'abord /scan.")
        return

    response = "--- Conseils de l'IA ---\n\n"
    for ticker in assets_to_advise:
        advice = get_advice(ticker, investment_amount=1000)
        response += advice + "\n\n"
        
    await update.message.reply_text(response)


def main():
    """Lance le bot."""
    # Création de l'application et passage du token
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Définition des commandes
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(CommandHandler("advise", advise_command))

    # Lancement du bot
    print("Le bot est en ligne et écoute les commandes...")
    application.run_polling()

if __name__ == "__main__":
    main()