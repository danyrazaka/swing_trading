import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Constantes pour la configuration de l'environnement
INITIAL_ACCOUNT_BALANCE = 10000
TRANSACTION_PENALTY = 5 # Pénalité fixe pour simuler les frais de courtage et le spread

class StockTradingEnv(gym.Env):
    """
    Environnement de trading d'actions personnalisé pour l'apprentissage par renforcement,
    compatible avec Gymnasium.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (-np.inf, np.inf)

        # Espace d'actions : 0 = Vendre, 1 = Conserver, 2 = Acheter
        self.action_space = spaces.Discrete(3)

        # Espace d'observation :
        # S'adapte dynamiquement au nombre de colonnes du DataFrame (prix + indicateurs)
        # plus 2 valeurs pour notre portefeuille (solde, actions détenues).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(df.columns) + 2,), 
            dtype=np.float32
        )

    def _next_observation(self):
        """Prépare la prochaine observation pour l'agent."""
        # L'observation inclut toutes les données de marché et l'état du portefeuille.
        obs = np.concatenate((
            self.df.loc[self.current_step].values,
            [self.balance, self.shares_held]
        ))
        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        Compatible avec la nouvelle norme Gymnasium.
        """
        super().reset(seed=seed) # Bonne pratique pour la reproductibilité

        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.current_step = 0 # On se place au début des données historiques
        
        # La fonction reset doit retourner l'observation et un dictionnaire d'infos
        return self._next_observation(), {}

    def step(self, action):
        """Exécute une étape dans l'environnement."""
        prev_net_worth = self.net_worth
        self.current_step += 1
        
        # Vérifier si on est à la fin du jeu
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1
        if done:
            # Si le jeu est terminé, on ne fait rien et on calcule la récompense finale
            reward = self.net_worth - prev_net_worth
            return self._next_observation(), reward, done, False, {} # Gymnasium retourne un 5ème élément 'truncated'

        transaction_occured = False
        # Action 0 : Vendre
        if action == 0 and self.shares_held > 0:
            sell_price = self.df.loc[self.current_step, 'Close']
            self.balance += self.shares_held * sell_price
            self.shares_held = 0
            transaction_occured = True
        # Action 2 : Acheter
        elif action == 2:
            buy_price = self.df.loc[self.current_step, 'Close']
            if self.balance > buy_price:
                # On utilise tout le solde pour acheter
                shares_to_buy = self.balance / buy_price
                self.shares_held += shares_to_buy
                self.balance = 0
                transaction_occured = True

        # Mise à jour de la valeur nette du portefeuille
        current_price = self.df.loc[self.current_step, 'Close']
        self.net_worth = self.balance + self.shares_held * current_price

        # Calcul de la récompense
        reward = self.net_worth - prev_net_worth
        if transaction_occured:
            reward -= TRANSACTION_PENALTY # Appliquer la pénalité de transaction

        # Gymnasium retourne un tuple de 5 éléments
        return self._next_observation(), reward, done, False, {}

    def render(self, mode='human', close=False):
        """Affiche l'état actuel de l'environnement."""
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Shares held: {self.shares_held:.4f}')
        print(f'Net Worth: {self.net_worth:.2f} (Profit: {profit:.2f})')

def train_model(ticker, data, timesteps=30000):
    """
    Crée, entraîne et sauvegarde le modèle d'Apprentissage par Renforcement.
    """
    # Créer l'environnement vectorisé pour Stable Baselines
    env = DummyVecEnv([lambda: StockTradingEnv(data.reset_index(drop=True))])

    # Utilisation de l'algorithme PPO, performant et stable
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./tensorboard_logs/")
    
    # Créer un nom de log propre pour TensorBoard
    tb_log_name = f"ppo_{ticker.replace('=', '').replace('^', '').replace('.DE', '_DE').replace('.PA', '_PA').replace('-USD', '_USD')}"
    
    print(f"Début de l'entraînement du modèle PPO pour {ticker}...")
    model.learn(total_timesteps=timesteps, tb_log_name=tb_log_name)

    # Sauvegarder le modèle entraîné
    model_path = f"data/trained_models/{ticker.replace('=', '').replace('^', '')}_model.zip"
    model.save(model_path)
    
    print(f"Modèle PPO entraîné et sauvegardé pour {ticker} à l'emplacement : {model_path}")
    return model