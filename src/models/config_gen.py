"""Configuration Pydantic pour l'IA neuroevolution.

Algorithme genetique qui evolue les poids d'un petit reseau de neurones.
Le genome = vecteur des poids et biais du MLP.
Le reseau prend l'etat du quadrupede en entree et sort 8 activations musculaires.

Usage:
    import src.models.config_gen as cfg
    print(cfg.SEED)                       # 42
    print(cfg.GA_CONFIG['population_size'])  # 40

Les valeurs peuvent etre surchargees via variables d'environnement
(prefixe NEURO_GA_), ou via un fichier .env a la racine du projet.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from src.config import DISPLAY_ENABLED


class NeuroGASettings(BaseSettings):
    """Hyperparametres centralises et valides par Pydantic."""

    model_config = SettingsConfigDict(
        env_prefix="NEURO_GA_",
        env_file=".env",
        extra="ignore",
    )

    # ----- Reproductibilite -----
    SEED: int = 42

    # ----- Architecture du reseau de neurones -----
    # INPUT_SIZE = nombre d'entrees "de base" (phase, vitesse, angle, hauteur).
    # Si USE_PROPRIOCEPTION est actif, on ajoute automatiquement 2 entrees par
    # muscle actionne (angle + vitesse du joint), soit la taille effective
    # INPUT_SIZE + 2 * OUTPUT_SIZE. Ne pas coder cette taille en dur ailleurs.
    INPUT_SIZE: int = 7
    HIDDEN_SIZE: int = 16
    OUTPUT_SIZE: int = 8
    USE_PROPRIOCEPTION: bool = True  # Injecte les angles/vitesses des joints
    MAX_MUSCLE_SPEED: float = Field(3.0, gt=0.0)  # Pour normaliser les vitesses
    ACTION_THRESHOLD: float = Field(0.33, ge=0.0, le=1.0)  # conserve (compat), inutilise en continu
    TIME_PERIOD: float = Field(1.5, gt=0.0)

    # ----- Algorithme genetique -----
    POPULATION_SIZE: int = Field(40, gt=1)
    MUTATION_RATE: float = Field(0.1, ge=0.0, le=1.0)
    MUTATION_STRENGTH: float = Field(0.2, ge=0.0)
    CROSSOVER_RATE: float = Field(0.7, ge=0.0, le=1.0)
    ELITE_SIZE: int = Field(4, ge=1)
    TOURNAMENT_SIZE: int = Field(3, ge=2)
    INIT_STD: float = Field(0.5, gt=0.0)
    FALL_PENALTY: float = Field(100.0, ge=0.0)  # Retranche du fitness si le quadrupede tombe

    # ----- Temps adaptatif -----
    ADAPTIVE_TIME: bool = True
    BASE_TIME: int = Field(500, gt=0)
    MAX_TIME: int = Field(2000, gt=0)
    REWARD_THRESHOLD_FOR_MAX_TIME: float = 5000.0

    # ----- Entrainement -----
    MAX_GENERATIONS: int = Field(200, ge=1)
    SAVE_EVERY: int = Field(5, ge=1)
    AUTO_CONTINUE: bool = True
    SPEED_MULTIPLIER: int = 50 if not DISPLAY_ENABLED else 1

    # ----- MLflow -----
    # SQLite recommande par le skill ai-training (le file store est deprecated).
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "quadruped-neuro-ga"
    MODEL_NAME: str = "neuro-ga"

    # ----- Chemins (conformes au skill thibault-ia-init) -----
    MODELS_DIR: str = "outputs/models"
    RESULTS_DIR: str = "outputs/results"
    LEGACY_SAVE_FILE: str = "outputs/models/fox_ai_neuro.pkl"  # checkpoint compat retro
    LEGACY_CSV_FILE: str = "outputs/results/training_data_neuro.csv"


# Singleton charge une seule fois a l'import (lit aussi les variables d'env).
SETTINGS = NeuroGASettings()


# ============ ACCES STYLE DICT (compat avec main.py et IABase) ============
SEED = SETTINGS.SEED

NN_CONFIG = {
    'input_size': SETTINGS.INPUT_SIZE,
    'hidden_size': SETTINGS.HIDDEN_SIZE,
    'output_size': SETTINGS.OUTPUT_SIZE,
    'use_proprioception': SETTINGS.USE_PROPRIOCEPTION,
    'max_muscle_speed': SETTINGS.MAX_MUSCLE_SPEED,
    'action_threshold': SETTINGS.ACTION_THRESHOLD,
    'time_period': SETTINGS.TIME_PERIOD,
}

GA_CONFIG = {
    'population_size': SETTINGS.POPULATION_SIZE,
    'mutation_rate': SETTINGS.MUTATION_RATE,
    'mutation_strength': SETTINGS.MUTATION_STRENGTH,
    'crossover_rate': SETTINGS.CROSSOVER_RATE,
    'elite_size': SETTINGS.ELITE_SIZE,
    'tournament_size': SETTINGS.TOURNAMENT_SIZE,
    'init_std': SETTINGS.INIT_STD,
    'fall_penalty': SETTINGS.FALL_PENALTY,
    'adaptive_time': SETTINGS.ADAPTIVE_TIME,
    'base_time': SETTINGS.BASE_TIME,
    'max_time': SETTINGS.MAX_TIME,
    'reward_threshold_for_max_time': SETTINGS.REWARD_THRESHOLD_FOR_MAX_TIME,
}

TRAINING_CONFIG = {
    'max_generations': SETTINGS.MAX_GENERATIONS,
    'save_every': SETTINGS.SAVE_EVERY,
    'auto_continue': SETTINGS.AUTO_CONTINUE,
    'speed_multiplier': SETTINGS.SPEED_MULTIPLIER,
    # Le 'save_file' reste pour compat retro avec main.py (load au demarrage),
    # mais le vrai sauvegarde se fait dans models/{name}_run-NN_date-... via la
    # convention de nommage du skill ai-training.
    'save_file': SETTINGS.LEGACY_SAVE_FILE,
    'csv_file': SETTINGS.LEGACY_CSV_FILE,
    'mlflow_tracking_uri': SETTINGS.MLFLOW_TRACKING_URI,
    'mlflow_experiment_name': SETTINGS.MLFLOW_EXPERIMENT_NAME,
    'model_name': SETTINGS.MODEL_NAME,
    'models_dir': SETTINGS.MODELS_DIR,
    'results_dir': SETTINGS.RESULTS_DIR,
}
