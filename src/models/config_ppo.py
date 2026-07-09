"""Configuration Pydantic pour PPO (Proximal Policy Optimization).

PPO custom en PyTorch : environnements vectorises, GAE, clipping,
normalisation des observations. Entrainement via train.py --algo ppo,
visualisation via main.py avec IA_TYPE = "ppo".

Les valeurs peuvent etre surchargees via variables d'environnement
(prefixe PPO_), ou via un fichier .env a la racine du projet.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.animals import get_animal
from src.config import ANIMAL

_ANIMAL_DEF = get_animal(ANIMAL)


class PPOSettings(BaseSettings):
    """Hyperparametres PPO centralises et valides par Pydantic."""

    model_config = SettingsConfigDict(
        env_prefix="PPO_",
        env_file=".env",
        extra="ignore",
    )

    # ----- Reproductibilite -----
    SEED: int = 42

    # ----- Reseau actor-critic -----
    # Observations : 7 features de base + 2 par muscle actionne (proprioception),
    # comme la neuroevolution. Actions : 1 activation continue par muscle.
    ACT_DIM: int = _ANIMAL_DEF.num_actuated
    HIDDEN_SIZE: int = 64
    LOG_STD_INIT: float = -0.5

    # ----- Collecte (environnements vectorises) -----
    N_ENVS: int = Field(16, ge=1)          # environnements Box2D en parallele
    N_STEPS: int = Field(256, ge=8)        # steps par env avant chaque update
    MAX_EPISODE_FRAMES: int = Field(1000, ge=1)
    STAGNATION_FRAMES: int = Field(180, ge=0)   # 0 = desactive
    STAGNATION_MIN_PROGRESS: float = Field(0.05, ge=0.0)

    # ----- Recompense -----
    # r = (progres en x) * 100 par step (meme echelle que la fitness du GA),
    # penalite en cas de chute, cout d'action optionnel.
    FALL_PENALTY: float = Field(100.0, ge=0.0)
    ACTION_COST: float = Field(0.0, ge=0.0)
    # Bonus de stabilite (dos parallele au sol) : ajoute STABILITY_COEF * cos(angle)
    # a CHAQUE step. Meme critere que le GA, mais applique par pas (PPO optimise
    # par transition). Garder le coef modeste devant le reward de progression.
    # Active par defaut pour la poule (bipede), surchargeable via PPO_USE_STABILITY_REWARD.
    USE_STABILITY_REWARD: bool = (ANIMAL == "chicken")
    STABILITY_COEF: float = Field(1.0, ge=0.0)

    # ----- Optimisation PPO -----
    TOTAL_UPDATES: int = Field(500, ge=1)  # 500 updates x (16 envs x 256 steps) = ~2M steps
    LEARNING_RATE: float = Field(3e-4, gt=0.0)
    GAMMA: float = Field(0.99, ge=0.0, le=1.0)
    GAE_LAMBDA: float = Field(0.95, ge=0.0, le=1.0)
    CLIP_RANGE: float = Field(0.2, gt=0.0)
    ENTROPY_COEF: float = Field(0.003, ge=0.0)
    VALUE_COEF: float = Field(0.5, ge=0.0)
    MAX_GRAD_NORM: float = Field(0.5, gt=0.0)
    N_EPOCHS: int = Field(10, ge=1)
    MINIBATCH_SIZE: int = Field(256, ge=8)
    NORM_OBS: bool = True

    # ----- Features d'observation -----
    TIME_PERIOD: float = Field(1.5, gt=0.0)      # phase sin/cos (aide le rythme)
    MAX_MUSCLE_SPEED: float = Field(3.0, gt=0.0)

    # ----- Materiel -----
    DEVICE: str = "auto"  # "auto" -> cuda si dispo, sinon cpu

    # ----- Sauvegardes / tracking -----
    SAVE_EVERY: int = Field(10, ge=1)  # updates entre deux checkpoints
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = f"quadruped-ppo-{ANIMAL}"
    MODEL_NAME: str = f"ppo-{ANIMAL}"
    MODELS_DIR: str = "outputs/models"
    RESULTS_DIR: str = "outputs/results"
    # Checkpoint "courant" (repris par train.py, charge par main.py en mode ppo).
    SAVE_FILE: str = f"outputs/models/{ANIMAL}_ppo.pt"


SETTINGS = PPOSettings()
SEED = SETTINGS.SEED

# ============ ACCES STYLE DICT (compat avec main.py) ============
# main.py lit TRAINING_CONFIG['save_file'] et ['max_generations'] quel que
# soit le type d'IA : on fournit les cles attendues.
TRAINING_CONFIG = {
    'save_file': SETTINGS.SAVE_FILE,
    'max_generations': 10 ** 9,   # le player d'inference ne "termine" jamais
    'auto_continue': True,
    'save_every': SETTINGS.SAVE_EVERY,
    'mlflow_tracking_uri': SETTINGS.MLFLOW_TRACKING_URI,
    'mlflow_experiment_name': SETTINGS.MLFLOW_EXPERIMENT_NAME,
    'model_name': SETTINGS.MODEL_NAME,
    'models_dir': SETTINGS.MODELS_DIR,
    'results_dir': SETTINGS.RESULTS_DIR,
}
