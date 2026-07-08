# Entraînement sur Runpod

Guide pour lancer les entraînements headless (`train.py`) sur un pod Runpod.

## Quel pod choisir ?

**Un pod CPU multi-cœurs, PAS un GPU.** Ce projet est limité par la physique
Box2D (CPU pur) et les réseaux sont minuscules (environ 500 paramètres pour le
GA, environ 5000 pour PPO) : un GPU resterait inutilisé à 95 %.

| Algo | Ressource critique | Recommandation |
|------|--------------------|----------------|
| GA (`--algo ga`) | Nombre de cœurs (1 épisode = 1 worker) | CPU Pod 32 vCPU ou plus |
| PPO (`--algo ppo`) | Vitesse mono-cœur + un peu de multi | CPU Pod 16 vCPU suffit |

Sur un CPU Pod 32 vCPU, le GA évalue environ 100 à 200 épisodes/s : une
génération de 40 individus prend moins d'une seconde, 200 générations
prennent quelques minutes.

## Installation sur le pod

```bash
git clone https://github.com/<ton-user>/Quadruped-AI.git
cd Quadruped-AI
pip install -r requirements.txt --no-deps 2>/dev/null || pip install pygame box2d-py numpy pandas mlflow pydantic pydantic-settings tqdm
# torch version CPU (léger, suffisant, PAS la version CUDA) :
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Note : pygame est requis par les imports mais aucun affichage n'est ouvert
par train.py. Si l'import de pygame râle sur un serveur sans écran :

```bash
export SDL_VIDEODRIVER=dummy
```

## Configuration

Choisir l'animal dans `src/config.py` (`ANIMAL = "fox"` ou `"chicken"`), ou
surcharger les hyperparamètres par variables d'environnement sans toucher au
code (préfixes `NEURO_GA_` et `PPO_`) :

```bash
export NEURO_GA_POPULATION_SIZE=128      # grosse population : gratuit avec 32 coeurs
export NEURO_GA_MAX_GENERATIONS=500
export PPO_N_ENVS=32
```

## Lancer l'entraînement (arrière-plan + suivi)

```bash
mkdir -p outputs/logs
# GA :
nohup python train.py --algo ga  > outputs/logs/$(date +%F)-train-ga.log 2>&1 &
# PPO :
nohup python train.py --algo ppo > outputs/logs/$(date +%F)-train-ppo.log 2>&1 &

# Suivre la progression :
tail -f outputs/logs/*-train-ga.log
```

Les deux entraîneurs sauvegardent régulièrement (toutes les `SAVE_EVERY`
générations/updates) et à l'interruption (Ctrl+C ou kill) : on peut couper le
pod et **reprendre plus tard**, `train.py` recharge automatiquement le dernier
checkpoint (`outputs/models/{animal}_ai_neuro.pkl` pour le GA,
`outputs/models/{animal}_ppo.pt` pour PPO).

## Récupérer les résultats

Tout est dans `outputs/` (modèles, résultats, champions) plus `mlflow.db` :

```bash
tar czf results.tar.gz outputs/ mlflow.db
# puis téléchargement via l'interface Runpod, runpodctl, ou scp
```

De retour en local : extraire à la racine du projet, puis

- rejouer les champions GA : `python replay.py`
- voir la politique PPO : `IA_TYPE = "ppo"` dans `src/config.py` puis `python main.py`
- comparer les courbes : `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Estimations de coût/temps (ordre de grandeur)

- GA, 500 générations x 128 individus sur 32 vCPU : de l'ordre de 15 à 45 min.
- PPO, 2M steps (500 updates) sur 16 vCPU : de l'ordre de 1 à 2 h.

Un pod CPU 32 vCPU coûte quelques dizaines de centimes de l'heure : une
session complète d'entraînement revient à moins d'un euro.
