# config_ia.py
"""Configuration pour l'entraînement des IA"""

# ============ OPTIONS PRINCIPALES ============

# ========== SÉLECTION DE L'ANIMAL ==========
# Chaque animal est défini dans src/animals/ (squelette + peau procédurale).
# Pour en ajouter un : créer le module puis l'enregistrer dans
# src/animals/__init__.py (get_animal).
ANIMAL = "chicken"  # "fox" (renard, quadrupède) ou "chicken" (poule, bipède)

# Mode de contrôle
HUMAN_CONTROL = False  # True = contrôle humain, False = contrôle IA

# Affichage
DISPLAY_ENABLED = True  # True = afficher l'écran, False = mode rapide sans affichage


CONFIG = {
    'speed_multiplier': 50 if not DISPLAY_ENABLED else 1,  # Vitesse en mode rapide
}

# ========== SÉLECTION DE L'IA ==========
IA_TYPE = "neuro_ga"  # "choreography" ou "neuro_ga" Change ici pour choisir l'IA !

"""
Tu peux maintenant changer facilement d'IA en modifiant juste cette variable :
- `"choreography"` → Algorithme génétique sur sequence d'actions (boucle ouverte)
- `"neuro_ga"`     → Neuroevolution : algo génétique sur les poids d'un MLP (boucle fermee, reactive)
- `"ppo"` → PPO (quand tu l'implémenteras)
- `"dqn"` → DQN (quand tu l'implémenteras)
- `"neat"` → NEAT (quand tu l'implémenteras)
"""
