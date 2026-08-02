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

# ========== FAÇONNAGE DE LA RÉCOMPENSE (shaped reward) ==========
# True = l'animal est pénalisé quand il touche le sol avec autre chose que ses
# pieds (genoux, torse, museau). Sans ça, le GA et PPO trouvent des solutions
# dégénérées : le renard marche sur les genoux ou se jette en avant, la poule
# se traîne sur le torse. Ces démarches avancent bien mais ne ressemblent à
# rien.
#
# Les appuis légitimes de chaque animal sont déclarés dans `foot_bones`
# (src/animals/fox.py et chicken.py). La pénalité est proportionnelle au temps
# passé en appui fautif : elle ampute le gain de distance de
# GROUND_CONTACT_WEIGHT (voir config_gen.py et config_ppo.py) pour un animal
# qui rampe en permanence, et de rien du tout pour un animal qui marche.
#
# False = comportement historique, les runs déjà publiés restent reproductibles.
SHAPED_REWARD = True


CONFIG = {
    'speed_multiplier': 50 if not DISPLAY_ENABLED else 1,  # Vitesse en mode rapide
}

# ========== SÉLECTION DE L'IA ==========
IA_TYPE = "ppo"  # "choreography", "neuro_ga" ou "ppo" Change ici pour choisir l'IA !

"""
Tu peux maintenant changer facilement d'IA en modifiant juste cette variable :
- `"choreography"` → Algorithme génétique sur sequence d'actions (boucle ouverte)
- `"neuro_ga"`     → Neuroevolution : algo génétique sur les poids d'un MLP (boucle fermee, reactive)
- `"ppo"`          → PPO (PyTorch). Entrainement : python train.py --algo ppo
                     main.py sert alors a visualiser la politique entrainee.
- `"dqn"` → DQN (quand tu l'implémenteras)
- `"neat"` → NEAT (quand tu l'implémenteras)

Entrainement RAPIDE (headless, parallele, recommande / Runpod) :
    python train.py --algo ga     (neuroevolution, tous les coeurs CPU)
    python train.py --algo ppo    (PPO vectorise, PyTorch)
"""
