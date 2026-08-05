"""Evaluation HEADLESS d'un genome : un episode complet sans pygame.

Utilise par train.py (dans les workers multiprocessing). Reproduit
EXACTEMENT la semantique d'un episode de main.py :
- etat construit AVANT le pas de physique,
- episode_time incremente avant l'action (t = frame * DT),
- fin d'episode sur chute (is_upside_down) ou budget de frames atteint,
- distance = x final - x de depart, dt fixe a 1/60 (physique fidele).

S'y ajoute une TERMINAISON ANTICIPEE optionnelle : si l'animal n'a pas
progresse d'au moins `stagnation_min_progress` metres depuis
`stagnation_frames` frames, l'episode s'arrete (fitness inchangee, temps
de calcul economise). Mettre stagnation_frames=0 pour desactiver.

Module leger : numpy + Box2D uniquement (pas de mlflow/pandas/pygame).
"""

import math
from typing import Tuple

import numpy as np

from src.core_engine.physics import PhysicsWorld, Quadruped

DT = 1.0 / 60.0
SPAWN_X = 6.0


def run_episode(definition, policy, genome, max_frames: int,
                stagnation_frames: int = 0,
                stagnation_min_progress: float = 0.05,
                track_ground_contacts: bool = False
                ) -> Tuple[float, int, bool, float, int]:
    """Joue un episode et retourne (distance, frames, is_fallen, uprightness, bad_frames).

    uprightness = moyenne de cos(angle du corps) sur l'episode (1 = dos
    horizontal, 0 = vertical, negatif = retourne). Sert au bonus de stabilite.

    bad_frames = nombre de frames pendant lesquelles un os autre qu'un pied
    touchait le sol (l'animal rampe ou marche sur les genoux). Vaut toujours 0
    si track_ground_contacts est False : la detection parcourt les contacts de
    tous les os, autant ne la payer que quand le shaped reward l'exploite.

    Args:
        definition: AnimalDefinition (squelette + spawn_y)
        policy: NeuroPolicy (construction des entrees + MLP)
        genome: vecteur de poids du MLP
        max_frames: budget de frames (= ia.current_max_time)
        stagnation_frames: nb de frames sans progres avant arret (0 = off)
        stagnation_min_progress: progres minimal (m) pour reinitialiser le compteur
        track_ground_contacts: compter les frames en appui fautif
    """
    genome = np.asarray(genome, dtype=np.float32)

    # Squelette EVOLUTIF : quand la politique reserve des genes de morphologie,
    # la definition recue n'est qu'un gabarit par defaut. Chaque individu se
    # reconstruit un corps depuis SES genes, si bien que le GA fait evoluer le
    # corps et le cerveau dans le meme genome.
    morpho, _ = policy.split_genome(genome)
    if morpho is not None:
        from src.animals.alien import build_alien
        definition = build_alien(morpho)

    world = PhysicsWorld(gravity=(0, -10))
    quad = Quadruped(world, x=SPAWN_X, y=definition.spawn_y, definition=definition)

    start_x = quad.body.body.position.x
    best_x = start_x
    last_progress_frame = 0
    is_fallen = False
    uprightness_sum = 0.0
    bad_contact_frames = 0
    frame = 0

    for frame in range(1, max_frames + 1):
        t = frame * DT  # meme convention que main.py (increment avant action)

        body = quad.body.body
        dog_state = {
            'position': (body.position.x, body.position.y),
            'velocity': (body.linearVelocity.x, body.linearVelocity.y),
            'angle': body.angle,
            'muscle_angles': [m.get_angle() for m in quad.muscles],
            'muscle_speeds': [m.get_speed() for m in quad.muscles],
        }

        action = policy.act(t, dog_state, genome)
        policy.apply(quad, action)

        quad.update()
        world.step(DT)

        # Stabilite : cos de l'angle du corps APRES le pas de physique.
        uprightness_sum += math.cos(quad.body.body.angle)

        # Appui fautif : un os autre qu'un pied touche le sol.
        if track_ground_contacts and quad.has_bad_ground_contact():
            bad_contact_frames += 1

        # Fin d'episode : chute (comme main.py).
        if quad.is_upside_down():
            is_fallen = True
            break

        # Terminaison anticipee : aucun progres depuis trop longtemps.
        if stagnation_frames > 0:
            x = body.position.x
            if x > best_x + stagnation_min_progress:
                best_x = x
                last_progress_frame = frame
            elif frame - last_progress_frame >= stagnation_frames:
                break

    distance = quad.body.body.position.x - start_x
    uprightness = uprightness_sum / max(frame, 1)
    return float(distance), frame, is_fallen, float(uprightness), bad_contact_frames
