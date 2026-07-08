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

from typing import Tuple

import numpy as np

from src.core_engine.physics import PhysicsWorld, Quadruped

DT = 1.0 / 60.0
SPAWN_X = 6.0


def run_episode(definition, policy, genome, max_frames: int,
                stagnation_frames: int = 0,
                stagnation_min_progress: float = 0.05) -> Tuple[float, int, bool]:
    """Joue un episode complet et retourne (distance, frames, is_fallen).

    Args:
        definition: AnimalDefinition (squelette + spawn_y)
        policy: NeuroPolicy (construction des entrees + MLP)
        genome: vecteur de poids du MLP
        max_frames: budget de frames (= ia.current_max_time)
        stagnation_frames: nb de frames sans progres avant arret (0 = off)
        stagnation_min_progress: progres minimal (m) pour reinitialiser le compteur
    """
    world = PhysicsWorld(gravity=(0, -10))
    quad = Quadruped(world, x=SPAWN_X, y=definition.spawn_y, definition=definition)
    genome = np.asarray(genome, dtype=np.float32)

    start_x = quad.body.body.position.x
    best_x = start_x
    last_progress_frame = 0
    is_fallen = False
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
    return float(distance), frame, is_fallen
