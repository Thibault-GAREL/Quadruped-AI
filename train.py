"""Entrainement HEADLESS et PARALLELE (sans pygame, sans rendu).

C'est le point d'entree pour les gros entrainements (local ou Runpod) :
- physique a dt fixe 1/60 (fidele a ce qu'on voit dans main.py),
- GA : population evaluee en parallele sur tous les coeurs CPU,
- PPO : environnements vectorises + PyTorch (voir src/models/ia_ppo.py),
- terminaison anticipee des episodes qui stagnent,
- barre de progression tqdm + logs MLflow identiques a main.py.

Usage :
    python train.py                          # GA, animal de src/config.py
    python train.py --algo ga --generations 300
    python train.py --algo ppo
    python train.py --algo ga --workers 8

L'animal s'choisit via ANIMAL dans src/config.py (fox / chicken).
Pour lancer en arriere-plan avec suivi (PowerShell) :
    python train.py *> outputs\\logs\\train.log ; puis Get-Content -Wait
"""

import argparse
import multiprocessing
import pickle
import time

import src.config as config_ia


# ============ BARRE DE PROGRESSION (tqdm si dispo, fallback sinon) ============

def _progress(iterable, total, desc):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False, ncols=90)
    except ImportError:
        return iterable  # les logs de generation restent affiches


# ============ WORKERS GA (multiprocessing) ============
# Chaque worker garde en memoire la definition de l'animal et la politique
# (initialisees une seule fois), et reconstruit un monde Box2D par episode.

_WORKER = {}


def _init_worker(animal_name, nn_config, stagnation_frames, stagnation_min_progress):
    from src.animals import get_animal
    from src.models.policy import NeuroPolicy
    _WORKER['definition'] = get_animal(animal_name)
    _WORKER['policy'] = NeuroPolicy(nn_config)
    _WORKER['stagnation'] = (stagnation_frames, stagnation_min_progress)


def _eval_genome(task):
    genome, max_frames = task
    from src.models.evaluate import run_episode
    stag_frames, stag_progress = _WORKER['stagnation']
    return run_episode(_WORKER['definition'], _WORKER['policy'], genome, max_frames,
                       stagnation_frames=stag_frames,
                       stagnation_min_progress=stag_progress)


# ============ ENTRAINEMENT GA ============

def train_ga(args):
    import src.models.config_gen as cfg
    from src.models.ia_gen import IAGenetic

    ia = IAGenetic(cfg)
    try:
        ia.load(cfg.TRAINING_CONFIG['save_file'])
        print(f"   IA chargée: Génération {ia.generation}")
    except FileNotFoundError:
        print("   Nouvelle IA créée")
    except (ValueError, KeyError, AssertionError, pickle.UnpicklingError) as e:
        print(f"   Checkpoint incompatible ({e}). Nouvelle IA créée")

    n_workers = args.workers or cfg.SETTINGS.N_WORKERS or multiprocessing.cpu_count()
    max_generations = args.generations or cfg.TRAINING_CONFIG['max_generations']
    target_generation = ia.generation + max_generations if args.generations \
        else max_generations

    print(f"🚀 Entraînement GA headless : {n_workers} workers, "
          f"objectif génération {target_generation}")
    print(f"   Terminaison anticipée : "
          f"{cfg.SETTINGS.STAGNATION_FRAMES} frames sans progrès "
          f"(0 = désactivée)")

    pool = multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(config_ia.ANIMAL, cfg.NN_CONFIG,
                  cfg.SETTINGS.STAGNATION_FRAMES,
                  cfg.SETTINGS.STAGNATION_MIN_PROGRESS),
    )

    start_time = time.time()
    episodes_done = 0
    try:
        while ia.generation < target_generation:
            max_frames = int(ia.current_max_time)
            tasks = [(genome, max_frames) for genome in ia.population]

            # imap conserve l'ordre : fitness_scores[i] correspond bien a population[i].
            results = list(_progress(pool.imap(_eval_genome, tasks),
                                     total=len(tasks),
                                     desc=f"Gen {ia.generation}"))

            for distance, frames, is_fallen in results:
                ia.on_episode_end(distance, frames, {'is_fallen': is_fallen})
                episodes_done += 1

            prev_generation = ia.generation
            ia.should_reset_simulation()  # evolue la population + generation++
            if ia.generation != prev_generation:
                ia.on_generation_end()
            ia.reset_episode()

    except KeyboardInterrupt:
        print("\n⏹️ Interruption clavier : sauvegarde du checkpoint...")
    finally:
        pool.terminate()
        pool.join()
        ia.save(cfg.TRAINING_CONFIG['save_file'])
        ia.close()

    elapsed = time.time() - start_time
    if elapsed > 0 and episodes_done:
        print(f"⏱️ {episodes_done} épisodes en {elapsed:.1f}s "
              f"({episodes_done / elapsed:.1f} épisodes/s)")


# ============ ENTRAINEMENT PPO ============

def train_ppo(args):
    # Import local : torch n'est necessaire que pour PPO.
    try:
        from src.models.ia_ppo import run_training
    except ImportError as e:
        raise SystemExit(
            f"PPO indisponible ({e}).\n"
            "Installer PyTorch dans le venv :\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install tqdm"
        )
    run_training(animal_name=config_ia.ANIMAL, total_updates=args.updates)


# ============ POINT D'ENTREE ============

def main():
    parser = argparse.ArgumentParser(description="Entrainement headless (GA ou PPO)")
    parser.add_argument("--algo", choices=["ga", "ppo"], default="ga",
                        help="Algorithme d'entrainement (defaut: ga)")
    parser.add_argument("--generations", type=int, default=0,
                        help="GA : nombre de generations A AJOUTER (defaut: config)")
    parser.add_argument("--workers", type=int, default=0,
                        help="GA : nombre de processus (defaut: config ou tous les coeurs)")
    parser.add_argument("--updates", type=int, default=0,
                        help="PPO : nombre d'updates (defaut: config)")
    args = parser.parse_args()

    print(f"🐾 Animal : {config_ia.ANIMAL} | Algo : {args.algo.upper()}")

    if args.algo == "ga":
        train_ga(args)
    else:
        train_ppo(args)


if __name__ == "__main__":
    # Indispensable sous Windows (multiprocessing en mode spawn).
    multiprocessing.freeze_support()
    main()
