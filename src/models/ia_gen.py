"""
IA par neuroevolution : un petit reseau de neurones (MLP) dont les poids
sont evolues par un algorithme genetique.

Contrairement a IAChoreography (boucle ouverte, sequence figee d'actions),
cette IA est REACTIVE : elle prend l'etat du quadrupede en entree
(y compris la proprioception : angle et vitesse de chaque muscle actionne)
et choisit ses activations musculaires en continu.

Architecture du reseau (proprioception activee, 8 muscles) :
    [entree (7 + 2*8 = 23)] -> [cache (16) tanh] -> [sortie (8) tanh]

Le genome est le vecteur applati de tous les poids et biais.

Bonnes pratiques implementees (skill ai-training):
- Random seeds (random + numpy) pour reproductibilite
- Pydantic config (cf. config_gen.py) pour valider les hyperparametres
- MLflow tracking : params + metriques par generation + artefact final
- Convention de nommage : models/{name}_run-{NN}_date-{YYYY-MM-DD}/
- Archive des champions : le meilleur genome de CHAQUE generation est
  sauvegarde (results/{run}/champions.pkl) pour rejeu deterministe via replay.py
"""

import json
import os
import pickle
import random
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from src.models.ia_base import IABase
# MLP et NeuroPolicy vivent dans policy.py (module leger, sans mlflow) pour
# que les workers de train.py demarrent vite. Re-exportes ici pour compat
# retro (replay.py importait NeuroPolicy depuis ia_gen).
from src.models.policy import MLP, NeuroPolicy  # noqa: F401


def _seed_everything(seed: int) -> None:
    """Fixe les seeds pour la reproductibilite."""
    random.seed(seed)
    np.random.seed(seed)


def _next_run_number(models_dir: Path, model_name: str) -> int:
    """Determine le prochain numero de run en scannant models_dir."""
    if not models_dir.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(model_name)}_run-(\d+)_date-")
    existing = []
    for d in models_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                existing.append(int(m.group(1)))
    return (max(existing) + 1) if existing else 1


class IAGenetic(IABase):
    """IA neuroevolution : evolue les poids d'un MLP par algorithme genetique."""

    def __init__(self, config):
        super().__init__(config)

        nn_cfg = config.NN_CONFIG
        ga_cfg = config.GA_CONFIG
        train_cfg = config.TRAINING_CONFIG

        # ---- Reproductibilite : seed AVANT toute initialisation aleatoire ----
        self.seed = getattr(config, "SEED", 42)
        _seed_everything(self.seed)

        # ---- Politique (entrees + reseau + application) ----
        self.policy = NeuroPolicy(nn_cfg)
        self.input_size = self.policy.input_size          # taille effective (avec proprio)
        self.hidden_size = self.policy.hidden_size
        self.output_size = self.policy.output_size
        self.time_period = self.policy.time_period
        self.genome_length = self.policy.num_params

        self.fall_penalty = float(ga_cfg.get('fall_penalty', 100.0))

        # ---- Population ----
        pop_size = ga_cfg['population_size']
        init_std = ga_cfg['init_std']
        self.population: List[np.ndarray] = [
            np.random.randn(self.genome_length).astype(np.float32) * init_std
            for _ in range(pop_size)
        ]
        self.fitness_scores = [0.0] * pop_size

        # ---- Etat de l'episode ----
        self.current_frame = 0
        self.current_max_time = ga_cfg['base_time']
        self.best_reward_ever = 0.0

        # ---- Stats de generation ----
        self.generation_best = 0.0
        self.generation_avg = 0.0
        self.generation_start_time = datetime.now()

        # ---- Convention de nommage : models/{name}_run-NN_date-YYYY-MM-DD/ ----
        self.model_name = train_cfg['model_name']
        self.models_dir = Path(train_cfg['models_dir'])
        self.results_dir = Path(train_cfg['results_dir'])
        self.run_number = _next_run_number(self.models_dir, self.model_name)
        today = date.today().isoformat()
        run_id = f"{self.model_name}_run-{self.run_number:02d}_date-{today}"
        self.run_dir = self.models_dir / run_id
        self.results_run_dir = self.results_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_run_dir.mkdir(parents=True, exist_ok=True)

        # CSV de generation a cote des autres resultats du run.
        self.csv_file = str(self.results_run_dir / "generations.csv")

        # ---- Archive des champions (meilleur genome de chaque generation) ----
        self.champions: List[Dict[str, Any]] = []
        self.champions_file = self.results_run_dir / "champions.pkl"

        # ---- MLflow tracking ----
        mlflow.set_tracking_uri(train_cfg['mlflow_tracking_uri'])
        mlflow.set_experiment(train_cfg['mlflow_experiment_name'])
        self._mlflow_run = mlflow.start_run(run_name=run_id)
        self._log_initial_params()

        print(f"📁 Run dir : {self.run_dir}")
        print(f"📊 MLflow  : {train_cfg['mlflow_tracking_uri']} / "
              f"experiment={train_cfg['mlflow_experiment_name']}")
        print(f"🔢 Seed    : {self.seed}")
        print(f"🧠 Reseau  : {self.input_size} -> {self.hidden_size} -> {self.output_size} "
              f"({self.genome_length} params, proprio={self.policy.use_proprioception})")

    # ============ INFERENCE ============

    def get_action(self, time: float, dog_state: Dict[str, Any]) -> np.ndarray:
        genome = self.population[self.current_individual]
        out = self.policy.act(time, dog_state, genome)
        self.current_frame += 1
        return out

    def apply_to_quadruped(self, quadruped, action: np.ndarray):
        self.policy.apply(quadruped, action)

    # ============ CYCLE D'EVALUATION ============

    def reset_episode(self):
        self.current_frame = 0

    def on_episode_end(self, distance: float, frames_survived: int, dog_state: Dict[str, Any]):
        # Fitness = distance parcourue (x100), penalisee si le quadrupede tombe.
        # Plus de bonus de survie : rester immobile ne rapporte plus rien.
        is_fallen = bool(dog_state.get('is_fallen', False))
        fitness = distance * 100.0
        if is_fallen:
            fitness -= self.fall_penalty

        self.fitness_scores[self.current_individual] = fitness

        if fitness > self.best_reward_ever:
            self.best_reward_ever = fitness
            # NB: current_max_time n'est PAS modifie ici (fige pour la generation).
            print(f"🏆 Nouveau record intra-gen: {fitness:.2f}")

        if fitness > self.best_distance:
            self.best_distance = fitness

        self.current_individual += 1

    def should_reset_simulation(self) -> bool:
        if self.current_individual >= self.config.GA_CONFIG['population_size']:
            self._evolve_population()
            self.current_individual = 0
            self.generation += 1
        return True

    # ============ EVOLUTION ============

    def _evolve_population(self):
        self.generation_best = max(self.fitness_scores)
        self.generation_avg = sum(self.fitness_scores) / len(self.fitness_scores)

        # Champion de la generation (avant reset des scores) -> archive.
        best_idx = max(range(len(self.fitness_scores)),
                       key=lambda i: self.fitness_scores[i])
        self._record_champion(self.generation, self.fitness_scores[best_idx],
                              self.population[best_idx])

        self._save_generation_stats()
        self._log_generation_to_mlflow()

        sorted_idx = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )

        elite_count = max(1, self.config.GA_CONFIG['elite_size'])
        new_pop = [self.population[i].copy() for i in sorted_idx[:elite_count]]

        while len(new_pop) < self.config.GA_CONFIG['population_size']:
            p1 = self._tournament_selection(sorted_idx)
            p2 = self._tournament_selection(sorted_idx)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.fitness_scores = [0.0] * len(self.population)

        # Temps adaptatif : fige la duree d'episode pour TOUTE la generation
        # suivante (evite que les individus tardifs aient plus de temps).
        self.current_max_time = self._calculate_max_time_from_reward()

    def _tournament_selection(self, sorted_idx: List[int]) -> np.ndarray:
        """Selection par tournoi CORRIGEE : le gagnant est le candidat au
        meilleur fitness (auparavant `min(candidates)` prenait le plus petit
        indice de population, ce qui rendait la selection quasi aleatoire)."""
        size = self.config.GA_CONFIG.get('tournament_size', 3)
        pool = sorted_idx[:max(size, len(sorted_idx) // 2)]
        k = min(size, len(pool))
        candidates = random.sample(pool, k)
        winner = max(candidates, key=lambda i: self.fitness_scores[i])
        return self.population[winner].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if random.random() > self.config.GA_CONFIG['crossover_rate']:
            return p1.copy()
        mask = np.random.rand(len(p1)) < 0.5
        return np.where(mask, p1, p2).astype(p1.dtype)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        rate = self.config.GA_CONFIG['mutation_rate']
        strength = self.config.GA_CONFIG['mutation_strength']
        mask = np.random.rand(len(genome)) < rate
        noise = np.random.randn(len(genome)).astype(genome.dtype) * strength
        return genome + mask.astype(genome.dtype) * noise

    def _calculate_max_time_from_reward(self) -> int:
        if not self.config.GA_CONFIG.get('adaptive_time', True):
            return self.config.GA_CONFIG['base_time']
        threshold = self.config.GA_CONFIG.get('reward_threshold_for_max_time', 5000.0)
        progress = min(self.best_reward_ever / threshold, 1.0)
        time_range = (
            self.config.GA_CONFIG['max_time'] - self.config.GA_CONFIG['base_time']
        )
        return self.config.GA_CONFIG['base_time'] + int(progress * time_range)

    # ============ ARCHIVE DES CHAMPIONS ============

    def _nn_config_for_save(self) -> Dict[str, Any]:
        """Config minimale pour reconstruire la NeuroPolicy (rejeu)."""
        return {
            'input_size': self.policy.base_input,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'time_period': self.time_period,
            'use_proprioception': self.policy.use_proprioception,
            'max_muscle_speed': self.policy.max_muscle_speed,
        }

    def _record_champion(self, generation: int, fitness: float, genome: np.ndarray):
        """Ajoute le champion de la generation a l'archive et sauvegarde.

        Le rejeu est deterministe : conserver le genome suffit a reproduire
        exactement le mouvement (pas besoin de logger les trajectoires).
        """
        self.champions.append({
            'generation': int(generation),
            'fitness': float(fitness),
            'genome': np.asarray(genome, dtype=np.float32).copy(),
        })
        self._save_champions()

    def _save_champions(self):
        archive = {
            'nn_config': self._nn_config_for_save(),
            'model_name': self.model_name,
            'run_number': self.run_number,
            'champions': self.champions,
        }
        with open(self.champions_file, 'wb') as f:
            pickle.dump(archive, f)

    # ============ LOGGING ============

    def _log_initial_params(self):
        """Log tous les hyperparametres une fois au debut du run."""
        params = {
            "model_name": self.model_name,
            "run_number": self.run_number,
            "seed": self.seed,
            "genome_length": self.genome_length,
            "input_size_effective": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "use_proprioception": self.policy.use_proprioception,
            "time_period": self.time_period,
            "fall_penalty": self.fall_penalty,
            **{f"ga_{k}": v for k, v in self.config.GA_CONFIG.items()},
            **{f"train_{k}": v for k, v in self.config.TRAINING_CONFIG.items()
               if k not in {"mlflow_tracking_uri", "mlflow_experiment_name"}},
        }
        # Pydantic settings : variables d'env / .env potentiellement appliques.
        if hasattr(self.config, "SETTINGS"):
            mlflow.set_tag("pydantic_config", "true")
        mlflow.log_params(params)

    def _log_generation_to_mlflow(self):
        metrics = {
            "fitness_best": float(self.generation_best),
            "fitness_avg": float(self.generation_avg),
            "fitness_worst": float(min(self.fitness_scores)) if self.fitness_scores else 0.0,
            "fitness_std": float(pd.Series(self.fitness_scores).std()) if self.fitness_scores else 0.0,
            "best_distance_ever": float(self.best_distance),
            "best_reward_ever": float(self.best_reward_ever),
            "current_max_time": float(self.current_max_time),
        }
        mlflow.log_metrics(metrics, step=self.generation)

    def _save_generation_stats(self):
        """Sauvegarde aussi en CSV (utile pour analyses Power BI existantes)."""
        os.makedirs(os.path.dirname(self.csv_file) or '.', exist_ok=True)

        end = datetime.now()
        duration = (end - self.generation_start_time).total_seconds()
        data = {
            'generation': self.generation,
            'timestamp': end.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'fitness_best': self.generation_best,
            'fitness_avg': self.generation_avg,
            'fitness_worst': min(self.fitness_scores) if self.fitness_scores else 0.0,
            'fitness_std': float(pd.Series(self.fitness_scores).std()) if self.fitness_scores else 0.0,
            'best_distance_ever': self.best_distance,
            'current_max_time': self.current_max_time,
            'population_size': len(self.population),
            'mutation_rate': self.config.GA_CONFIG['mutation_rate'],
            'mutation_strength': self.config.GA_CONFIG['mutation_strength'],
            'elite_size': self.config.GA_CONFIG['elite_size'],
        }
        df = pd.DataFrame([data])
        if os.path.exists(self.csv_file):
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_file, mode='w', header=True, index=False)

        self.generation_start_time = datetime.now()

    # ============ I/O ============

    def _build_save_data(self) -> Dict[str, Any]:
        return {
            'population': self.population,
            'fitness_scores': self.fitness_scores,
            'generation': self.generation,
            'best_distance': self.best_distance,
            'best_reward_ever': self.best_reward_ever,
            'current_max_time': self.current_max_time,
            'seed': self.seed,
            'genome_length': self.genome_length,
            'nn_config': self._nn_config_for_save(),
            'parameters': {
                'population_size': self.config.GA_CONFIG['population_size'],
                'mutation_rate': self.config.GA_CONFIG['mutation_rate'],
                'mutation_strength': self.config.GA_CONFIG['mutation_strength'],
                'crossover_rate': self.config.GA_CONFIG['crossover_rate'],
                'elite_size': self.config.GA_CONFIG['elite_size'],
            },
        }

    def save(self, filepath: Optional[str] = None):
        """Sauvegarde le modele.

        Convention : par defaut on ecrit dans models/{name}_run-NN_date-YYYY-MM-DD/best_model.pkl.
        Le parametre `filepath` reste accepte pour la compat retro avec main.py
        (chargement initial via TRAINING_CONFIG['save_file']).
        """
        save_data = self._build_save_data()

        primary = self.run_dir / "best_model.pkl"
        primary.parent.mkdir(parents=True, exist_ok=True)
        with open(primary, 'wb') as f:
            pickle.dump(save_data, f)

        # S'assure que l'archive des champions est a jour sur le disque.
        self._save_champions()

        # Snapshot des metriques courantes pour reference rapide.
        metrics = {
            "generation": self.generation,
            "best_distance": self.best_distance,
            "best_reward_ever": self.best_reward_ever,
            "current_max_time": self.current_max_time,
            "num_champions": len(self.champions),
        }
        with open(self.results_run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Copie compat retro a l'emplacement legacy si demande.
        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

        # Log MLflow : artefact + metrique snapshot.
        mlflow.log_artifact(str(primary), artifact_path="models")
        mlflow.log_artifact(str(self.results_run_dir / "metrics.json"),
                            artifact_path="results")

        print(f"✅ Modele sauvegarde : {primary}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier {filepath} introuvable")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        saved_pop = data.get('population', [])
        if not saved_pop:
            print("⚠️ Sauvegarde vide : nouvelle IA conservee.")
            return

        # Verifie la compatibilite d'architecture (taille du genome).
        saved_len = int(np.asarray(saved_pop[0]).ravel().shape[0])
        if saved_len != self.genome_length:
            print(
                f"⚠️ Checkpoint incompatible (genome {saved_len} vs "
                f"{self.genome_length} attendu). Nouvelle IA demarree.\n"
                f"   Le checkpoint reste rejouable avec son ancienne "
                f"architecture (replay.py)."
            )
            return

        self.population = [np.asarray(g, dtype=np.float32) for g in saved_pop]
        self.fitness_scores = data.get(
            'fitness_scores', [0.0] * len(self.population)
        )
        self.generation = data.get('generation', 0)
        self.best_distance = data.get('best_distance', 0.0)
        self.best_reward_ever = data.get('best_reward_ever', 0.0)
        self.current_max_time = data.get(
            'current_max_time', self.config.GA_CONFIG['base_time']
        )
        print(
            f"✅ Chargement IA neuroevolution : Gen {self.generation}, "
            f"Best {self.best_distance:.2f}, Time {self.current_max_time}f"
        )

    # ============ FIN DE SESSION ============

    def close(self):
        """Termine proprement le run MLflow. A appeler depuis main.py."""
        # Sauvegarde finale de l'archive des champions.
        self._save_champions()
        if self._mlflow_run is not None:
            try:
                mlflow.log_artifact(str(self.champions_file), artifact_path="results")
            except Exception:
                pass
            mlflow.end_run()
            self._mlflow_run = None
            print("📊 Run MLflow termine")

    # ============ STATS / HOOKS ============

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            'generation_best': self.generation_best,
            'generation_avg': self.generation_avg,
            'population_size': len(self.population),
            'current_max_time': self.current_max_time,
        })
        return stats

    def on_generation_end(self):
        print(
            f"🧬 Génération {self.generation} | "
            f"Best: {self.generation_best:.2f} | "
            f"Avg: {self.generation_avg:.2f} | "
            f"All-time: {self.best_distance:.2f} | "
            f"Time: {self.current_max_time}f ({self.current_max_time / 60:.1f}s)"
        )

        if (self.generation + 1) % self.config.TRAINING_CONFIG['save_every'] == 0:
            self.save(self.config.TRAINING_CONFIG.get('save_file'))
            print(f"💾 Sauvegarde périodique (Gen {self.generation})")
