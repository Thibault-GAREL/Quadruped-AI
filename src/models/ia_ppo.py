"""PPO custom en PyTorch pour le quadrupede/bipede Box2D.

Pourquoi PPO ici : contrairement au GA (boucle ouverte sur les poids),
PPO exploite CHAQUE step de simulation (gradient sur toutes les
transitions), ce qui le rend beaucoup plus efficace en echantillons.

Implementation (boucle custom, pas de Stable-Baselines) :
- environnements Box2D vectorises en-process (N_ENVS mondes independants),
- observations = 7 features de base + proprioception (comme la neuroevolution),
  normalisees par moyenne/variance glissantes (RunningMeanStd),
- actor-critic MLP separes, actions continues via gaussienne diagonale,
- GAE (lambda), clipping PPO, bonus d'entropie, gradient clipping,
- bootstrap correct des episodes TRONQUES (temps/stagnation) vs TERMINES (chute),
- MLflow + convention models/{name}_run-NN_date-YYYY-MM-DD/ (skill ai-training),
- checkpoint reprennable (poids + optimiseur + stats de normalisation).

Entrainement : python train.py --algo ppo   (headless, sans pygame)
Visualisation : IA_TYPE = "ppo" dans src/config.py puis python main.py
    (IAPPOPlayer charge le checkpoint et joue la politique en deterministe).
"""

import json
import os
import pickle
import random
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.animals import get_animal
from src.core_engine.physics import PhysicsWorld, Quadruped
from src.models.ia_base import IABase

DT = 1.0 / 60.0
SPAWN_X = 6.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _next_run_number(models_dir: Path, model_name: str) -> int:
    """Prochain numero de run (meme convention que ia_gen)."""
    if not models_dir.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(model_name)}_run-(\d+)_date-")
    existing = [int(m.group(1)) for d in models_dir.iterdir()
                if d.is_dir() and (m := pattern.match(d.name))]
    return (max(existing) + 1) if existing else 1


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ============ OBSERVATIONS ============

def obs_dim_for(act_dim: int) -> int:
    return 7 + 2 * act_dim


def build_obs(t: float, dog_state: Dict[str, Any], act_dim: int,
              time_period: float, max_muscle_speed: float,
              ref_y: float) -> np.ndarray:
    """Memes features que NeuroPolicy, hauteur normalisee par ref_y (spawn)."""
    _, y = dog_state['position']
    vx, vy = dog_state['velocity']
    angle = dog_state['angle']

    phase = 2.0 * np.pi * t / time_period
    feats: List[float] = [
        np.sin(phase),
        np.cos(phase),
        np.clip(vx / 5.0, -1.0, 1.0),
        np.clip(vy / 5.0, -1.0, 1.0),
        np.sin(angle),
        np.cos(angle),
        np.clip((y - ref_y) / 2.0, -1.0, 1.0),
    ]
    m_ang = dog_state.get('muscle_angles', [])
    m_spd = dog_state.get('muscle_speeds', [])
    for i in range(act_dim):
        a = m_ang[i] if i < len(m_ang) else 0.0
        s = m_spd[i] if i < len(m_spd) else 0.0
        feats.append(float(np.clip(a / np.pi, -1.0, 1.0)))
        feats.append(float(np.clip(s / max_muscle_speed, -1.0, 1.0)))
    return np.array(feats, dtype=np.float32)


class RunningMeanStd:
    """Normalisation glissante des observations (algorithme de Welford)."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        b_mean = batch.mean(axis=0)
        b_var = batch.var(axis=0)
        b_count = batch.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean += delta * b_count / total
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta ** 2 * self.count * b_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = np.asarray(state['mean'], dtype=np.float64)
        self.var = np.asarray(state['var'], dtype=np.float64)
        self.count = float(state['count'])


# ============ ENVIRONNEMENT ============

class QuadrupedEnv:
    """Un monde Box2D + un animal, interface reset/step (sans pygame).

    Recompense par step : progres en x * 100 (meme echelle que la fitness GA),
    moins un cout d'action optionnel. Chute -> terminated + penalite.
    Temps/stagnation -> truncated (bootstrap de la valeur, pas de penalite).
    """

    def __init__(self, definition, settings):
        self.definition = definition
        self.s = settings
        self.act_dim = settings.ACT_DIM
        self.world: Optional[PhysicsWorld] = None
        self.quad: Optional[Quadruped] = None
        self.frame = 0
        self.start_x = 0.0
        self.best_x = 0.0
        self.last_progress_frame = 0

    def _dog_state(self) -> Dict[str, Any]:
        body = self.quad.body.body
        return {
            'position': (body.position.x, body.position.y),
            'velocity': (body.linearVelocity.x, body.linearVelocity.y),
            'angle': body.angle,
            'muscle_angles': [m.get_angle() for m in self.quad.muscles],
            'muscle_speeds': [m.get_speed() for m in self.quad.muscles],
        }

    def _obs(self) -> np.ndarray:
        return build_obs(self.frame * DT, self._dog_state(), self.act_dim,
                         self.s.TIME_PERIOD, self.s.MAX_MUSCLE_SPEED,
                         self.definition.spawn_y)

    def reset(self) -> np.ndarray:
        self.world = PhysicsWorld(gravity=(0, -10))
        self.quad = Quadruped(self.world, x=SPAWN_X, y=self.definition.spawn_y,
                              definition=self.definition)
        self.frame = 0
        self.start_x = self.quad.body.body.position.x
        self.best_x = self.start_x
        self.last_progress_frame = 0
        return self._obs()

    def step(self, action: np.ndarray):
        self.frame += 1
        x_before = self.quad.body.body.position.x

        for i in range(self.act_dim):
            self.quad.set_muscle_activation(i, float(action[i]))
        self.quad.update()
        self.world.step(DT)

        x = self.quad.body.body.position.x
        reward = (x - x_before) * 100.0
        if self.s.ACTION_COST > 0:
            reward -= self.s.ACTION_COST * float(np.sum(np.square(action)))

        terminated = self.quad.is_upside_down()
        if terminated:
            reward -= self.s.FALL_PENALTY

        truncated = False
        if not terminated:
            if x > self.best_x + self.s.STAGNATION_MIN_PROGRESS:
                self.best_x = x
                self.last_progress_frame = self.frame
            stagnant = (self.s.STAGNATION_FRAMES > 0 and
                        self.frame - self.last_progress_frame >= self.s.STAGNATION_FRAMES)
            truncated = self.frame >= self.s.MAX_EPISODE_FRAMES or stagnant

        info = {}
        if terminated or truncated:
            info['episode'] = {
                'distance': x - self.start_x,
                'frames': self.frame,
                'fallen': terminated,
            }
        return self._obs(), reward, terminated, truncated, info


class VecEnvs:
    """N environnements en-process, auto-reset, avec obs terminale exposee
    (necessaire pour bootstrapper correctement les episodes tronques)."""

    def __init__(self, definition, settings, n_envs: int):
        self.envs = [QuadrupedEnv(definition, settings) for _ in range(n_envs)]

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions: np.ndarray):
        obs_batch, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                info['terminal_obs'] = obs
                obs = env.reset()
            obs_batch.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return (np.stack(obs_batch), np.array(rewards, dtype=np.float32),
                np.array(terminateds), np.array(truncateds), infos)


# ============ RESEAU ACTOR-CRITIC ============

class ActorCritic(nn.Module):
    """Acteur (gaussienne diagonale, moyenne tanh) et critique separes."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, log_std_init: float):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, torch.exp(self.log_std))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist = self.dist(obs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob, self.value(obs)

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


# ============ ENTRAINEMENT ============

def run_training(animal_name: str, total_updates: int = 0):
    """Boucle PPO complete (appelee par train.py --algo ppo)."""
    import mlflow

    from src.models import config_ppo as cfg
    s = cfg.SETTINGS

    _seed_everything(s.SEED)
    device = _resolve_device(s.DEVICE)
    definition = get_animal(animal_name)

    act_dim = s.ACT_DIM
    obs_dim = obs_dim_for(act_dim)
    total_updates = total_updates or s.TOTAL_UPDATES

    # ----- Convention de nommage des runs (skill ai-training) -----
    models_dir = Path(s.MODELS_DIR)
    results_dir = Path(s.RESULTS_DIR)
    run_number = _next_run_number(models_dir, s.MODEL_NAME)
    run_id = f"{s.MODEL_NAME}_run-{run_number:02d}_date-{date.today().isoformat()}"
    run_dir = models_dir / run_id
    results_run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_run_dir.mkdir(parents=True, exist_ok=True)

    # ----- Modele, optimiseur, normalisation -----
    net = ActorCritic(obs_dim, act_dim, s.HIDDEN_SIZE, s.LOG_STD_INIT).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=s.LEARNING_RATE)
    obs_rms = RunningMeanStd(obs_dim)
    start_update = 0
    best_distance = -1e9

    # Reprise depuis le checkpoint courant s'il existe (pratique sur Runpod).
    save_file = Path(s.SAVE_FILE)
    if save_file.exists():
        ckpt = torch.load(save_file, map_location=device, weights_only=False)
        if ckpt.get('obs_dim') == obs_dim and ckpt.get('act_dim') == act_dim:
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            obs_rms.load_state_dict(ckpt['obs_rms'])
            start_update = int(ckpt.get('update', 0))
            best_distance = float(ckpt.get('best_distance', -1e9))
            print(f"✅ Reprise du checkpoint {save_file} (update {start_update})")
        else:
            print(f"⚠️ Checkpoint {save_file} incompatible (animal different ?). "
                  "Nouveau modele.")

    # ----- MLflow -----
    mlflow.set_tracking_uri(s.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(s.MLFLOW_EXPERIMENT_NAME)
    mlflow.start_run(run_name=run_id)
    mlflow.log_params({
        'animal': animal_name, 'seed': s.SEED, 'device': str(device),
        'obs_dim': obs_dim, 'act_dim': act_dim, 'hidden': s.HIDDEN_SIZE,
        'n_envs': s.N_ENVS, 'n_steps': s.N_STEPS, 'lr': s.LEARNING_RATE,
        'gamma': s.GAMMA, 'gae_lambda': s.GAE_LAMBDA, 'clip': s.CLIP_RANGE,
        'entropy_coef': s.ENTROPY_COEF, 'value_coef': s.VALUE_COEF,
        'epochs': s.N_EPOCHS, 'minibatch': s.MINIBATCH_SIZE,
        'max_episode_frames': s.MAX_EPISODE_FRAMES,
        'stagnation_frames': s.STAGNATION_FRAMES,
        'fall_penalty': s.FALL_PENALTY, 'total_updates': total_updates,
    })

    print(f"📁 Run dir : {run_dir}")
    print(f"🧠 PPO {obs_dim} -> {s.HIDDEN_SIZE} -> {act_dim} sur {device} | "
          f"{s.N_ENVS} envs x {s.N_STEPS} steps = "
          f"{s.N_ENVS * s.N_STEPS} transitions/update")

    envs = VecEnvs(definition, s, s.N_ENVS)
    raw_obs = envs.reset()
    if s.NORM_OBS:
        obs_rms.update(raw_obs)
    obs = obs_rms.normalize(raw_obs) if s.NORM_OBS else raw_obs.astype(np.float32)

    try:
        from tqdm import trange
        update_range = trange(start_update, total_updates, desc="PPO", ncols=90)
    except ImportError:
        update_range = range(start_update, total_updates)

    ep_distances: List[float] = []
    ep_frames: List[int] = []
    ep_falls: List[bool] = []

    try:
        for update in update_range:
            t0 = time.time()

            # ================= COLLECTE =================
            T, N = s.N_STEPS, s.N_ENVS
            b_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
            b_actions = np.zeros((T, N, act_dim), dtype=np.float32)
            b_logprobs = np.zeros((T, N), dtype=np.float32)
            b_rewards = np.zeros((T, N), dtype=np.float32)
            b_values = np.zeros((T, N), dtype=np.float32)
            b_dones = np.zeros((T, N), dtype=np.float32)  # terminated OU truncated

            for step in range(T):
                obs_t = torch.as_tensor(obs, device=device)
                action, logprob, value = net.act(obs_t)
                action_np = action.cpu().numpy()

                raw_next, rewards, terminateds, truncateds, infos = envs.step(
                    np.clip(action_np, -1.0, 1.0))

                # Episodes tronques (temps/stagnation) : bootstrap de la valeur
                # de l'etat terminal (sinon le critique croit que tout s'arrete).
                for i, info in enumerate(infos):
                    if truncateds[i] and not terminateds[i]:
                        term_obs = info['terminal_obs']
                        term_obs = (obs_rms.normalize(term_obs[None])[0]
                                    if s.NORM_OBS else term_obs)
                        with torch.no_grad():
                            v_term = net.value(torch.as_tensor(
                                term_obs[None], device=device)).item()
                        rewards[i] += s.GAMMA * v_term
                    if 'episode' in info:
                        ep_distances.append(info['episode']['distance'])
                        ep_frames.append(info['episode']['frames'])
                        ep_falls.append(info['episode']['fallen'])

                b_obs[step] = obs
                b_actions[step] = action_np
                b_logprobs[step] = logprob.cpu().numpy()
                b_rewards[step] = rewards
                b_values[step] = value.cpu().numpy()
                b_dones[step] = np.logical_or(terminateds, truncateds)

                if s.NORM_OBS:
                    obs_rms.update(raw_next)
                    obs = obs_rms.normalize(raw_next)
                else:
                    obs = raw_next.astype(np.float32)

            # ================= GAE =================
            with torch.no_grad():
                next_value = net.value(
                    torch.as_tensor(obs, device=device)).cpu().numpy()

            advantages = np.zeros_like(b_rewards)
            last_gae = np.zeros(N, dtype=np.float32)
            for step in reversed(range(T)):
                nonterminal = 1.0 - b_dones[step]
                v_next = next_value if step == T - 1 else b_values[step + 1]
                delta = (b_rewards[step] + s.GAMMA * v_next * nonterminal
                         - b_values[step])
                last_gae = delta + s.GAMMA * s.GAE_LAMBDA * nonterminal * last_gae
                advantages[step] = last_gae
            returns = advantages + b_values

            # ================= UPDATE PPO =================
            batch = T * N
            f_obs = torch.as_tensor(b_obs.reshape(batch, obs_dim), device=device)
            f_actions = torch.as_tensor(b_actions.reshape(batch, act_dim), device=device)
            f_logprobs = torch.as_tensor(b_logprobs.reshape(batch), device=device)
            f_adv = torch.as_tensor(advantages.reshape(batch), device=device)
            f_returns = torch.as_tensor(returns.reshape(batch), device=device)
            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

            idx = np.arange(batch)
            pg_losses, v_losses, entropies, kls, clip_fracs = [], [], [], [], []
            for _ in range(s.N_EPOCHS):
                np.random.shuffle(idx)
                for start in range(0, batch, s.MINIBATCH_SIZE):
                    mb = idx[start:start + s.MINIBATCH_SIZE]
                    dist = net.dist(f_obs[mb])
                    logprob = dist.log_prob(f_actions[mb]).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                    ratio = torch.exp(logprob - f_logprobs[mb])

                    pg1 = -f_adv[mb] * ratio
                    pg2 = -f_adv[mb] * torch.clamp(
                        ratio, 1 - s.CLIP_RANGE, 1 + s.CLIP_RANGE)
                    pg_loss = torch.max(pg1, pg2).mean()

                    value = net.value(f_obs[mb])
                    v_loss = 0.5 * (value - f_returns[mb]).pow(2).mean()

                    loss = (pg_loss + s.VALUE_COEF * v_loss
                            - s.ENTROPY_COEF * entropy)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), s.MAX_GRAD_NORM)
                    optimizer.step()

                    with torch.no_grad():
                        pg_losses.append(pg_loss.item())
                        v_losses.append(v_loss.item())
                        entropies.append(entropy.item())
                        kls.append((f_logprobs[mb] - logprob).mean().item())
                        clip_fracs.append(
                            ((ratio - 1.0).abs() > s.CLIP_RANGE).float().mean().item())

            # ================= LOGS + CHECKPOINTS =================
            fps = batch / max(time.time() - t0, 1e-6)
            recent_d = ep_distances[-50:]
            mean_distance = float(np.mean(recent_d)) if recent_d else 0.0
            metrics = {
                'ep_distance_mean': mean_distance,
                'ep_frames_mean': float(np.mean(ep_frames[-50:])) if ep_frames else 0.0,
                'ep_fall_rate': float(np.mean(ep_falls[-50:])) if ep_falls else 0.0,
                'policy_loss': float(np.mean(pg_losses)),
                'value_loss': float(np.mean(v_losses)),
                'entropy': float(np.mean(entropies)),
                'approx_kl': float(np.mean(kls)),
                'clip_fraction': float(np.mean(clip_fracs)),
                'fps': fps,
                'total_steps': (update + 1) * batch,
                'log_std_mean': float(net.log_std.detach().mean().item()),
            }
            mlflow.log_metrics(metrics, step=update)
            if hasattr(update_range, 'set_postfix'):
                update_range.set_postfix(dist=f"{mean_distance:.2f}m",
                                         fps=f"{fps:.0f}")

            def save_checkpoint(path: Path):
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'obs_rms': obs_rms.state_dict(),
                    'obs_dim': obs_dim, 'act_dim': act_dim,
                    'hidden': s.HIDDEN_SIZE,
                    'log_std_init': s.LOG_STD_INIT,
                    'norm_obs': s.NORM_OBS,
                    'time_period': s.TIME_PERIOD,
                    'max_muscle_speed': s.MAX_MUSCLE_SPEED,
                    'animal': animal_name,
                    'update': update + 1,
                    'best_distance': best_distance,
                }, path)

            if mean_distance > best_distance and recent_d:
                best_distance = mean_distance
                save_checkpoint(run_dir / "best_model.pt")

            if (update + 1) % s.SAVE_EVERY == 0 or update == total_updates - 1:
                save_checkpoint(run_dir / "last_model.pt")
                save_checkpoint(save_file)  # checkpoint courant (reprise/main.py)
                with open(results_run_dir / "metrics.json", 'w') as f:
                    json.dump({'update': update + 1, **metrics,
                               'best_distance': best_distance}, f, indent=2)

    except KeyboardInterrupt:
        print("\n⏹️ Interruption : sauvegarde du checkpoint...")
    finally:
        # Sauvegarde de sortie (meme si interrompu au milieu d'un update).
        final = run_dir / "last_model.pt"
        torch.save({
            'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
            'obs_rms': obs_rms.state_dict(),
            'obs_dim': obs_dim, 'act_dim': act_dim, 'hidden': s.HIDDEN_SIZE,
            'log_std_init': s.LOG_STD_INIT, 'norm_obs': s.NORM_OBS,
            'time_period': s.TIME_PERIOD, 'max_muscle_speed': s.MAX_MUSCLE_SPEED,
            'animal': animal_name, 'update': -1, 'best_distance': best_distance,
        }, final)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.load(final, map_location='cpu', weights_only=False), save_file)
        mlflow.log_artifact(str(final), artifact_path="models")
        mlflow.end_run()
        print(f"💾 Checkpoint : {save_file} | meilleur : {best_distance:.2f} m "
              f"(best_model.pt dans {run_dir})")


# ============ PLAYER POUR main.py (inference seule) ============

class IAPPOPlayer(IABase):
    """Joue une politique PPO entrainee dans la simulation fenetree.

    Inference deterministe (moyenne de la gaussienne). L'entrainement PPO se
    fait exclusivement via train.py --algo ppo (environnements vectorises).
    """

    def __init__(self, config):
        super().__init__(config)
        self.s = config.SETTINGS
        self.device = torch.device("cpu")  # inference : le CPU suffit largement
        self.act_dim = self.s.ACT_DIM
        self.obs_dim = obs_dim_for(self.act_dim)
        self.net = ActorCritic(self.obs_dim, self.act_dim,
                               self.s.HIDDEN_SIZE, self.s.LOG_STD_INIT)
        self.obs_rms = RunningMeanStd(self.obs_dim)
        self.norm_obs = self.s.NORM_OBS
        self.loaded = False

        import src.config as global_cfg
        self.ref_y = get_animal(global_cfg.ANIMAL).spawn_y

        # main.py borne les episodes avec current_max_time : en visualisation
        # on laisse courir (reset uniquement en cas de chute).
        self.current_max_time = 10 ** 9

    # ----- Interface IABase -----

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Aucun checkpoint PPO ({filepath}). "
                "Lancer d'abord : python train.py --algo ppo")
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        if ckpt.get('obs_dim') != self.obs_dim or ckpt.get('act_dim') != self.act_dim:
            raise ValueError(
                f"Checkpoint PPO incompatible (obs {ckpt.get('obs_dim')} vs "
                f"{self.obs_dim}, act {ckpt.get('act_dim')} vs {self.act_dim})")
        self.net.load_state_dict(ckpt['model'])
        self.net.eval()
        self.obs_rms.load_state_dict(ckpt['obs_rms'])
        self.norm_obs = bool(ckpt.get('norm_obs', True))
        self.loaded = True
        print(f"✅ Politique PPO chargée ({filepath}, "
              f"update {ckpt.get('update')}, animal {ckpt.get('animal')})")

    def get_action(self, time_s: float, dog_state: Dict[str, Any]) -> np.ndarray:
        obs = build_obs(time_s, dog_state, self.act_dim,
                        self.s.TIME_PERIOD, self.s.MAX_MUSCLE_SPEED, self.ref_y)
        if self.norm_obs:
            obs = self.obs_rms.normalize(obs[None])[0]  # stats FIGEES (pas d'update)
        with torch.no_grad():
            action = self.net.act_deterministic(
                torch.as_tensor(obs[None], dtype=torch.float32))
        return action.numpy()[0]

    def apply_to_quadruped(self, quadruped, action: np.ndarray):
        for i in range(self.act_dim):
            quadruped.set_muscle_activation(i, float(action[i]))

    def on_episode_end(self, distance: float, frames_survived: int,
                       dog_state: Dict[str, Any]):
        if distance > self.best_distance:
            self.best_distance = distance
        print(f"🏁 Episode PPO : {distance:.2f} m en {frames_survived} frames")

    def should_reset_simulation(self) -> bool:
        return True  # reset apres chaque chute

    def save(self, filepath: Optional[str] = None):
        # Rien a sauvegarder en inference (le checkpoint appartient a train.py).
        pass

    def hud_text(self) -> str:
        status = "politique chargée" if self.loaded else "ALEATOIRE (pas de checkpoint)"
        return f"PPO ({status}) | Best {self.best_distance:.2f}m"
