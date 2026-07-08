"""Politique de la neuroevolution : MLP NumPy + construction des entrees.

Module volontairement LEGER (numpy uniquement, pas de mlflow/pandas/pygame) :
il est importe par les workers multiprocessing de train.py, qui doivent
demarrer vite. IAGenetic (ia_gen.py) et replay.py l'importent aussi.
"""

from typing import Any, Dict, List

import numpy as np


class MLP:
    """Reseau de neurones feedforward minimal en NumPy.

    Le genome est un vecteur 1D contenant a la suite W1, b1, W2, b2.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self._w1_size = input_size * hidden_size
        self._b1_size = hidden_size
        self._w2_size = hidden_size * output_size
        self._b2_size = output_size

        self.num_params = (
            self._w1_size + self._b1_size + self._w2_size + self._b2_size
        )

    def forward(self, x: np.ndarray, genome: np.ndarray) -> np.ndarray:
        idx = 0
        w1 = genome[idx:idx + self._w1_size].reshape(self.input_size, self.hidden_size)
        idx += self._w1_size
        b1 = genome[idx:idx + self._b1_size]
        idx += self._b1_size
        w2 = genome[idx:idx + self._w2_size].reshape(self.hidden_size, self.output_size)
        idx += self._w2_size
        b2 = genome[idx:idx + self._b2_size]

        h = np.tanh(x @ w1 + b1)
        out = np.tanh(h @ w2 + b2)
        return out


class NeuroPolicy:
    """Politique reactive : construction des entrees + MLP + application continue.

    Cette classe est PARTAGEE entre l'entrainement (IAGenetic, train.py) et le
    rejeu (replay.py). Ainsi, le mouvement rejoue est rigoureusement identique
    a celui evalue pendant l'entrainement (memes entrees, meme reseau).

    `nn_config` doit contenir :
        input_size (base), hidden_size, output_size, time_period,
        use_proprioception (bool), max_muscle_speed (float).
    """

    def __init__(self, nn_config: Dict[str, Any]):
        self.base_input = int(nn_config['input_size'])
        self.hidden_size = int(nn_config['hidden_size'])
        self.output_size = int(nn_config['output_size'])
        self.time_period = float(nn_config['time_period'])
        self.use_proprioception = bool(nn_config.get('use_proprioception', True))
        self.max_muscle_speed = float(nn_config.get('max_muscle_speed', 3.0))

        # Proprioception : angle + vitesse de chaque muscle actionne.
        proprio_dim = 2 * self.output_size if self.use_proprioception else 0
        self.input_size = self.base_input + proprio_dim

        self.mlp = MLP(self.input_size, self.hidden_size, self.output_size)
        self.num_params = self.mlp.num_params

    def build_input(self, time: float, dog_state: Dict[str, Any]) -> np.ndarray:
        """Construit le vecteur d'entree normalise a partir de l'etat."""
        _, y = dog_state['position']
        vx, vy = dog_state['velocity']
        angle = dog_state['angle']

        phase = 2.0 * np.pi * time / self.time_period
        feats: List[float] = [
            np.sin(phase),
            np.cos(phase),
            np.clip(vx / 5.0, -1.0, 1.0),
            np.clip(vy / 5.0, -1.0, 1.0),
            np.sin(angle),
            np.cos(angle),
            np.clip((y - 3.0) / 2.0, -1.0, 1.0),
        ]

        if self.use_proprioception:
            m_ang = dog_state.get('muscle_angles', [])
            m_spd = dog_state.get('muscle_speeds', [])
            for i in range(self.output_size):
                a = m_ang[i] if i < len(m_ang) else 0.0
                s = m_spd[i] if i < len(m_spd) else 0.0
                feats.append(float(np.clip(a / np.pi, -1.0, 1.0)))
                feats.append(float(np.clip(s / self.max_muscle_speed, -1.0, 1.0)))

        return np.array(feats, dtype=np.float32)

    def act(self, time: float, dog_state: Dict[str, Any], genome: np.ndarray) -> np.ndarray:
        """Retourne les activations continues dans [-1, 1]."""
        return self.mlp.forward(self.build_input(time, dog_state), genome)

    def apply(self, quadruped, action: np.ndarray) -> None:
        """Applique les activations continues aux muscles actionnes."""
        for i in range(self.output_size):
            quadruped.set_muscle_activation(i, float(action[i]))
