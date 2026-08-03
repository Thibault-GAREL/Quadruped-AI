"""Visualisation d'un modele DEJA ENTRAINE (neuroevolution ou PPO).

C'est le point d'entree pour REGARDER un resultat, quel que soit l'algorithme
qui l'a produit. L'entrainement, lui, se fait avec train.py (headless) ou
main.py (fenetre, GA et choregraphie uniquement).

Deux sources sont reconnues, detectees automatiquement :

- **GA** : `outputs/results/{run}/champions.pkl`. Chaque generation y archive
  son meilleur genome, on peut donc naviguer dans l'historique et voir la
  demarche se construire. La physique Box2D et le reseau etant deterministes,
  rejouer un genome reproduit EXACTEMENT le mouvement evalue pendant
  l'entrainement (pas besoin de logger les trajectoires image par image).
- **PPO** : `outputs/models/{animal}_ppo.pt` ou le `best_model.pt` d'un run.
  Il n'y a qu'une politique finale, donc pas de navigation par generation.
  L'inference est deterministe (moyenne de la gaussienne, sans exploration).

Usage :
    python replay.py                                   # dernier modele trouve
    python replay.py outputs/results/neuro-ga_run-23_date-2026-08-03
    python replay.py outputs/models/fox_ppo.pt

Commandes en cours de rejeu :
    ESPACE   : rejouer depuis le debut
    F1       : basculer suivi camera
    ÉCHAP    : quitter
    → / ←    : generation suivante / precedente        (GA seulement)
    ↑        : sauter au meilleur champion             (GA seulement)
    HOME/END : premiere / derniere generation          (GA seulement)

L'animal rejoue est celui de ANIMAL dans src/config.py : il doit correspondre
a celui qui a ete entraine, sinon le genome ne colle pas au squelette.
"""

import pickle
import sys
from pathlib import Path

import pygame

from src.core_engine.physics import PhysicsWorld, Quadruped
from src.core_engine.display import Display
from src.core_engine.overlay import VisualOverlay
from src.core_engine.parallax import build_default_scene
from src.models.ia_gen import NeuroPolicy
from src.animals import get_animal
from src.config import ANIMAL

RESULTS_DIR = Path("outputs/results")
MODELS_DIR = Path("outputs/models")
TARGET_FPS = 60
MAX_REPLAY_FRAMES = 3000  # garde-fou si l'animal ne tombe jamais


# ============ SOURCES D'ACTION ============
# Deux facons de piloter l'animal, meme interface : le reste du script
# (physique, rendu, camera) est rigoureusement identique dans les deux cas.

class ChampionsSource:
    """Champions du GA : une politique par generation, navigables."""

    navigable = True

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            archive = pickle.load(f)
        self.champions = archive.get("champions", [])
        if not self.champions:
            raise ValueError(f"Archive vide : {path}")
        self.policy = NeuroPolicy(archive["nn_config"])
        self.best_index = max(range(len(self.champions)),
                              key=lambda i: self.champions[i]["fitness"])
        self.current = self.best_index

    def describe(self) -> str:
        p = self.policy
        return (f"🧬 {len(self.champions)} champions | reseau "
                f"{p.input_size}->{p.hidden_size}->{p.output_size} "
                f"(proprio={p.use_proprioception})")

    def act(self, episode_time: float, dog_state: dict, quadruped):
        genome = self.champions[self.current]["genome"]
        action = self.policy.act(episode_time, dog_state, genome)
        self.policy.apply(quadruped, action)

    def hud(self, distance: float) -> str:
        champion = self.champions[self.current]
        star = "  ⭐ BEST" if self.current == self.best_index else ""
        return (f"Champion {self.current + 1}/{len(self.champions)}  |  "
                f"Gen {champion['generation']}  |  "
                f"fitness {champion['fitness']:.1f}  |  "
                f"dist actuelle {distance:.2f} m{star}")

    def keys_help(self) -> str:
        return "← → : generation | ↑ : best | ESPACE : rejouer | F1 : camera | ESC : quitter"

    def select(self, index: int):
        """Change de champion. Retourne True si l'index a bouge."""
        new = max(0, min(len(self.champions) - 1, index))
        changed = new != self.current
        self.current = new
        return changed


class PpoSource:
    """Politique PPO entrainee : une seule politique, pas de navigation."""

    navigable = False

    def __init__(self, path: Path):
        # Import local : torch n'est necessaire que pour PPO.
        try:
            from src.models.ia_ppo import IAPPOPlayer
        except ImportError as e:
            raise SystemExit(
                f"PPO indisponible ({e}).\n"
                "Installer PyTorch dans le venv :\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        import torch
        import src.models.config_ppo as cfg

        # Le checkpoint sait pour quel animal il a ete entraine. Le verifier
        # AVANT de charger evite une erreur de dimensions incomprehensible
        # ("obs 19 vs 23") quand ANIMAL ne correspond pas au modele demande.
        trained_on = torch.load(str(path), map_location="cpu",
                                weights_only=False).get("animal")
        if trained_on and trained_on != ANIMAL:
            raise SystemExit(
                f"Ce modele a ete entraine sur \"{trained_on}\", "
                f"or ANIMAL vaut \"{ANIMAL}\" dans src/config.py.\n"
                f"Corrige la ligne : ANIMAL = \"{trained_on}\""
            )

        self.player = IAPPOPlayer(cfg)
        self.player.load(str(path))
        if not self.player.loaded:
            raise SystemExit(f"Politique PPO illisible : {path}")

    def describe(self) -> str:
        p = self.player
        return (f"🤖 politique PPO | reseau "
                f"{p.obs_dim}->{p.s.HIDDEN_SIZE}->{p.act_dim} (inference deterministe)")

    def act(self, episode_time: float, dog_state: dict, quadruped):
        action = self.player.get_action(episode_time, dog_state)
        self.player.apply_to_quadruped(quadruped, action)

    def hud(self, distance: float) -> str:
        return f"PPO  |  dist actuelle {distance:.2f} m"

    def keys_help(self) -> str:
        return "ESPACE : rejouer | F1 : camera | ESC : quitter"


# ============ RESOLUTION DU CHEMIN ============

def find_latest_model() -> Path:
    """Retourne le modele le plus recent, GA ou PPO confondus."""
    candidates = (list(RESULTS_DIR.glob("*/champions.pkl"))
                  + list(MODELS_DIR.glob("*_ppo.pt"))
                  + list(MODELS_DIR.glob("*/best_model.pt")))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun modele trouve sous {RESULTS_DIR}/ ni {MODELS_DIR}/.\n"
            "Lance d'abord un entrainement : python train.py --algo ga (ou ppo)."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_path(arg: str) -> Path:
    """Accepte un champions.pkl, un .pt PPO, ou un dossier de run."""
    p = Path(arg)
    if p.is_dir():
        # Un dossier de run contient l'un ou l'autre selon l'algorithme.
        for candidate in (p / "champions.pkl", p / "best_model.pt", p / "last_model.pt"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Ni champions.pkl ni best_model.pt dans {p}")
    if not p.exists():
        raise FileNotFoundError(f"Introuvable : {p}")
    return p


def build_source(path: Path):
    """Instancie la bonne source selon l'extension du fichier."""
    if path.suffix == ".pt":
        return PpoSource(path)
    return ChampionsSource(path)


# ============ BOUCLE DE VISUALISATION ============

def main():
    path = resolve_path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_model()
    source = build_source(path)

    print(f"📂 Modele : {path}")
    print(f"   {source.describe()}")
    print(f"🐾 Animal : {ANIMAL}")

    animal = get_animal(ANIMAL)
    display = Display(width=1200, height=700, title=f"Rejeu - {animal.name}")
    overlay = VisualOverlay(display, parts_folder="assets", global_scale=0.3, definition=animal)
    parallax = build_default_scene()
    display.follow_mode = True

    def reset_world():
        world = PhysicsWorld(gravity=(0, -10))
        quad = Quadruped(world, x=6, y=animal.spawn_y, definition=animal)
        return world, quad, quad.body.body.position.x

    physics_world, quadruped, start_x = reset_world()
    episode_time = 0.0
    frame = 0
    time_step = 1.0 / TARGET_FPS
    font = pygame.font.Font(None, 28)

    def restart():
        """Relance l'episode depuis le debut (meme politique)."""
        nonlocal physics_world, quadruped, start_x, episode_time, frame
        physics_world, quadruped, start_x = reset_world()
        episode_time = 0.0
        frame = 0

    def go_to(index: int):
        """Navigation entre champions (GA uniquement)."""
        if source.navigable:
            source.select(index)
            restart()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    restart()
                elif event.key == pygame.K_F1:
                    display.toggle_follow_mode()
                elif source.navigable:
                    if event.key == pygame.K_RIGHT:
                        go_to(source.current + 1)
                    elif event.key == pygame.K_LEFT:
                        go_to(source.current - 1)
                    elif event.key == pygame.K_UP:
                        go_to(source.best_index)
                    elif event.key == pygame.K_HOME:
                        go_to(0)
                    elif event.key == pygame.K_END:
                        go_to(len(source.champions) - 1)

        # ----- Inference + application (identique a l'entrainement) -----
        dog_state = {
            'position': (quadruped.body.body.position.x, quadruped.body.body.position.y),
            'velocity': (quadruped.body.body.linearVelocity.x, quadruped.body.body.linearVelocity.y),
            'angle': quadruped.body.body.angle,
            'muscle_angles': [m.get_angle() for m in quadruped.muscles],
            'muscle_speeds': [m.get_speed() for m in quadruped.muscles],
        }
        source.act(episode_time, dog_state, quadruped)

        quadruped.update()
        physics_world.step(time_step)
        episode_time += time_step
        frame += 1

        # Boucle : rejoue en continu (reset a la chute ou apres le garde-fou).
        if quadruped.is_upside_down() or frame >= MAX_REPLAY_FRAMES:
            restart()

        if display.follow_mode:
            display.follow_target(quadruped.body.body.position, smoothness=0.08)

        # ----- Rendu -----
        display.clear()
        parallax.draw_background(display)
        display.draw_ground(physics_world.ground)
        overlay.draw_quadruped(quadruped)
        # Premier plan dessine en dernier : l'animal passe derriere les buissons proches.
        parallax.draw_foreground(display)

        distance = quadruped.body.body.position.x - start_x
        surface = font.render(source.hud(distance), True, (255, 255, 255))
        bg = pygame.Surface((surface.get_width() + 12, surface.get_height() + 6))
        bg.fill((0, 0, 0))
        bg.set_alpha(160)
        display.screen.blit(bg, (5, 5))
        display.screen.blit(surface, (11, 8))
        display.draw_text(source.keys_help(), (10, display.height - 30), (200, 200, 200))

        display.update()
        display.tick(TARGET_FPS)

    display.quit()


if __name__ == "__main__":
    main()
