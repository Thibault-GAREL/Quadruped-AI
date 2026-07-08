"""Rejeu des champions de la neuroevolution.

Chaque generation d'entrainement archive son meilleur genome dans
`outputs/results/{run}/champions.pkl`. Comme la physique Box2D et le reseau
sont deterministes, rejouer un genome depuis la position de depart reproduit
EXACTEMENT le mouvement evalue pendant l'entrainement (pas besoin de logger
les trajectoires image par image).

Usage :
    python replay.py                      # dernier run trouve
    python replay.py <chemin champions.pkl / dossier de run>

Commandes en cours de rejeu :
    →  / ←   : generation suivante / precedente
    ↑        : sauter au meilleur champion (fitness max)
    HOME/END : premiere / derniere generation
    ESPACE   : rejouer depuis le debut
    F1       : basculer suivi camera
    ÉCHAP    : quitter
"""

import pickle
import sys
from pathlib import Path

import pygame

from src.core_engine.physics import PhysicsWorld, Quadruped
from src.core_engine.display import Display
from src.core_engine.overlay import VisualOverlay
from src.core_engine.parallax import ParallaxManager
from src.models.ia_gen import NeuroPolicy
from src.animals import get_animal
from src.config import ANIMAL

RESULTS_DIR = Path("outputs/results")
TARGET_FPS = 60
MAX_REPLAY_FRAMES = 3000  # garde-fou si le champion ne tombe jamais


def find_latest_champions() -> Path:
    """Retourne le champions.pkl le plus recent sous outputs/results/."""
    candidates = list(RESULTS_DIR.glob("*/champions.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun champions.pkl trouve sous {RESULTS_DIR}/. "
            "Lance d'abord un entrainement (main.py en mode neuro_ga)."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_path(arg: str) -> Path:
    """Accepte un fichier champions.pkl ou un dossier de run."""
    p = Path(arg)
    if p.is_dir():
        p = p / "champions.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Introuvable : {p}")
    return p


def load_archive(path: Path):
    with open(path, "rb") as f:
        archive = pickle.load(f)
    champions = archive.get("champions", [])
    if not champions:
        raise ValueError(f"Archive vide : {path}")
    nn_config = archive["nn_config"]
    return archive, champions, NeuroPolicy(nn_config)


def build_scene():
    """Cree un decor leger (fond + sol) pour le rejeu."""
    parallax = ParallaxManager()
    parallax.add_layer("assets/cloud.png", depth=0.07, x_position=-1, y_position=6,
                       repeat=True, repeat_spacing=(9, 12), scale=0.4)
    parallax.add_layer("assets/mountain2.png", depth=0.1, x_position=0, y_position=0,
                       repeat=True, repeat_spacing=(9, 12), scale=1.3)
    parallax.add_layer("assets/hill1.png", depth=0.15, x_position=-4, y_position=-0.16,
                       repeat=True, repeat_spacing=(6, 10), scale=1.4)
    parallax.add_layer("assets/tree3.png", depth=0.7, x_position=0, y_position=0,
                       repeat=True, repeat_spacing=(4, 10))
    parallax.add_layer("assets/bush.png", depth=0.92, x_position=-2, y_position=0.2,
                       repeat=True, repeat_spacing=(4, 6), scale=0.4)
    return parallax


def main():
    if len(sys.argv) > 1:
        path = resolve_path(sys.argv[1])
    else:
        path = find_latest_champions()

    archive, champions, policy = load_archive(path)
    print(f"📂 Archive : {path}")
    print(f"🧬 {len(champions)} champions | reseau "
          f"{policy.input_size}->{policy.hidden_size}->{policy.output_size} "
          f"(proprio={policy.use_proprioception})")

    # Index du champion courant (on demarre sur le meilleur).
    best_index = max(range(len(champions)), key=lambda i: champions[i]["fitness"])
    current = best_index

    # L'animal doit etre le meme que celui de l'entrainement rejoue
    # (ANIMAL dans src/config.py).
    animal = get_animal(ANIMAL)

    display = Display(width=1200, height=700, title=f"Rejeu des champions - {animal.name}")
    overlay = VisualOverlay(display, parts_folder="assets", global_scale=0.3, definition=animal)
    parallax = build_scene()
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

    def switch_to(new_index):
        """Change de champion et reinitialise la simulation."""
        nonlocal current, physics_world, quadruped, start_x, episode_time, frame
        current = max(0, min(len(champions) - 1, new_index))
        physics_world, quadruped, start_x = reset_world()
        episode_time = 0.0
        frame = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    switch_to(current + 1)
                elif event.key == pygame.K_LEFT:
                    switch_to(current - 1)
                elif event.key == pygame.K_UP:
                    switch_to(best_index)
                elif event.key == pygame.K_HOME:
                    switch_to(0)
                elif event.key == pygame.K_END:
                    switch_to(len(champions) - 1)
                elif event.key == pygame.K_SPACE:
                    switch_to(current)
                elif event.key == pygame.K_F1:
                    display.toggle_follow_mode()

        champion = champions[current]
        genome = champion["genome"]

        # ----- Inference + application (identique a l'entrainement) -----
        dog_state = {
            'position': (quadruped.body.body.position.x, quadruped.body.body.position.y),
            'velocity': (quadruped.body.body.linearVelocity.x, quadruped.body.body.linearVelocity.y),
            'angle': quadruped.body.body.angle,
            'muscle_angles': [m.get_angle() for m in quadruped.muscles],
            'muscle_speeds': [m.get_speed() for m in quadruped.muscles],
        }
        action = policy.act(episode_time, dog_state, genome)
        policy.apply(quadruped, action)

        quadruped.update()
        physics_world.step(time_step)
        episode_time += time_step
        frame += 1

        # Boucle : rejoue en continu (reset a la chute ou apres le garde-fou).
        if quadruped.is_upside_down() or frame >= MAX_REPLAY_FRAMES:
            switch_to(current)

        if display.follow_mode:
            display.follow_target(quadruped.body.body.position, smoothness=0.08)

        # ----- Rendu -----
        display.clear()
        parallax.draw_background(display)
        display.draw_ground(physics_world.ground)
        parallax.draw_foreground(display)
        overlay.draw_quadruped(quadruped)

        distance = quadruped.body.body.position.x - start_x
        info = (f"Champion {current + 1}/{len(champions)}  |  "
                f"Gen {champion['generation']}  |  "
                f"fitness {champion['fitness']:.1f}  |  "
                f"dist actuelle {distance:.2f} m"
                f"{'  ⭐ BEST' if current == best_index else ''}")
        surface = font.render(info, True, (255, 255, 255))
        bg = pygame.Surface((surface.get_width() + 12, surface.get_height() + 6))
        bg.fill((0, 0, 0))
        bg.set_alpha(160)
        display.screen.blit(bg, (5, 5))
        display.screen.blit(surface, (11, 8))
        display.draw_text("← → : generation | ↑ : best | ESPACE : rejouer | F1 : camera | ESC : quitter",
                          (10, display.height - 30), (200, 200, 200))

        display.update()
        display.tick(TARGET_FPS)

    display.quit()


if __name__ == "__main__":
    main()
