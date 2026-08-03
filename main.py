import math
import pickle
import time

import pygame
from src.core_engine.physics import PhysicsWorld, Quadruped
from src.core_engine.display import Display
from src.core_engine.overlay import VisualOverlay
from src.core_engine.parallax import build_default_scene
from src.animals import get_animal

# ===== IMPORTS POUR L'IA =====
import src.config as config_ia

# Dispatch dynamique selon IA_TYPE (defini dans src/config.py)
if config_ia.IA_TYPE == "choreography":
    from src.models.ia_chore import IAChoreography as IAClass
    import src.models.config_chore as ia_config
elif config_ia.IA_TYPE == "neuro_ga":
    from src.models.ia_gen import IAGenetic as IAClass
    import src.models.config_gen as ia_config
elif config_ia.IA_TYPE == "ppo":
    # PPO ne s'entraine JAMAIS ici : sa boucle est vectorisee et headless
    # (train.py --algo ppo). main.py sert a controler l'animal au clavier et a
    # entrainer les algorithmes qui tournent episode par episode dans la
    # fenetre, ce qui n'est pas le cas de PPO. La visualisation d'une politique
    # entrainee est le role de replay.py.
    raise SystemExit(
        "IA_TYPE = \"ppo\" n'est pas gere par main.py.\n"
        "  Pour ENTRAINER   : python train.py --algo ppo\n"
        "  Pour VISUALISER  : python replay.py outputs/models/"
        f"{config_ia.ANIMAL}_ppo.pt\n"
        "main.py sert au controle clavier et a l'entrainement fenetre "
        "(neuro_ga, choreography)."
    )
else:
    raise ValueError(
        f"IA_TYPE inconnu : {config_ia.IA_TYPE!r}. "
        "Valeurs supportees : 'choreography', 'neuro_ga', 'ppo'."
    )


def main():
    HUMAN_CONTROL = config_ia.HUMAN_CONTROL  # Pour garder la compatibilité
    DISPLAY_ENABLED = config_ia.DISPLAY_ENABLED  # Pour garder la compatibilité

    # ===== ANIMAL SÉLECTIONNÉ (src/config.py -> ANIMAL) =====
    animal = get_animal(config_ia.ANIMAL)
    print(f"🐾 Animal : {animal.name} ({animal.num_actuated} muscles actionnés)")

    # Initialiser les systèmes
    physics_world = PhysicsWorld(gravity=(0, -10))

    # ===== GESTION DE L'AFFICHAGE =====
    display = Display(width=1200, height=700, title=f"Simulation muscles - {animal.name}")

    quadruped = Quadruped(physics_world, x=6, y=animal.spawn_y, definition=animal)

    # Initialiser le système d'overlay visuel (peau procédurale + modes debug)
    overlay = VisualOverlay(display, parts_folder="assets", global_scale=0.3, definition=animal)

    # Initialiser le système de parallaxe (décor partagé avec replay.py)
    parallax = build_default_scene()

    episode_time = 0.0
    episode_start_x = quadruped.body.body.position.x
    # Somme de cos(angle du corps) sur l'episode -> moyenne = stabilite (uprightness).
    episode_uprightness_sum = 0.0
    # Frames passees au sol sur autre chose que les pieds (shaped reward).
    episode_bad_contact_frames = 0

    print("\n🔍 Appuyez sur P pour afficher les angles des os")

    # Afficher le mode d'affichage
    if DISPLAY_ENABLED:
        print("🖥️ Mode AFFICHAGE ACTIVÉ")
    else:
        print("⚡ Mode RAPIDE (sans affichage)")

    print("\n💡 Appuyez sur F2 pour basculer l'affichage pendant l'exécution")

    # Variable pour gérer l'affichage dynamique
    display_active = DISPLAY_ENABLED

    # ===== INITIALISATION DE L'IA =====
    if not HUMAN_CONTROL:
        print(f"🤖 Mode IA ACTIVÉ (type={config_ia.IA_TYPE})")
        ia = IAClass(ia_config)
        # Essayer de charger une sauvegarde existante
        try:
            ia.load(ia_config.TRAINING_CONFIG['save_file'])
            print(f"   IA chargée: Génération {ia.generation}")
        except FileNotFoundError:
            print("   Nouvelle IA créée")
        except (ValueError, KeyError, AssertionError, pickle.UnpicklingError) as e:
            # Checkpoint present mais incompatible (ex: architecture modifiee).
            print(f"   Checkpoint incompatible ({e}). Nouvelle IA créée")
    else:
        print("👤 Mode CONTRÔLE HUMAIN")

    # Paramètres de simulation
    # IMPORTANT : le pas de physique est TOUJOURS 1/60 s. L'ancien mode rapide
    # multipliait TIME_STEP par 50, ce qui faussait complètement la dynamique
    # Box2D (un dt géant ne simule pas la même physique). Désormais le mode
    # rapide (F2) coupe simplement le rendu et la limite de FPS : la boucle
    # tourne à la vitesse du CPU avec une physique fidèle.
    # Pour les gros entraînements, utiliser train.py (headless + parallèle).
    TARGET_FPS = 60
    TIME_STEP = 1.0 / TARGET_FPS

    # Boucle principale
    running = True
    frame_count = 0
    last_debug_time = time.time()  # pour le print de pose toutes les 5s
    while running:
        frame_count += 1
        episode_time += TIME_STEP

        # Gestion des événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Basculer entre les 3 modes avec TAB
                elif event.key == pygame.K_TAB:
                    overlay.toggle_mode()
                # Basculer le mode suivi de caméra avec F1
                elif event.key == pygame.K_F1:
                    follow = display.toggle_follow_mode()
                    print(f"📷 Mode caméra: {'SUIVI AUTO' if follow else 'MANUEL'}")
                # Basculer l'affichage avec F2 (la physique reste a dt = 1/60 :
                # le mode rapide coupe juste le rendu et la limite de FPS)
                elif event.key == pygame.K_F2:
                    display_active = not display_active
                    if display_active:
                        print("🖥️ AFFICHAGE ACTIVÉ")
                    else:
                        print("⚡ AFFICHAGE DÉSACTIVÉ (boucle à vitesse CPU, physique fidèle)")
                        print("   💡 Pour un vrai gros entraînement : python train.py (parallèle)")
                # Sauvegarder manuellement avec S
                elif event.key == pygame.K_s and not HUMAN_CONTROL:
                    ia.save(ia_config.TRAINING_CONFIG['save_file'])
                    print(f"💾 Sauvegarde manuelle effectuée")

        # Gestion des touches (contrôle manuel)
        keys = pygame.key.get_pressed()

        # ===== CONTRÔLES CAMÉRA (flèches directionnelles) =====
        if not display.follow_mode:  # Seulement en mode manuel
            if keys[pygame.K_LEFT]:
                display.move_camera(-display.camera_speed, 0)
            if keys[pygame.K_RIGHT]:
                display.move_camera(display.camera_speed, 0)
            if keys[pygame.K_UP]:
                display.move_camera(0, display.camera_speed)
            if keys[pygame.K_DOWN]:
                display.move_camera(0, -display.camera_speed)

        # ===== CONTRÔLE DES MUSCLES =====
        # Relâcher tous les muscles actionnés
        for i in range(quadruped.num_actuated):
            quadruped.control_muscles(i, 'relax')

        if HUMAN_CONTROL:
            # ===== MODE HUMAIN: Contrôle par clavier =====
            if keys[pygame.K_t]:
                quadruped.control_muscles(0, 'contract')
            elif keys[pygame.K_g]:
                quadruped.control_muscles(0, 'extend')

            if keys[pygame.K_y]:
                quadruped.control_muscles(1, 'contract')
            elif keys[pygame.K_h]:
                quadruped.control_muscles(1, 'extend')

            if keys[pygame.K_u]:
                quadruped.control_muscles(2, 'contract')
            elif keys[pygame.K_j]:
                quadruped.control_muscles(2, 'extend')

            if keys[pygame.K_i]:
                quadruped.control_muscles(3, 'contract')
            elif keys[pygame.K_k]:
                quadruped.control_muscles(3, 'extend')

            if keys[pygame.K_r]:
                quadruped.control_muscles(4, 'contract')
            elif keys[pygame.K_f]:
                quadruped.control_muscles(4, 'extend')

            if keys[pygame.K_e]:
                quadruped.control_muscles(5, 'contract')
            elif keys[pygame.K_d]:
                quadruped.control_muscles(5, 'extend')

            if keys[pygame.K_z]:
                quadruped.control_muscles(6, 'contract')
            elif keys[pygame.K_s]:
                quadruped.control_muscles(6, 'extend')

            if keys[pygame.K_a]:
                quadruped.control_muscles(7, 'contract')
            elif keys[pygame.K_q]:
                quadruped.control_muscles(7, 'extend')
        else:
            # ===== MODE IA: Contrôle automatique =====
            dog_state = {
                'position': (quadruped.body.body.position.x, quadruped.body.body.position.y),
                'velocity': (quadruped.body.body.linearVelocity.x, quadruped.body.body.linearVelocity.y),
                'angle': quadruped.body.body.angle,
                # Proprioception : angles et vitesses de tous les joints.
                # L'IA neuroevolution en consomme les premiers (muscles actionnes).
                'muscle_angles': [m.get_angle() for m in quadruped.muscles],
                'muscle_speeds': [m.get_speed() for m in quadruped.muscles],
            }

            # Interface unifiee : chaque IA decide comment appliquer son action.
            action = ia.get_action(episode_time, dog_state)
            ia.apply_to_quadruped(quadruped, action)

        if quadruped.is_upside_down():
            if display_active:
                display.draw_text("RETOURNÉ!", (display.width // 2 - 50, 50), (255, 0, 0))
                print("Retourné !!")

        # Mettre à jour la physique
        quadruped.update()
        physics_world.step(TIME_STEP)

        # Accumule la stabilite (cos de l'angle du corps) apres le pas physique.
        episode_uprightness_sum += math.cos(quadruped.body.body.angle)

        # Appui fautif : un os autre qu'un pied touche le sol (shaped reward).
        if config_ia.SHAPED_REWARD and quadruped.has_bad_ground_contact():
            episode_bad_contact_frames += 1

        # ===== DEBUG POSE : etat complet toutes les 5 secondes =====
        # Pour regler une pose d'equilibre : lire les angles des os et des joints.
        # now = time.time()
        # if now - last_debug_time >= 5.0:
        #     quadruped.debug_print(episode_time)
        #     last_debug_time = now

        # ===== ÉVALUATION DE L'IA =====
        if not HUMAN_CONTROL:
            current_x = quadruped.body.body.position.x
            is_fallen = quadruped.is_upside_down()

            max_time_frames = ia.current_max_time

            if is_fallen or frame_count >= max_time_frames:
                distance = current_x - episode_start_x
                # Transmet l'etat de chute a l'IA (fitness penalisee si tombe).
                dog_state['is_fallen'] = is_fallen
                # Moyenne de stabilite sur l'episode (pour le bonus optionnel).
                dog_state['uprightness'] = episode_uprightness_sum / max(frame_count, 1)
                # Frames en appui fautif (penalite du shaped reward).
                dog_state['bad_contact_frames'] = episode_bad_contact_frames
                ia.on_episode_end(distance, frame_count, dog_state)

                # Vérifier si on doit reset
                prev_generation = ia.generation
                if ia.should_reset_simulation():
                    # Log de génération seulement quand une génération vient de s'achever
                    # (should_reset_simulation est appelé à chaque fin d'épisode mais
                    # ia.generation n'est incrémenté qu'en fin de population complète).
                    if ia.generation != prev_generation:
                        ia.on_generation_end()

                    if ia.generation >= ia_config.TRAINING_CONFIG['max_generations']:
                        print(f"\n✅ Training terminé: {ia.generation} générations")
                        ia.save(ia_config.TRAINING_CONFIG['save_file'])

                        if ia_config.TRAINING_CONFIG['auto_continue']:
                            print(f"\n🔄 Démarrage du prochain training...")
                            ia.generation = 0
                            # Continue automatiquement (ne fait rien, la boucle continue)
                        else:
                            print(f"\n🛑 Arrêt (auto_continue désactivé)")
                            running = False

                    # Reset physique
                    del physics_world
                    del quadruped
                    physics_world = PhysicsWorld(gravity=(0, -10))
                    quadruped = Quadruped(physics_world, x=6, y=animal.spawn_y, definition=animal)

                    episode_time = 0.0
                    episode_start_x = quadruped.body.body.position.x
                    episode_uprightness_sum = 0.0
                    episode_bad_contact_frames = 0
                    frame_count = 0

                    ia.reset_episode()


        # ===== MISE À JOUR CAMÉRA =====
        # En mode suivi automatique, la caméra suit le corps du quadrupède
        if display.follow_mode:
            body_pos = quadruped.body.body.position
            display.follow_target(body_pos, smoothness=0.08)

        # ===== AFFICHAGE (seulement si display_active) =====
        if display_active:
            display.clear()

            # 1. Dessiner les arrière-plans parallaxe (depth < 0.9)
            parallax.draw_background(display)

            # 2. Dessiner le sol
            display.draw_ground(physics_world.ground)

            # 3. Dessiner le quadrupède
            overlay.draw_quadruped(quadruped)

            # 4. Dessiner les premier plans parallaxe (depth >= 0.9) - DEVANT le sol
            #    et DEVANT l'animal (l'animal passe derrière les buissons proches)
            parallax.draw_foreground(display)

            display.draw_instructions()
            display.draw_camera_info()
            overlay.draw_status()

            # Ajouter l'instruction pour afficher les angles
            display.draw_text("P: Afficher angles des os", (10, display.height - 105), (200, 200, 200))

            # Instruction pour basculer l'affichage
            display.draw_text("F2: Basculer affichage", (10, display.height - 135), (200, 200, 200))

            # Afficher les infos de l'IA (optionnel)
            if not HUMAN_CONTROL:
                font = pygame.font.Font(None, 24)
                stats = ia.get_stats()
                if hasattr(ia, 'hud_text'):
                    text = ia.hud_text()
                else:
                    text = f"Gen {stats['generation']} | Individu {stats['current_individual'] + 1}/{ia_config.GA_CONFIG['population_size']} | Best {stats['best_distance']:.2f}m"

                surface = font.render(text, True, (255, 255, 255))
                bg_surf = pygame.Surface((surface.get_width() + 10, surface.get_height() + 5))
                bg_surf.fill((0, 0, 0))
                bg_surf.set_alpha(150)
                display.screen.blit(bg_surf, (5, 5))
                display.screen.blit(surface, (10, 5))

            display.update()

        if display_active:
            display.tick(TARGET_FPS)
        # En mode rapide : aucune limite de FPS, la boucle tourne à fond
        # (physique inchangée, on simule juste plus de frames par seconde réelle).

    # Sauvegarde finale
    if not HUMAN_CONTROL:
        ia.save(ia_config.TRAINING_CONFIG['save_file'])
        print(f"\n💾 Sauvegarde finale effectuée")
        # Cloture propre du tracking (MLflow notamment).
        if hasattr(ia, "close"):
            ia.close()

    # Fermer proprement Pygame
    display.quit()


if __name__ == "__main__":
    main()