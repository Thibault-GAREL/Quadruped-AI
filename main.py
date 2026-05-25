import pygame
from src.core_engine.physics import PhysicsWorld, Quadruped
from src.core_engine.display import Display
from src.core_engine.overlay import VisualOverlay
from src.core_engine.parallax import ParallaxManager

# ===== IMPORTS POUR L'IA =====
import src.config as config_ia

# Dispatch dynamique selon IA_TYPE (defini dans src/config.py)
if config_ia.IA_TYPE == "choreography":
    from src.models.ia_chore import IAChoreography as IAClass
    import src.models.config_chore as ia_config
elif config_ia.IA_TYPE == "neuro_ga":
    from src.models.ia_gen import IAGenetic as IAClass
    import src.models.config_gen as ia_config
else:
    raise ValueError(
        f"IA_TYPE inconnu : {config_ia.IA_TYPE!r}. "
        "Valeurs supportees : 'choreography', 'neuro_ga'."
    )


def main():
    HUMAN_CONTROL = config_ia.HUMAN_CONTROL  # Pour garder la compatibilité
    DISPLAY_ENABLED = config_ia.DISPLAY_ENABLED  # Pour garder la compatibilité

    # Initialiser les systèmes
    physics_world = PhysicsWorld(gravity=(0, -10))

    # ===== GESTION DE L'AFFICHAGE =====
    display = Display(width=1200, height=700, title="Quadrupède muscles - Chat texture")

    quadruped = Quadruped(physics_world, x=6, y=3)

    # Initialiser le système d'overlay visuel avec l'image du chat
    overlay = VisualOverlay(display, parts_folder="assets", global_scale=0.3)

    # Initialiser le système de parallaxe
    parallax = ParallaxManager()

    episode_time = 0.0
    episode_start_x = quadruped.body.body.position.x

    # Ajouter des couches d'arrière-plan
    parallax.add_layer("assets/cloud.png", depth=0.07, x_position=-1, y_position=6, repeat=True, repeat_spacing=(9, 12),
                       scale=0.4)
    parallax.add_layer("assets/cloud2.png", depth=0.05, x_position=5, y_position=5, repeat=True, repeat_spacing=(5, 7),
                       scale=0.3)

    parallax.add_layer("assets/mountain2.png", depth=0.1, x_position=0, y_position=0, repeat=True, repeat_spacing=(9, 12),
                       scale=1.3)

    parallax.add_layer("assets/hill1.png", depth=0.15, x_position=-4, y_position=-0.16, repeat=True,
                       repeat_spacing=(6, 10), scale=1.4)
    parallax.add_layer("assets/hill2.png", depth=0.14, x_position=15, y_position=-0.16, repeat=True,
                       repeat_spacing=(5, 10))
    parallax.add_layer("assets/hill3.png", depth=0.19, x_position=-15, y_position=-0.16, repeat=True,
                       repeat_spacing=(4, 8))
    parallax.add_layer("assets/hill4.png", depth=0.23, x_position=8, y_position=-0.16, repeat=True, repeat_spacing=(6, 8))
    parallax.add_layer("assets/hill1.png", depth=0.15, x_position=-6, y_position=-0.16, repeat=True,
                       repeat_spacing=(6, 10))
    parallax.add_layer("assets/hill2.png", depth=0.14, x_position=20, y_position=-0.16, repeat=True,
                       repeat_spacing=(5, 10))
    parallax.add_layer("assets/hill3.png", depth=0.19, x_position=-19, y_position=-0.16, repeat=True,
                       repeat_spacing=(5, 6))
    parallax.add_layer("assets/hill4.png", depth=0.23, x_position=14, y_position=-0.16, repeat=True, repeat_spacing=(6, 8))

    parallax.add_layer("assets/tree3.png", depth=0.7, x_position=0, y_position=0, repeat=True, repeat_spacing=(4, 10))
    parallax.add_layer("assets/tree3.png", depth=0.7, x_position=3, y_position=0, repeat=True, repeat_spacing=(4, 10),
                       scale=1.1)

    parallax.add_layer("assets/tree4.png", depth=0.6, x_position=-2, y_position=0, repeat=True, repeat_spacing=(4, 10),
                       scale=0.9)
    parallax.add_layer("assets/tree5.png", depth=0.5, x_position=-6, y_position=0, repeat=True, repeat_spacing=(4, 10))

    parallax.add_layer("assets/bush.png", depth=0.8, x_position=-2, y_position=0.35, repeat=True, repeat_spacing=(4, 10),
                       scale=0.25)
    parallax.add_layer("assets/bush2.png", depth=0.83, x_position=-5, y_position=0.35, repeat=True, repeat_spacing=(4, 10),
                       scale=0.15)
    parallax.add_layer("assets/bush3.png", depth=0.84, x_position=-7, y_position=0.34, repeat=True, repeat_spacing=(4, 10),
                       scale=0.2)
    parallax.add_layer("assets/bush4.png", depth=0.87, x_position=-9, y_position=0.33, repeat=True, repeat_spacing=(4, 10),
                       scale=0.1)

    parallax.add_layer("assets/bush.png", depth=0.92, x_position=-2, y_position=0.2, repeat=True, repeat_spacing=(4, 6),
                       scale=0.4)
    parallax.add_layer("assets/bush2.png", depth=0.93, x_position=-5, y_position=0.15, repeat=True, repeat_spacing=(4, 7),
                       scale=0.45)
    parallax.add_layer("assets/bush3.png", depth=0.97, x_position=-7, y_position=0.1, repeat=True, repeat_spacing=(3, 8),
                       scale=0.2)
    parallax.add_layer("assets/bush4.png", depth=0.99, x_position=-9, y_position=0.1, repeat=True, repeat_spacing=(2, 9),
                       scale=0.3)

    print("💡 Pour ajouter des arrière-plans, décommentez les lignes add_layer() dans main.py")
    print("   - depth: 0.0=lointain, 1.0=proche")
    print("   - x_position: point de départ horizontal (0=centre)")
    print("   - y_position: position verticale en mètres (0=sol)")
    print("   - repeat: True=répète, False=image unique")
    print("   - repeat_spacing: None=collé, nombre=fixe, (min,max)=aléatoire")

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
    else:
        print("👤 Mode CONTRÔLE HUMAIN")

    # Paramètres de simulation
    TARGET_FPS = 60
    BASE_TIME_STEP = 1.0 / TARGET_FPS

    if display_active:
        TIME_STEP = BASE_TIME_STEP
    else:
        TIME_STEP = BASE_TIME_STEP * config_ia.CONFIG['speed_multiplier']

    # Boucle principale
    running = True
    frame_count = 0
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
                # Basculer l'affichage avec F2
                elif event.key == pygame.K_F2:
                    display_active = not display_active

                    # ✅ AJOUT : Recalculer TIME_STEP
                    if display_active:
                        TIME_STEP = BASE_TIME_STEP
                        print("🖥️ AFFICHAGE ACTIVÉ")
                    else:
                        TIME_STEP = BASE_TIME_STEP * config_ia.CONFIG['speed_multiplier']
                        print(f"⚡ AFFICHAGE DÉSACTIVÉ (mode rapide x{config_ia.CONFIG['speed_multiplier']})")
                # Sauvegarder manuellement avec S
                elif event.key == pygame.K_s and not HUMAN_CONTROL:
                    ia.save(ia_config.TRAINING_CONFIG['save_file'])
                    print(f"💾 Sauvegarde manuelle effectuée")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    # Augmenter la vitesse
                    config_ia.CONFIG['speed_multiplier'] = min(config_ia.CONFIG['speed_multiplier'] + 5, 100.0)
                    if not display_active:
                        TIME_STEP = BASE_TIME_STEP * config_ia.CONFIG['speed_multiplier']
                    print(f"⚡ Vitesse: x{config_ia.CONFIG['speed_multiplier']}")

                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    # Diminuer la vitesse
                    config_ia.CONFIG['speed_multiplier'] = max(config_ia.CONFIG['speed_multiplier'] - 5, 1.0)
                    if not display_active:
                        TIME_STEP = BASE_TIME_STEP * config_ia.CONFIG['speed_multiplier']
                    print(f"🐌 Vitesse: x{config_ia.CONFIG['speed_multiplier']}")

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
        # Relâcher tous les muscles
        for i in range(8):
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
                'angle': quadruped.body.body.angle
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

        # ===== ÉVALUATION DE L'IA =====
        if not HUMAN_CONTROL:
            current_x = quadruped.body.body.position.x
            is_fallen = quadruped.is_upside_down()

            max_time_frames = ia.current_max_time

            if is_fallen or frame_count >= max_time_frames:
                distance = current_x - episode_start_x
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
                    quadruped = Quadruped(physics_world, x=6, y=3)

                    episode_time = 0.0
                    episode_start_x = quadruped.body.body.position.x
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

            # 3. Dessiner les premier plans parallaxe (depth >= 0.9) - DEVANT le sol
            parallax.draw_foreground(display)

            # 4. Dessiner le quadrupède
            overlay.draw_quadruped(quadruped)

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
        else:
            # ✅ En mode rapide, on ne limite pas le FPS
            # Le TIME_STEP fait déjà tout le travail
            display.tick(10000)  # Limite très haute = pas de limite

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