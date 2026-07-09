# ============================================
# display.py - Gestion de l'affichage Pygame
# ============================================

import pygame


class Display:
    """Gestionnaire de l'affichage avec Pygame"""

    def __init__(self, width=1200, height=700, title="Quadrupède"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.PPM = 100.0  # Pixels par mètre

        # Système de caméra
        self.camera_x = 0.0  # Offset horizontal de la caméra (en mètres)
        self.camera_y = 0.0  # Offset vertical de la caméra (en mètres)
        self.camera_speed = 0.1  # Vitesse de déplacement de la caméra
        self.follow_mode = True  # Mode suivi automatique du quadrupède
        self.follow_offset_y = 2  # Offset Y pour le suivi (négatif = quadrupède plus bas que le centre)

    def to_screen(self, pos):
        """Convertit les coordonnées Box2D en coordonnées Pygame avec offset caméra"""
        return (int((pos[0] - self.camera_x) * self.PPM),
                int(self.height - (pos[1] - self.camera_y) * self.PPM))

    def move_camera(self, dx, dy):
        """Déplace la caméra manuellement"""
        self.camera_x += dx
        self.camera_y += dy

    def follow_target(self, target_pos, smoothness=0.1):
        """Fait suivre la caméra à une position cible (suivi fluide)"""
        if self.follow_mode:
            # Interpolation linéaire pour un mouvement fluide
            # Utiliser follow_offset_y pour décentrer verticalement
            self.camera_x += (target_pos[0] - 6 - self.camera_x) * smoothness
            self.camera_y += (target_pos[1] - 3.5 + self.follow_offset_y - self.camera_y) * smoothness

    def toggle_follow_mode(self):
        """Active/désactive le mode suivi automatique"""
        self.follow_mode = not self.follow_mode
        return self.follow_mode

    def clear(self, color=(30, 30, 40)):
        """Efface l'écran avec un dégradé de ciel"""
        # Créer un dégradé de ciel bleu
        top_color = (135, 206, 235)  # Bleu ciel clair
        bottom_color = (200, 230, 255)  # Bleu très clair / blanc

        for y in range(self.height):
            progress = y / self.height
            color = (
                int(top_color[0] + (bottom_color[0] - top_color[0]) * progress),
                int(top_color[1] + (bottom_color[1] - top_color[1]) * progress),
                int(top_color[2] + (bottom_color[2] - top_color[2]) * progress)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

    def draw_ground(self, ground_body):
        """Dessine le sol"""
        vertices = [(ground_body.transform * v) * self.PPM
                    for v in ground_body.fixtures[0].shape.vertices]
        vertices = [(v[0] - self.camera_x * self.PPM, self.height - (v[1] - self.camera_y * self.PPM))
                    for v in vertices]
        pygame.draw.polygon(self.screen, (108, 163, 51), vertices)

    def draw_bone(self, bone, color=(255, 255, 255)):
        """Dessine un os (rectangle blanc)"""
        vertices = [(bone.body.transform * v) * self.PPM
                    for v in bone.fixture.shape.vertices]
        vertices = [(v[0] - self.camera_x * self.PPM, self.height - (v[1] - self.camera_y * self.PPM))
                    for v in vertices]
        pygame.draw.polygon(self.screen, color, vertices)

    def draw_muscle(self, muscle, color=(255, 0, 0)):
        """Dessine un muscle (ligne rouge)"""
        # Position des points d'ancrage
        pos_a = muscle.body_a.transform * muscle.anchor_a
        pos_b = muscle.body_b.transform * muscle.anchor_b

        # Convertir en coordonnées écran
        screen_a = self.to_screen(pos_a)
        screen_b = self.to_screen(pos_b)

        # Dessiner la ligne et les points
        pygame.draw.line(self.screen, color, screen_a, screen_b, 4)
        pygame.draw.circle(self.screen, (200, 0, 0), screen_a, 5)
        pygame.draw.circle(self.screen, (200, 0, 0), screen_b, 5)

    def draw_text(self, text, position, color=(255, 255, 255)):
        """Affiche du texte à l'écran"""
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, position)

    def draw_instructions(self):
        """Affiche les instructions de contrôle"""
        instructions = [
            "Contrôles des muscles :",
            "R/F : Muscle 1 | T/G : Muscle 2 | Y/H : Muscle 3",
            "E/D : Muscle 4 | Z/S : Muscle 5 | A/Q : Muscle 6",
            "─────────────────────────────",
            "Caméra: Flèches directionnelles",
            "F1: Mode suivi AUTO/MANUEL",
            "TAB: Changer mode visuel | ESC: Quitter"
        ]

        for i, text in enumerate(instructions):
            self.draw_text(text, (10, 30 + i * 25))

    def draw_camera_info(self):
        """Affiche les informations de la caméra"""
        mode_text = "SUIVI AUTO" if self.follow_mode else "MANUEL"
        color = (100, 255, 100) if self.follow_mode else (255, 150, 100)
        self.draw_text(f"Caméra: {mode_text}", (self.width - 200, 10), color)
        self.draw_text(f"X: {self.camera_x:.1f}m Y: {self.camera_y:.1f}m",
                       (self.width - 200, 35), (200, 200, 200))

    def update(self):
        """Rafraîchit l'affichage"""
        pygame.display.flip()

    def tick(self, fps=60):
        """Limite le framerate"""
        self.clock.tick(fps)

    def quit(self):
        """Ferme Pygame"""
        pygame.quit()