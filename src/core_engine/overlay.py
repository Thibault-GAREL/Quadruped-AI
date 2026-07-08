# ============================================
# overlay.py - Système d'overlay avec PARTIES DÉCOUPÉES
# ============================================

import pygame
import math
import os

from src.core_engine.procedural_skin import ProceduralSkin


class BoneTexture:
    """Charge et dessine une texture pour un os spécifique"""

    def __init__(self, image_path, scale=1.0):
        """Charge l'image pour cet os

        Args:
            image_path: Chemin vers l'image
            scale: Facteur d'échelle (1.0 = taille originale, 0.5 = moitié, etc.)
        """
        self.image = None
        self.loaded = False
        self.scale = scale

        if os.path.exists(image_path):
            try:
                original = pygame.image.load(image_path).convert_alpha()

                # Appliquer le scaling
                if scale != 1.0:
                    new_width = int(original.get_width() * scale)
                    new_height = int(original.get_height() * scale)
                    self.image = pygame.transform.scale(original, (new_width, new_height))
                else:
                    self.image = original

                self.loaded = True
                print(f"✅ Chargé : {os.path.basename(image_path)} (échelle: {scale}x)")
            except Exception as e:
                print(f"❌ Erreur : {image_path} - {e}")
        else:
            print(f"⚠️ Non trouvé : {image_path}")

    def draw(self, screen, bone, display, offset=(0, 0), rotation_offset=0):
        """Dessine la texture sur l'os avec la même rotation

        Args:
            screen: Surface pygame
            bone: Os du squelette
            display: Objet Display
            offset: Tuple (x, y) en pixels pour décaler l'image
            rotation_offset: Angle en DEGRÉS à ajouter à la rotation (positif = horaire)
        """
        if not self.loaded or self.image is None:
            return

        # Récupérer la position et l'angle de l'os
        bone_pos = bone.body.position
        bone_angle = bone.body.angle

        # Convertir l'angle de l'os en degrés et ajouter le rotation_offset
        # On utilise le même sens de rotation que le squelette
        final_angle_degrees = math.degrees(bone_angle) - rotation_offset

        # Rotation de l'image selon l'angle de l'os
        rotated_image = pygame.transform.rotate(self.image, final_angle_degrees)

        # Position à l'écran
        screen_pos = display.to_screen(bone_pos)

        # Appliquer l'offset (rotation de l'offset selon l'angle de l'os)
        offset_x, offset_y = offset
        if offset_x != 0 or offset_y != 0:
            # Rotation de l'offset selon l'angle de l'os
            cos_a = math.cos(bone_angle)
            sin_a = math.sin(bone_angle)
            rotated_offset_x = offset_x * cos_a - offset_y * sin_a
            rotated_offset_y = offset_x * sin_a + offset_y * cos_a

            screen_pos = (
                screen_pos[0] + rotated_offset_x,
                screen_pos[1] - rotated_offset_y  # Inverser Y car Pygame a Y vers le bas
            )

        # Centrer l'image sur la position (avec offset)
        rect = rotated_image.get_rect(center=screen_pos)

        # Dessiner
        screen.blit(rotated_image, rect.topleft)


class TexturedOverlay:
    """Système d'overlay utilisant des parties découpées"""

    def __init__(self, parts_folder="fox_parts", global_scale=1.0):
        """Initialise et charge toutes les textures

        Args:
            parts_folder: Dossier contenant les images
            global_scale: Échelle globale pour toutes les images (0.3 = 30% de la taille)
        """
        self.parts_folder = parts_folder
        self.textures = {}
        self.global_scale = global_scale

        # Facteurs d'échelle individuels par partie (multipliés par global_scale)
        self.part_scales = {
            'body': 1.3,
            'neck': 1.2,
            'head': 1.2,
            'front_thigh': 1.1,
            'front_shin': 1.5,
            'front_ankle': 1.2,
            'front_foot': 1.0,
            'back_thigh': 1.0,
            'back_shin': 1.3,
            'back_ankle': 1.3,
            'back_foot': 1.3,
            'tail_bottom': 1.3,
            'tail_mid': 1.3,
            'tail_high': 1.3,
        }

        # Décalages (offsets) en pixels pour ajuster la position de chaque image
        self.part_offsets = {
            'body': (-10, 0),
            'neck': (0, 20),
            'head': (0, -10),
            'front_thigh': (0, 0),
            'front_shin': (-5, 0),
            'front_ankle': (0, 0),
            'front_foot': (0, 0),
            'back_thigh': (0, 0),
            'back_shin': (0, 0),
            'back_ankle': (0, -5),
            'back_foot': (0, 0),
            'tail_bottom': (0, 0),
            'tail_mid': (0, 0),
            'tail_high': (0, -10),
        }

        # Rotation offsets en DEGRÉS pour ajuster l'angle de chaque image
        self.rotation_offsets = {
            'body': 0.24,
            'neck': 126.27,
            'head': -116,
            'front_thigh': -10,
            'front_shin': 3.45 - 10,
            'front_ankle': 14.14 - 20,
            'front_foot': 82.07,
            'back_thigh': -4.55,
            'back_shin': -6.57,
            'back_ankle': 0,
            'back_foot': 87.51,
            'tail_bottom': -107.67 + 95,
            'tail_mid': -160.63 + 130,
            'tail_high': 128.06 + 170,
        }

        # Ordre de dessin
        self.draw_order = [
            'tail_high',  # Queue (fond)
            'tail_mid',
            'tail_bottom',
            'back_foot',  # Patte arrière (fond)
            'back_ankle',
            'back_shin',
            'back_thigh',
            'front_foot',  # Patte avant
            'front_ankle',
            'front_shin',
            'front_thigh',
            'neck',  # Cou
            'head',  # Tête
            'body',  # Corps (premier plan)
        ]

        # Mapping nom -> fichier
        self.file_mapping = {
            'body': 'fox_texture_body.png',
            'neck': 'fox_texture_neck.png',
            'head': 'fox_texture_head.png',
            'front_thigh': 'fox_texture_front_thigh.png',
            'front_shin': 'fox_texture_front_shin.png',
            'front_ankle': 'fox_texture_front_anckle.png',
            'front_foot': 'fox_texture_front_foot.png',
            'back_thigh': 'fox_texture_back_thigh.png',
            'back_shin': 'fox_texture_back_shin.png',
            'back_ankle': 'fox_texture_back_anckle.png',
            'back_foot': 'fox_texture_back_foot.png',
            'tail_bottom': 'fox_texture_tail_bottom.png',
            'tail_mid': 'fox_texture_tail_mid.png',
            'tail_high': 'fox_texture_tail_high.png',
        }

        self.load_textures()

    def load_textures(self):
        """Charge toutes les textures des parties avec leur échelle"""
        print(f"\n🦊 Chargement des textures depuis {self.parts_folder}/")
        print(f"🔍 Échelle globale: {self.global_scale}x")

        for part_name, filename in self.file_mapping.items():
            filepath = os.path.join(self.parts_folder, filename)
            final_scale = self.global_scale * self.part_scales.get(part_name, 1.0)
            self.textures[part_name] = BoneTexture(filepath, scale=final_scale)

        loaded_count = sum(1 for tex in self.textures.values() if tex.loaded)
        print(f"✅ {loaded_count}/{len(self.textures)} textures chargées\n")

    def draw_quadruped(self, display, quadruped, extra_bones=None):
        """Dessine le quadrupède avec les textures dans le bon ordre

        Args:
            extra_bones: dict nom -> os virtuel pour les parties qui n'existent
                plus dans la physique (ex: la queue, devenue procédurale).
        """
        extra_bones = extra_bones or {}

        for part_name in self.draw_order:
            if part_name not in self.textures:
                continue
            # Os physique si présent, sinon os virtuel (queue procédurale).
            bone = getattr(quadruped, part_name, None) or extra_bones.get(part_name)
            if bone is None:
                continue
            texture = self.textures[part_name]
            offset = self.part_offsets.get(part_name, (0, 0))
            rotation_offset = self.rotation_offsets.get(part_name, 0)
            texture.draw(display.screen, bone, display, offset=offset, rotation_offset=rotation_offset)


class VisualOverlay:
    """Gestionnaire d'overlay visuel.

    Modes (touche TAB) :
        PROCEDURAL : peau vectorielle low poly dessinée depuis les os
        TEXTURED   : ancien système de PNG collés (legacy, renard uniquement)
        SKELETON   : os + muscles (debug)
        OVERLAY    : peau procédurale + squelette semi-transparent

    Le mode TEXTURED n'apparaît que si l'animal a des textures découpées
    (has_legacy_textures dans son AnimalDefinition).
    """

    def __init__(self, display, parts_folder="fox_parts", global_scale=0.3, definition=None):
        self.display = display
        self.render_mode = 0

        # Animal par défaut : le renard.
        if definition is None:
            from src.animals.fox import FOX
            definition = FOX
        self.definition = definition
        self.procedural_skin = ProceduralSkin(definition.skin)

        if definition.has_legacy_textures:
            self.textured_overlay = TexturedOverlay(parts_folder, global_scale=global_scale)
            self.mode_names = ["PROCEDURAL", "TEXTURED", "SKELETON", "OVERLAY"]
        else:
            self.textured_overlay = None
            self.mode_names = ["PROCEDURAL", "SKELETON", "OVERLAY"]

        self.colors = {
            'bone': (255, 255, 255),
            'joint': (255, 200, 100),
            'muscle_relaxed': (255, 100, 100),
            'muscle_active': (255, 50, 50),
        }

        self.glow_surface = pygame.Surface((display.width, display.height), pygame.SRCALPHA)

    def toggle_mode(self):
        """Bascule entre les modes d'affichage"""
        self.render_mode = (self.render_mode + 1) % len(self.mode_names)
        print(f"🔄 Mode : {self.mode_names[self.render_mode]}")

    def get_bone_vertices(self, bone):
        """Récupère les sommets d'un os à l'écran"""
        vertices = [(bone.body.transform * v) * self.display.PPM
                    for v in bone.fixture.shape.vertices]
        return [(v[0] - self.display.camera_x * self.display.PPM,
                 self.display.height - (v[1] - self.display.camera_y * self.display.PPM))
                for v in vertices]

    def draw_skeleton_bone(self, bone):
        """Dessine un os en mode skeleton"""
        vertices = self.get_bone_vertices(bone)
        if len(vertices) >= 4:
            pygame.draw.polygon(self.display.screen, self.colors['bone'], vertices)
            for vertex in vertices:
                pygame.draw.circle(self.display.screen, self.colors['bone'],
                                   (int(vertex[0]), int(vertex[1])), 3)

    def draw_muscle(self, muscle):
        """Dessine un muscle"""
        pos_a = muscle.body_a.transform * muscle.anchor_a
        pos_b = muscle.body_b.transform * muscle.anchor_b
        screen_a = self.display.to_screen(pos_a)
        screen_b = self.display.to_screen(pos_b)

        if abs(muscle.target_speed) > 0.1:
            color = self.colors['muscle_active']
            thickness = 6
            self.draw_glow_line(screen_a, screen_b, color, radius=25)
        else:
            color = self.colors['muscle_relaxed']
            thickness = 4

        pygame.draw.line(self.display.screen, color, screen_a, screen_b, thickness)
        pygame.draw.circle(self.display.screen, color, screen_a, 5)
        pygame.draw.circle(self.display.screen, color, screen_b, 5)

    def draw_glow_line(self, pos_a, pos_b, color, radius=10):
        """Effet de glow pour les muscles actifs"""
        steps = 5
        base_alpha = 50
        for i in range(steps):
            alpha = base_alpha - i * 10
            glow_radius = radius - i * 4
            glow_color = (*color, alpha)
            pygame.draw.circle(self.glow_surface, glow_color, pos_a, glow_radius)
            pygame.draw.circle(self.glow_surface, glow_color, pos_b, glow_radius)

    def draw_quadruped(self, quadruped):
        """Dessine le quadrupède selon le mode"""
        self.glow_surface.fill((0, 0, 0, 0))

        # La queue procédurale et les oreilles sont mises à jour quel que soit
        # le mode : la queue sert aussi d'os virtuels au mode TEXTURED.
        self.procedural_skin.update(quadruped)

        mode = self.mode_names[self.render_mode]

        if mode == "PROCEDURAL":
            # MODE PROCEDURAL : peau vectorielle low poly
            self.procedural_skin.draw(self.display, quadruped)

        elif mode == "TEXTURED":
            # MODE TEXTURED (legacy) : PNG collés sur les os
            self.textured_overlay.draw_quadruped(
                self.display, quadruped,
                extra_bones=self.procedural_skin.get_tail_virtual_bones())

        elif mode == "SKELETON":
            # MODE SKELETON : seulement le squelette
            for bone in quadruped.bones:
                self.draw_skeleton_bone(bone)

            for muscle in quadruped.muscles:
                if abs(muscle.target_speed) > 0.1:
                    pos_a = muscle.body_a.transform * muscle.anchor_a
                    pos_b = muscle.body_b.transform * muscle.anchor_b
                    screen_a = self.display.to_screen(pos_a)
                    screen_b = self.display.to_screen(pos_b)
                    self.draw_glow_line(screen_a, screen_b, self.colors['muscle_active'], radius=25)

            self.display.screen.blit(self.glow_surface, (0, 0))

            for muscle in quadruped.muscles:
                self.draw_muscle(muscle)

        elif mode == "OVERLAY":
            # MODE OVERLAY : peau procédurale + squelette par-dessus pour calibrage
            # 1. Dessiner la peau procédurale d'abord
            self.procedural_skin.draw(self.display, quadruped)

            # 2. Dessiner le squelette semi-transparent par-dessus
            # Créer une surface temporaire avec transparence
            overlay_surface = pygame.Surface((self.display.width, self.display.height), pygame.SRCALPHA)

            # Dessiner les os avec transparence
            for bone in quadruped.bones:
                vertices = self.get_bone_vertices(bone)
                if len(vertices) >= 4:
                    # Os semi-transparent (blanc avec alpha)
                    pygame.draw.polygon(overlay_surface, (255, 255, 255, 100), vertices)
                    # Contour plus visible
                    pygame.draw.polygon(overlay_surface, (255, 255, 0, 200), vertices, 2)
                    # Points de jonction
                    for vertex in vertices:
                        pygame.draw.circle(overlay_surface, (255, 255, 0, 255),
                                           (int(vertex[0]), int(vertex[1])), 4)

            # Dessiner les muscles semi-transparents
            for muscle in quadruped.muscles:
                pos_a = muscle.body_a.transform * muscle.anchor_a
                pos_b = muscle.body_b.transform * muscle.anchor_b
                screen_a = self.display.to_screen(pos_a)
                screen_b = self.display.to_screen(pos_b)

                # Muscles avec transparence
                pygame.draw.line(overlay_surface, (255, 100, 100, 150), screen_a, screen_b, 3)
                pygame.draw.circle(overlay_surface, (255, 0, 0, 200), screen_a, 4)
                pygame.draw.circle(overlay_surface, (255, 0, 0, 200), screen_b, 4)

            # Appliquer la surface overlay sur l'écran
            self.display.screen.blit(overlay_surface, (0, 0))

    def draw_status(self):
        """Affiche le mode actuel"""
        mode_colors = {
            "PROCEDURAL": (120, 255, 170),
            "TEXTURED": (255, 150, 50),
            "SKELETON": (255, 255, 100),
            "OVERLAY": (150, 255, 150),
        }

        current_mode = self.mode_names[self.render_mode]

        self.display.draw_text(f"Mode: {current_mode} ({self.definition.name})",
                               (10, self.display.height - 30), mode_colors[current_mode])
        self.display.draw_text(f"TAB: Changer mode ({'/'.join(m.capitalize() for m in self.mode_names)})",
                               (10, self.display.height - 55), (200, 200, 200))

        if self.textured_overlay is not None:
            loaded = sum(1 for tex in self.textured_overlay.textures.values() if tex.loaded)
            total = len(self.textured_overlay.textures)
            self.display.draw_text(f"Textures: {loaded}/{total}",
                                   (10, self.display.height - 80), (150, 200, 255))

    # def draw_status(self):
    #     """Affiche le mode actuel"""
    #     mode_names = ["TEXTURED", "SKELETON", "OVERLAY", "SOFTBODY"]
    #     mode_colors = [(255, 150, 50), (255, 255, 100), (150, 255, 150), (100, 200, 255)]
    #
    #     current_mode = mode_names[self.render_mode]
    #     current_color = mode_colors[self.render_mode]
    #
    #     self.display.draw_text(f"Mode: {current_mode}",
    #                            (10, self.display.height - 30), current_color)
    #     self.display.draw_text("TAB: Changer mode (Textured/Skeleton/Overlay/Softbody)",
    #                            (10, self.display.height - 55), (200, 200, 200))