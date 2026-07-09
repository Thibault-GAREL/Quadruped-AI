# ============================================
# procedural_skin.py - Rendu procedural low poly
# ============================================
# Dessine la peau de l'animal par code, directement depuis les os Box2D :
# aucun asset image. Le style (palette, formes, facettes) vient du SkinSpec
# de l'AnimalDefinition (src/animals/), donc chaque animal est un fichier
# de config, pas un nouveau moteur de rendu.
#
# Contenu :
# - formes rigides attachees aux os (torse, tete) definies en coordonnees locales
# - pattes et cou en capsules effilees qui suivent les os (aucune couture possible)
# - queue purement esthetique : chaine de noeuds simulee (verlet + ressort de pose)
# - oreilles montees sur ressort qui reagissent aux mouvements de la tete
# - facettes low poly : triangulation + variation deterministe de teinte

import math

import pygame


def _shade(color, factor):
    """Eclaircit (>1) ou assombrit (<1) une couleur RGB."""
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def _facet_factor(key, index, jitter):
    """Variation de teinte deterministe par facette (stable, aucun scintillement)."""
    seed = sum(ord(c) for c in key)
    t = math.sin(index * 12.9898 + seed * 0.731)
    return 1.0 + jitter * t


class _VirtualBody:
    """Expose .position et .angle comme un b2Body (pour l'overlay texture legacy)."""

    def __init__(self, position, angle):
        self.position = position
        self.angle = angle


class VirtualBone:
    """Os virtuel (non physique) compatible avec BoneTexture.draw."""

    def __init__(self, position, angle):
        self.body = _VirtualBody(position, angle)


class ProceduralSkin:
    """Peau procedurale d'un animal, pilotee par un SkinSpec."""

    def __init__(self, skin):
        self.skin = skin
        # Etat de la queue (chaine de noeuds) et des oreilles (ressorts).
        self._tail_nodes = None      # [[x, y], ...] positions courantes
        self._tail_prev = None       # positions au pas precedent (verlet)
        self._ear_state = [[0.0, 0.0] for _ in skin.ears]  # [angle, vitesse]

    # ================= GEOMETRIE =================

    @staticmethod
    def _bone_frame(bone):
        """Retourne (centre, cos, sin) de l'os."""
        pos = bone.body.position
        a = bone.body.angle
        return (pos.x, pos.y), math.cos(a), math.sin(a)

    @staticmethod
    def _bone_ends(bone):
        """Extremites de l'os le long de son axe local +y (haut, bas)."""
        (cx, cy), cos_a, sin_a = ProceduralSkin._bone_frame(bone)
        h = bone.height / 2
        # local +y en monde = (-sin, cos)
        top = (cx - sin_a * h, cy + cos_a * h)
        bottom = (cx + sin_a * h, cy - cos_a * h)
        return top, bottom

    @staticmethod
    def _local_to_world(bone, points):
        """Transforme des points du repere local de l'os vers le monde."""
        (cx, cy), cos_a, sin_a = ProceduralSkin._bone_frame(bone)
        return [(cx + px * cos_a - py * sin_a, cy + px * sin_a + py * cos_a)
                for px, py in points]

    @staticmethod
    def _head_frame(head_bone):
        """Repere "museau" de la tete : f = vers le nez, u = vers le haut."""
        pos = head_bone.body.position
        a = head_bone.body.angle
        # L'os de la tete s'etend vers le museau le long de son axe local +y.
        f = (-math.sin(a), math.cos(a))
        u = (-f[1], f[0])
        return (pos.x, pos.y), f, u

    @staticmethod
    def _muzzle_to_world(origin, f, u, points):
        ox, oy = origin
        return [(ox + px * f[0] + py * u[0], oy + px * f[1] + py * u[1])
                for px, py in points]

    # ================= DESSIN BAS NIVEAU =================

    def _to_screen(self, display, pts):
        return [display.to_screen(p) for p in pts]

    def _draw_polygon(self, display, world_pts, color, key, facets=True):
        """Polygone plein, facettise en eventail autour du centroide."""
        pts = self._to_screen(display, world_pts)
        if len(pts) < 3:
            return
        # Base pleine d'abord : evite les fissures entre facettes (arrondis).
        pygame.draw.polygon(display.screen, color, pts)
        if not facets:
            return
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        jitter = self.skin.facet_jitter
        for i in range(len(pts)):
            tri = [(cx, cy), pts[i], pts[(i + 1) % len(pts)]]
            pygame.draw.polygon(display.screen, _shade(color, _facet_factor(key, i, jitter)), tri)

    def _draw_capsule(self, display, p_top, p_bottom, hw_top, hw_bottom,
                      color, key, facets=True, world_offset=(0.0, 0.0)):
        """Segment effile entre deux points, avec bouts arrondis."""
        ox, oy = world_offset
        p_top = (p_top[0] + ox, p_top[1] + oy)
        p_bottom = (p_bottom[0] + ox, p_bottom[1] + oy)
        dx = p_bottom[0] - p_top[0]
        dy = p_bottom[1] - p_top[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        # Perpendiculaire unitaire a l'axe du segment.
        px, py = -dy / length, dx / length
        quad = [
            (p_top[0] + px * hw_top, p_top[1] + py * hw_top),
            (p_top[0] - px * hw_top, p_top[1] - py * hw_top),
            (p_bottom[0] - px * hw_bottom, p_bottom[1] - py * hw_bottom),
            (p_bottom[0] + px * hw_bottom, p_bottom[1] + py * hw_bottom),
        ]
        self._draw_polygon(display, quad, color, key, facets=facets)
        # Bouts arrondis : remplit les articulations, aucune couture visible.
        ppm = display.PPM
        pygame.draw.circle(display.screen, color, display.to_screen(p_top),
                           max(1, int(hw_top * ppm)))
        pygame.draw.circle(display.screen, color, display.to_screen(p_bottom),
                           max(1, int(hw_bottom * ppm)))

    def _draw_shape(self, display, bone_or_frame, shape, is_head=False):
        """Dessine une Shape locale (polygone ou cercle) attachee a un os."""
        color = self.skin.palette[shape.color]
        if is_head:
            origin, f, u = bone_or_frame
            if shape.kind == 'circle':
                center = self._muzzle_to_world(origin, f, u, [shape.center])[0]
                pygame.draw.circle(display.screen, color, display.to_screen(center),
                                   max(1, int(shape.radius * display.PPM)))
            else:
                pts = self._muzzle_to_world(origin, f, u, shape.points)
                self._draw_polygon(display, pts, color, shape.color, facets=shape.facets)
        else:
            if shape.kind == 'circle':
                center = self._local_to_world(bone_or_frame, [shape.center])[0]
                pygame.draw.circle(display.screen, color, display.to_screen(center),
                                   max(1, int(shape.radius * display.PPM)))
            else:
                pts = self._local_to_world(bone_or_frame, shape.points)
                self._draw_polygon(display, pts, color, shape.color, facets=shape.facets)

    # ================= QUEUE PROCEDURALE =================

    def _tail_anchor_and_rest(self, quadruped):
        """Position d'ancrage et pose de repos de la queue (suit le corps)."""
        tail = self.skin.tail
        bone = quadruped.bones_by_name[tail.anchor_bone]
        anchor = self._local_to_world(bone, [tail.anchor_local])[0]
        body_angle = bone.body.angle
        rest = [anchor]
        x, y = anchor
        for deg in tail.rest_angles_deg:
            a = body_angle + math.radians(deg)
            x += tail.segment_length * math.cos(a)
            y += tail.segment_length * math.sin(a)
            rest.append((x, y))
        return anchor, rest

    def _reset_tail(self, rest):
        self._tail_nodes = [list(p) for p in rest]
        self._tail_prev = [list(p) for p in rest]

    def update_tail(self, quadruped, dt):
        """Simule la chaine de la queue : verlet + ressort vers la pose de repos."""
        tail = self.skin.tail
        if tail is None:
            return
        anchor, rest = self._tail_anchor_and_rest(quadruped)

        # Premiere frame, ou teleportation (reset d'episode) : pose de repos.
        if self._tail_nodes is None or \
                math.hypot(anchor[0] - self._tail_nodes[0][0],
                           anchor[1] - self._tail_nodes[0][1]) > 0.8:
            self._reset_tail(rest)
            return

        nodes, prev = self._tail_nodes, self._tail_prev
        nodes[0] = [anchor[0], anchor[1]]
        keep = max(0.0, 1.0 - tail.damping * dt)
        for i in range(1, len(nodes)):
            vx = (nodes[i][0] - prev[i][0]) * keep
            vy = (nodes[i][1] - prev[i][1]) * keep
            ax = tail.stiffness * (rest[i][0] - nodes[i][0])
            ay = tail.stiffness * (rest[i][1] - nodes[i][1]) - tail.gravity
            prev[i] = list(nodes[i])
            nodes[i][0] += vx + ax * dt * dt
            nodes[i][1] += vy + ay * dt * dt

        # Contraintes de distance : longueur des segments preservee.
        seg = tail.segment_length
        for _ in range(3):
            for i in range(1, len(nodes)):
                dx = nodes[i][0] - nodes[i - 1][0]
                dy = nodes[i][1] - nodes[i - 1][1]
                d = math.hypot(dx, dy)
                if d > 1e-6:
                    scale = seg / d
                    nodes[i][0] = nodes[i - 1][0] + dx * scale
                    nodes[i][1] = nodes[i - 1][1] + dy * scale

    def _draw_tail(self, display):
        tail = self.skin.tail
        if tail is None or self._tail_nodes is None:
            return
        nodes = self._tail_nodes
        n_seg = len(nodes) - 1
        tip_start = max(1, int(round(n_seg * (1.0 - tail.tip_ratio))))
        coat = self.skin.palette[tail.color]
        tip = self.skin.palette[tail.tip_color]
        jitter = self.skin.facet_jitter

        # Perpendiculaire en chaque noeud (moyenne des segments adjacents).
        perps = []
        for i in range(len(nodes)):
            a = nodes[max(0, i - 1)]
            b = nodes[min(len(nodes) - 1, i + 1)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            d = math.hypot(dx, dy)
            perps.append((-dy / d, dx / d) if d > 1e-6 else (0.0, 1.0))

        for i in range(n_seg):
            color = tip if i >= tip_start else coat
            hw_a = tail.half_widths[i]
            hw_b = tail.half_widths[i + 1]
            quad = [
                (nodes[i][0] + perps[i][0] * hw_a, nodes[i][1] + perps[i][1] * hw_a),
                (nodes[i][0] - perps[i][0] * hw_a, nodes[i][1] - perps[i][1] * hw_a),
                (nodes[i + 1][0] - perps[i + 1][0] * hw_b, nodes[i + 1][1] - perps[i + 1][1] * hw_b),
                (nodes[i + 1][0] + perps[i + 1][0] * hw_b, nodes[i + 1][1] + perps[i + 1][1] * hw_b),
            ]
            pts = self._to_screen(display, quad)
            pygame.draw.polygon(display.screen, color, pts)
            # Deux facettes par segment.
            pygame.draw.polygon(display.screen,
                                _shade(color, _facet_factor('tail', i * 2, jitter)),
                                [pts[0], pts[1], pts[2]])
            pygame.draw.polygon(display.screen,
                                _shade(color, _facet_factor('tail', i * 2 + 1, jitter)),
                                [pts[0], pts[2], pts[3]])

    def get_tail_virtual_bones(self):
        """Os virtuels de la queue pour l'overlay texture legacy (3 segments)."""
        if self._tail_nodes is None or len(self._tail_nodes) < 7:
            return {}
        nodes = self._tail_nodes

        def vbone(i0, i1):
            cx = (nodes[i0][0] + nodes[i1][0]) / 2
            cy = (nodes[i0][1] + nodes[i1][1]) / 2
            dx = nodes[i1][0] - nodes[i0][0]
            dy = nodes[i1][1] - nodes[i0][1]
            # Angle equivalent d'un os dont l'axe local +y suit le segment.
            return VirtualBone((cx, cy), math.atan2(dy, dx) - math.pi / 2)

        return {
            'tail_bottom': vbone(0, 2),
            'tail_mid': vbone(2, 4),
            'tail_high': vbone(4, 6),
        }

    # ================= OREILLES A RESSORT =================

    def update_ears(self, quadruped, dt):
        """Ressort amorti : les oreilles reagissent a la vitesse angulaire de la tete."""
        if not self.skin.ears:
            return
        head = quadruped.bones_by_name.get('head')
        if head is None:
            return
        omega = head.body.angularVelocity
        for spec, state in zip(self.skin.ears, self._ear_state):
            target = max(-spec.max_deflection,
                         min(spec.max_deflection, -omega * spec.react_gain))
            acc = spec.stiffness * (target - state[0]) - spec.damping * state[1]
            state[1] += acc * dt
            state[0] += state[1] * dt
            state[0] = max(-spec.max_deflection, min(spec.max_deflection, state[0]))

    def _draw_ear(self, display, head_frame, spec, state):
        origin, f, u = head_frame
        bx, by = spec.base_local
        cos_o, sin_o = math.cos(state[0]), math.sin(state[0])

        def deflect(points):
            # Rotation des points de l'oreille autour de sa base (ressort).
            return [(bx + px * cos_o - py * sin_o, by + px * sin_o + py * cos_o)
                    for px, py in points]

        pts = self._muzzle_to_world(origin, f, u, deflect(spec.points))
        self._draw_polygon(display, pts, self.skin.palette[spec.color],
                           spec.color + '_ear')
        if spec.inner_points:
            inner = self._muzzle_to_world(origin, f, u, deflect(spec.inner_points))
            self._draw_polygon(display, inner, self.skin.palette[spec.inner_color],
                               spec.inner_color, facets=False)

    # ================= DESSIN COMPLET =================

    def _draw_leg_chain(self, display, quadruped, chain, darken=1.0, offset=(0.0, 0.0)):
        for bone_name in chain:
            style = self.skin.legs.get(bone_name)
            bone = quadruped.bones_by_name.get(bone_name)
            if style is None or bone is None:
                continue
            top, bottom = self._bone_ends(bone)
            color = _shade(self.skin.palette[style.color], darken)
            self._draw_capsule(display, top, bottom, style.hw_top, style.hw_bottom,
                               color, bone_name, world_offset=offset)

    def _draw_neck(self, display, quadruped):
        neck = quadruped.bones_by_name.get(self.skin.neck_bone)
        if neck is None:
            return
        top, bottom = self._bone_ends(neck)
        # top = cote corps, bottom = cote tete (cf. squelette du renard).
        self._draw_capsule(display, top, bottom, self.skin.neck_hw_base,
                           self.skin.neck_hw_top,
                           self.skin.palette[self.skin.neck_color], 'neck')

    def update(self, quadruped, dt=1.0 / 60.0):
        """Met a jour les animations procedurales (queue, oreilles)."""
        self.update_tail(quadruped, dt)
        self.update_ears(quadruped, dt)

    def draw(self, display, quadruped):
        """Dessine l'animal complet (appeler update() avant)."""
        skin = self.skin
        far_off = skin.far_leg_offset

        # 1. Pattes du fond (plus sombres, legerement decalees).
        for chain in skin.far_leg_chains:
            self._draw_leg_chain(display, quadruped, chain,
                                 darken=skin.far_leg_darken, offset=far_off)

        # 2. Queue.
        self._draw_tail(display)

        # 3. Pattes de devant.
        for chain in skin.leg_chains:
            self._draw_leg_chain(display, quadruped, chain)

        # 4. Cou.
        self._draw_neck(display, quadruped)

        # 5. Tete et oreilles (oreille du fond avant le crane, l'autre apres).
        head = quadruped.bones_by_name.get('head')
        if head is not None:
            head_frame = self._head_frame(head)
            if skin.ears:
                self._draw_ear(display, head_frame, skin.ears[0], self._ear_state[0])
            for shape in skin.head_shapes:
                self._draw_shape(display, head_frame, shape, is_head=True)
            for spec, state in zip(skin.ears[1:], self._ear_state[1:]):
                self._draw_ear(display, head_frame, spec, state)

        # 6. Torse (dessine en dernier pour passer au premier plan).
        body = quadruped.body
        for shape in skin.body_shapes:
            self._draw_shape(display, body, shape)
