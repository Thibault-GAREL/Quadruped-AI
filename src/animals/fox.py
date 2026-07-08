# ============================================
# fox.py - Definition du renard (squelette + peau procedurale)
# ============================================
# Squelette : memes valeurs que l'ancien Quadruped code en dur, SAUF la
# queue qui n'est plus simulee par Box2D : elle est devenue une chaine
# procedurale purement esthetique (cf. TailSpec), ce qui la rend fluide
# et supprime 3 os + 3 joints de la physique.

import math

from src.animals.definition import (
    AnimalDefinition, BoneDef, EarSpec, LegStyle, MuscleDef,
    SkeletonDef, Shape, SkinSpec, TailSpec,
)

# ----- Dimensions du squelette (identiques a l'ancien code) -----
WIDTH_BONE = 0.05
BODY_HEIGHT = 1.3
BODY_HEIGHT_TAIL_HEAD = 1.4
DENSITY_BONE = 0.5
THIGH_HEIGHT = 0.5
SHIN_HEIGHT_F = 0.3
SHIN_HEIGHT_B = 0.4
FOOT_HEIGHT = 0.2
ANKLE_HEIGHT_F = 0.2
ANKLE_HEIGHT_B = 0.2
NECK_HEIGHT = 0.4
MARGE = 0.05


def _build_skeleton() -> SkeletonDef:
    bones = [
        BoneDef('body', 0.0, 0.0, BODY_HEIGHT_TAIL_HEAD, WIDTH_BONE, DENSITY_BONE),

        BoneDef('front_thigh', 0.8, -0.5, WIDTH_BONE, THIGH_HEIGHT, DENSITY_BONE),
        BoneDef('front_shin', 0.8, -1.3, WIDTH_BONE, SHIN_HEIGHT_F, DENSITY_BONE),
        BoneDef('front_ankle', 0.8, -1.4, WIDTH_BONE, ANKLE_HEIGHT_F, DENSITY_BONE),
        BoneDef('front_foot', 0.7, -1.5, WIDTH_BONE, FOOT_HEIGHT, DENSITY_BONE),

        BoneDef('back_thigh', -0.8, -0.5, WIDTH_BONE, THIGH_HEIGHT, DENSITY_BONE),
        BoneDef('back_shin', -0.8, -1.3, WIDTH_BONE, SHIN_HEIGHT_B, DENSITY_BONE),
        BoneDef('back_ankle', -0.8, -1.4, WIDTH_BONE, ANKLE_HEIGHT_B, DENSITY_BONE),
        BoneDef('back_foot', -0.7, -1.5, WIDTH_BONE, FOOT_HEIGHT, DENSITY_BONE),

        BoneDef('neck', 0.9, 0.1, WIDTH_BONE, NECK_HEIGHT, DENSITY_BONE),
        BoneDef('head', 0.99, 0.1, WIDTH_BONE, NECK_HEIGHT, DENSITY_BONE),
    ]

    # IMPORTANT : les 8 muscles actionnes par l'IA d'abord (indices 0..7).
    muscles = [
        MuscleDef('front_hip', 'body', 'front_thigh',
                  (BODY_HEIGHT / 2, -WIDTH_BONE), (0, THIGH_HEIGHT / 2 + MARGE),
                  -math.pi * 0.45, math.pi * 0.1, max_torque=4000, actuated=True),
        MuscleDef('front_knee', 'front_thigh', 'front_shin',
                  (0, -THIGH_HEIGHT / 2 + MARGE), (0, SHIN_HEIGHT_F / 2 + MARGE),
                  0, math.pi * 0.8, max_torque=5000, actuated=True),
        MuscleDef('front_ankle_joint', 'front_shin', 'front_ankle',
                  (WIDTH_BONE, -(SHIN_HEIGHT_F / 2 + MARGE)), (WIDTH_BONE, ANKLE_HEIGHT_F / 2 + MARGE),
                  0, math.pi * 0.4, max_torque=5000, actuated=True),
        MuscleDef('front_foot_joint', 'front_ankle', 'front_foot',
                  (WIDTH_BONE, -(ANKLE_HEIGHT_F / 2 + MARGE)), (WIDTH_BONE, MARGE),
                  math.pi * 0.3, math.pi * 0.6, max_torque=2000, actuated=True),

        MuscleDef('back_hip', 'body', 'back_thigh',
                  (-BODY_HEIGHT / 2, -WIDTH_BONE), (0, THIGH_HEIGHT / 2 + MARGE),
                  -math.pi * 0.3, math.pi * 0.35, max_torque=4000, actuated=True),
        MuscleDef('back_knee', 'back_thigh', 'back_shin',
                  (0, -THIGH_HEIGHT / 2 + MARGE), (0, SHIN_HEIGHT_B / 2 + MARGE),
                  -math.pi * 0.7, 0, max_torque=5000, actuated=True),
        MuscleDef('back_ankle_joint', 'back_shin', 'back_ankle',
                  (0, -SHIN_HEIGHT_B / 2 + MARGE), (0, ANKLE_HEIGHT_B / 2 + MARGE),
                  -math.pi * 0.7, 0, max_torque=5000, actuated=True),
        MuscleDef('back_foot_joint', 'back_ankle', 'back_foot',
                  (WIDTH_BONE, -(ANKLE_HEIGHT_B / 2 + MARGE)), (WIDTH_BONE, MARGE),
                  math.pi * 0.3, math.pi * 0.6, max_torque=2000, actuated=True),

        # Cou et tete : joints figes (non actionnes), comme avant.
        MuscleDef('neck_joint', 'body', 'neck',
                  (BODY_HEIGHT_TAIL_HEAD / 2 + MARGE, WIDTH_BONE), (0, NECK_HEIGHT / 2),
                  math.pi * 0.7, math.pi * 0.7, max_torque=40),
        MuscleDef('head_joint', 'neck', 'head',
                  (WIDTH_BONE, -(NECK_HEIGHT / 2 + MARGE)), (WIDTH_BONE, -WIDTH_BONE),
                  math.pi * 0.55, math.pi * 0.55, max_torque=40),
    ]

    return SkeletonDef(bones=bones, muscles=muscles, root='body')


def _build_skin() -> SkinSpec:
    palette = {
        'coat': (227, 133, 60),        # orange du pelage
        'coat_dark': (198, 106, 42),   # orange sombre (ombres, croupe)
        'cream': (243, 233, 220),      # ventre, poitrail, bout de queue
        'socks': (61, 46, 40),         # chaussettes sombres
        'nose': (35, 28, 25),
        'eye': (30, 24, 20),
        'ear_inner': (84, 58, 48),
    }

    # ----- Torse : polygones dans le repere de l'os 'body' (+x = avant) -----
    torso_outline = [
        (-0.80, 0.14),   # racine de la queue
        (-0.72, 0.30),   # croupe
        (-0.35, 0.36),   # dos arriere
        (0.15, 0.34),    # dos
        (0.55, 0.30),    # garrot
        (0.74, 0.16),    # base du cou
        (0.76, -0.02),   # poitrail haut
        (0.64, -0.26),   # poitrail bas
        (0.28, -0.30),   # dessous de poitrine
        (-0.10, -0.22),  # ventre
        (-0.45, -0.20),  # pli du flanc
        (-0.70, -0.30),  # bas de cuisse
        (-0.84, -0.06),  # arriere de cuisse
    ]
    belly_strip = [
        (0.64, -0.26),
        (0.28, -0.30),
        (-0.10, -0.22),
        (-0.42, -0.19),
        (-0.30, -0.08),
        (0.30, -0.13),
        (0.60, -0.10),
    ]
    body_shapes = [
        Shape('coat', points=torso_outline),
        Shape('cream', points=belly_strip),
    ]

    # ----- Tete : repere "museau" (+x vers le nez, +y vers le haut) -----
    head_shapes = [
        Shape('coat', points=[
            (-0.14, 0.10),   # arriere du crane
            (0.08, 0.15),    # sommet de la tete
            (0.28, 0.03),    # haut du museau
            (0.44, -0.05),   # bout du nez
            (0.28, -0.13),   # dessous du museau
            (0.04, -0.17),   # machoire
            (-0.16, -0.08),  # arriere bas
        ]),
        Shape('cream', points=[
            (0.44, -0.05),
            (0.27, -0.12),
            (0.02, -0.15),
            (-0.10, -0.06),
            (0.06, -0.03),
            (0.28, -0.03),
        ], facets=False),
        Shape('nose', kind='circle', center=(0.43, -0.05), radius=0.028, facets=False),
        Shape('eye', kind='circle', center=(0.14, 0.02), radius=0.024, facets=False),
    ]

    # ----- Oreilles (repere museau, relatives a leur base) -----
    ear_shape = [(-0.06, 0.0), (0.0, 0.26), (0.09, 0.02)]
    ear_inner = [(-0.03, 0.02), (0.0, 0.18), (0.05, 0.03)]
    ears = [
        # Oreille du fond (dessinee en premier, plus sombre via inner seul)
        EarSpec(base_local=(-0.16, 0.06), points=ear_shape, inner_points=[],
                color='coat_dark'),
        # Oreille de devant
        EarSpec(base_local=(-0.04, 0.10), points=ear_shape, inner_points=ear_inner,
                color='coat'),
    ]

    # ----- Pattes : capsules suivant les os -----
    legs = {
        'front_thigh': LegStyle(hw_top=0.11, hw_bottom=0.055, color='coat'),
        'front_shin': LegStyle(hw_top=0.055, hw_bottom=0.04, color='coat'),
        'front_ankle': LegStyle(hw_top=0.04, hw_bottom=0.032, color='socks'),
        'front_foot': LegStyle(hw_top=0.034, hw_bottom=0.03, color='socks'),
        'back_thigh': LegStyle(hw_top=0.13, hw_bottom=0.06, color='coat'),
        'back_shin': LegStyle(hw_top=0.06, hw_bottom=0.042, color='coat'),
        'back_ankle': LegStyle(hw_top=0.042, hw_bottom=0.032, color='socks'),
        'back_foot': LegStyle(hw_top=0.034, hw_bottom=0.03, color='socks'),
    }
    leg_chains = [
        ['back_thigh', 'back_shin', 'back_ankle', 'back_foot'],
        ['front_thigh', 'front_shin', 'front_ankle', 'front_foot'],
    ]
    # Les 2 pattes physiques sont dupliquees au fond : effet 4 pattes.
    far_leg_chains = leg_chains

    # ----- Queue procedurale (ancree a l'arriere du corps) -----
    tail = TailSpec(
        anchor_bone='body',
        anchor_local=(-0.78, 0.08),
        segment_length=0.15,
        # 180 = droit vers l'arriere ; la queue retombe puis remonte au bout.
        rest_angles_deg=[195, 205, 200, 185, 170, 158],
        half_widths=[0.07, 0.105, 0.125, 0.12, 0.10, 0.07, 0.025],
        tip_color='cream',
        tip_ratio=0.30,
        color='coat',
        stiffness=35.0,
        damping=4.0,
        gravity=1.5,
    )

    return SkinSpec(
        palette=palette,
        facet_jitter=0.055,
        body_shapes=body_shapes,
        head_shapes=head_shapes,
        neck_bone='neck',
        neck_hw_base=0.16,
        neck_hw_top=0.11,
        neck_color='coat',
        legs=legs,
        leg_chains=leg_chains,
        far_leg_chains=far_leg_chains,
        far_leg_darken=0.68,
        far_leg_offset=(0.07, 0.0),
        tail=tail,
        ears=ears,
    )


FOX = AnimalDefinition(
    name='fox',
    skeleton=_build_skeleton(),
    skin=_build_skin(),
    spawn_y=3.0,
    has_legacy_textures=True,  # les PNG fox_texture_* restent utilisables (mode TEXTURED)
)
