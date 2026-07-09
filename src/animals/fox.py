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
        "coat": (190, 99, 32),  # orange du pelage (227, 133, 60)
        "coat_dark": (123, 70, 41),  # orange sombre (ombres, croupe) (198, 106, 42)
        "cream": (215, 208, 198),  # ventre, poitrail, bout de queue
        "socks": (45, 34, 27),  # chaussettes sombres
        "nose": (35, 28, 25),
        "eye": (30, 24, 20),
        "ear_inner": (84, 58, 48),
        "sun": (222, 116, 37),  # eclairage soleil sur le dessus du dos/tete
    }

    # ----- Torse : polygones dans le repere de l'os 'body' (+x = avant) -----
    # Silhouette volontairement gonflee (dos plus haut, ventre plus bas) pour
    # un renard charnu, pas colle aux os.
    torso_outline = [
        (-0.88, 0.16),   # racine de la queue
        (-0.78, 0.36),   # croupe
        (-0.38, 0.45),   # dos arriere
        (0.15, 0.43),    # dos
        (0.60, 0.37),    # garrot
        (0.82, 0.19),    # base du cou
        (0.85, -0.06),   # poitrail haut
        (0.70, -0.34),   # poitrail bas
        (0.30, -0.39),   # dessous de poitrine
        (-0.10, -0.32),  # ventre
        (-0.48, -0.28),  # pli du flanc
        (-0.76, -0.36),  # bas de cuisse
        (-0.92, -0.06),  # arriere de cuisse
    ]
    belly_strip = [
        (0.70, -0.34),
        (0.30, -0.39),
        (-0.10, -0.32),
        (-0.46, -0.26),
        (-0.32, -0.12),
        (0.32, -0.18),
        (0.66, -0.13),
    ]
    # Bande d'eclairage solaire : suit le haut du dos, teinte plus claire.
    sun_strip = [
        (-0.80, 0.30),   # croupe
        (-0.38, 0.42),   # dos arriere (haut)
        (0.15, 0.40),    # dos
        (0.58, 0.34),    # garrot
        (0.78, 0.17),    # vers la base du cou
        (0.58, 0.15),    # redescend (bord bas de la bande)
        (0.15, 0.23),
        (-0.38, 0.25),
        (-0.74, 0.17),
    ]
    body_shapes = [
        Shape('coat', points=torso_outline),
        Shape('sun', points=sun_strip),      # highlight du dessus (soleil)
        Shape('cream', points=belly_strip),
    ]

    # ----- Tete : repere "museau" (+x vers le nez, +y vers le haut) -----
    # Tete agrandie (~1.2x), joues plus pleines, oeil plus gros (+ reflet clair).
    head_shapes = [
        Shape('coat', points=[
            (-0.18, 0.12),   # arriere du crane
            (0.10, 0.19),    # sommet de la tete
            (0.34, 0.04),    # haut du museau
            (0.52, -0.06),   # bout du nez
            (0.34, -0.16),   # dessous du museau
            (0.05, -0.21),   # machoire (joue pleine)
            (-0.20, -0.10),  # arriere bas
        ]),
        Shape('cream', points=[
            (0.52, -0.06),
            (0.33, -0.15),
            (0.02, -0.18),
            (-0.12, -0.07),
            (0.07, -0.04),
            (0.34, -0.04),
        ], facets=False),
        Shape('nose', kind='circle', center=(0.51, -0.06), radius=0.034, facets=False),
        Shape('eye', kind='circle', center=(0.17, 0.03), radius=0.034, facets=False),
        # Petit reflet clair dans l'oeil (le rend vivant).
        Shape('cream', kind='circle', center=(0.19, 0.055), radius=0.013, facets=False),
    ]

    # ----- Oreilles (repere museau, relatives a leur base) -----
    # Un peu plus grandes pour rester proportionnees a la tete agrandie.
    ear_shape = [(-0.07, 0.0), (0.0, 0.30), (0.10, 0.02)]
    ear_inner = [(-0.035, 0.02), (0.0, 0.21), (0.055, 0.03)]
    ears = [
        # Oreille du fond (dessinee en premier, plus sombre via inner seul)
        EarSpec(base_local=(0.005, 0.05), points=ear_shape, inner_points=[],
                color='coat_dark'),
        # Oreille de devant
        EarSpec(base_local=(-0.1, 0.12), points=ear_shape, inner_points=ear_inner,
                color='coat'),
    ]

    # ----- Pattes : capsules suivant les os (plus charnues, surtout en haut) -----
    legs = {
        'front_thigh': LegStyle(hw_top=0.15, hw_bottom=0.085, color='coat'),
        'front_shin': LegStyle(hw_top=0.078, hw_bottom=0.052, color='coat'),
        'front_ankle': LegStyle(hw_top=0.046, hw_bottom=0.036, color='socks'),
        'front_foot': LegStyle(hw_top=0.038, hw_bottom=0.032, color='socks'),
        'back_thigh': LegStyle(hw_top=0.17, hw_bottom=0.095, color='coat'),
        'back_shin': LegStyle(hw_top=0.088, hw_bottom=0.056, color='coat'),
        'back_ankle': LegStyle(hw_top=0.05, hw_bottom=0.036, color='socks'),
        'back_foot': LegStyle(hw_top=0.038, hw_bottom=0.032, color='socks'),
    }
    back_chain = ['back_thigh', 'back_shin', 'back_ankle', 'back_foot']
    front_chain = ['front_thigh', 'front_shin', 'front_ankle', 'front_foot']
    # Patte arriere DERRIERE le torse (sort de sous la croupe), patte avant
    # DEVANT le torse (sa cuisse passe devant le ventre, sinon le blanc la mange).
    leg_chains = [back_chain]
    front_leg_chains = [front_chain]
    # Les 2 pattes physiques sont aussi dupliquees au fond : effet 4 pattes.
    far_leg_chains = [back_chain, front_chain]

    # ----- Queue procedurale (ancree a l'arriere du corps) -----
    tail = TailSpec(
        anchor_bone='body',
        anchor_local=(-0.80, 0.10),
        segment_length=0.15,
        # 180 = droit vers l'arriere ; la queue retombe puis remonte au bout.
        rest_angles_deg=[195, 205, 200, 185, 170, 158],
        half_widths=[0.10, 0.15, 0.18, 0.17, 0.14, 0.10, 0.03],
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
        neck_hw_base=0.24,
        neck_hw_top=0.17,
        neck_color='coat',
        legs=legs,
        leg_chains=leg_chains,
        front_leg_chains=front_leg_chains,
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
