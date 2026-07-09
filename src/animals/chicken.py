# ============================================
# chicken.py - Definition de la poule (bipede)
# ============================================
# Anatomie d'oiseau : par patte, 4 os articules comme la patte arriere du
# renard (hanche, genou, articulation intertarsienne "a l'envers", orteils),
# soit 2 pattes x 4 = 8 muscles actionnes : meme taille de sortie que le
# renard, les deux IA (choreography et neuro_ga) fonctionnent sans changement.
#
# Particularites bipede :
# - self_collide=False : les deux pattes se superposent en vue de profil,
#   elles ne doivent pas entrer en collision entre elles.
# - spawn_y bas : la poule apparait juste au-dessus du sol.
#
# Cote peau : crete et barbillon montes sur ressort (ils gigotent), queue en
# eventail procedurale (chaine courte et rigide).

import math

from src.animals.definition import (
    AnimalDefinition, BoneDef, EarSpec, LegStyle, MuscleDef,
    SkeletonDef, Shape, SkinSpec, TailSpec,
)

# ----- Dimensions du squelette -----
# Pattes courtes : une poule est dodue, les pattes visibles font environ la
# moitie de la hauteur du corps.
WIDTH_BONE = 0.05
BODY_LEN = 0.8
DENSITY_BONE = 0.5
THIGH_LEN = 0.22
SHIN_LEN = 0.22
SHANK_LEN = 0.18   # tarsometatarse : la partie fine et jaune de la patte
FOOT_LEN = 0.18
NECK_LEN = 0.28
HEAD_LEN = 0.22
MARGE = 0.05
HIP_X = 0.06       # les hanches sont proches du centre (ecart avant/arriere leger)


def _leg_bones(side: str, hip_x: float):
    """Os d'une patte (side = 'right' ou 'left')."""
    return [
        BoneDef(f'{side}_thigh', hip_x, -0.17, WIDTH_BONE, THIGH_LEN, DENSITY_BONE),
        BoneDef(f'{side}_shin', hip_x, -0.39, WIDTH_BONE, SHIN_LEN, DENSITY_BONE),
        BoneDef(f'{side}_shank', hip_x, -0.59, WIDTH_BONE, SHANK_LEN, DENSITY_BONE),
        BoneDef(f'{side}_foot', hip_x - 0.08, -0.72, WIDTH_BONE, FOOT_LEN, DENSITY_BONE),
    ]


def _leg_muscles(side: str, hip_x: float):
    """Muscles d'une patte : memes limites que la patte arriere du renard."""
    return [
        MuscleDef(f'{side}_hip', 'body', f'{side}_thigh',
                  (hip_x, -WIDTH_BONE), (0, THIGH_LEN / 2 + MARGE),
                  -math.pi * 0.3, math.pi * 0.35, max_torque=3000, actuated=True),
        MuscleDef(f'{side}_knee', f'{side}_thigh', f'{side}_shin',
                  (0, -THIGH_LEN / 2 + MARGE), (0, SHIN_LEN / 2 + MARGE),
                  -math.pi * 0.7, 0, max_torque=3500, actuated=True),
        MuscleDef(f'{side}_ankle', f'{side}_shin', f'{side}_shank',
                  (0, -SHIN_LEN / 2 + MARGE), (0, SHANK_LEN / 2 + MARGE),
                  -math.pi * 0.7, 0, max_torque=3500, actuated=True),
        MuscleDef(f'{side}_foot_joint', f'{side}_shank', f'{side}_foot',
                  (WIDTH_BONE, -(SHANK_LEN / 2 + MARGE)), (WIDTH_BONE, MARGE),
                  math.pi * 0.3, math.pi * 0.6, max_torque=1500, actuated=True),
    ]


def _build_skeleton() -> SkeletonDef:
    bones = [
        BoneDef('body', 0.0, 0.0, BODY_LEN, WIDTH_BONE, DENSITY_BONE),
        *_leg_bones('right', HIP_X),
        *_leg_bones('left', -HIP_X),
        BoneDef('neck', 0.50, 0.18, WIDTH_BONE, NECK_LEN, DENSITY_BONE),
        BoneDef('head', 0.55, 0.31, WIDTH_BONE, HEAD_LEN, DENSITY_BONE),
    ]

    # Les 8 muscles actionnes d'abord (patte droite puis gauche).
    muscles = [
        *_leg_muscles('right', HIP_X),
        *_leg_muscles('left', -HIP_X),
        # Cou quasi vertical et tete bec vers l'avant : joints figes.
        MuscleDef('neck_joint', 'body', 'neck',
                  (BODY_LEN / 2 + MARGE, WIDTH_BONE), (0, NECK_LEN / 2),
                  math.pi * 0.889, math.pi * 0.889, max_torque=40),
        MuscleDef('head_joint', 'neck', 'head',
                  (WIDTH_BONE, -(NECK_LEN / 2 + MARGE)), (WIDTH_BONE, -WIDTH_BONE),
                  math.pi * 0.58, math.pi * 0.58, max_torque=40),
    ]

    return SkeletonDef(bones=bones, muscles=muscles, root='body',
                       self_collide=False)


def _build_skin() -> SkinSpec:
    palette = {
        'plumage': (242, 238, 228),       # blanc casse du corps
        'wing': (226, 218, 202),          # aile legerement plus sombre
        'plumage_dark': (206, 196, 178),  # bout des plumes de la queue
        'comb': (214, 60, 50),            # crete et barbillon rouges
        'beak': (238, 172, 60),           # bec jaune-orange
        'legs_y': (230, 176, 70),         # pattes jaunes
        'eye': (25, 22, 20),
        'ear_inner': (180, 40, 34),       # defaut EarSpec (inutilise ici, robustesse)
    }

    # ----- Corps dodu (repere de l'os 'body', +x = avant) -----
    body_outline = [
        (-0.40, 0.20),   # naissance de la queue
        (-0.15, 0.28),   # dos
        (0.15, 0.27),
        (0.38, 0.18),    # epaule vers le cou
        (0.47, 0.03),    # jabot haut
        (0.43, -0.17),   # poitrail
        (0.22, -0.29),   # dessous avant
        (-0.10, -0.30),  # ventre
        (-0.33, -0.21),  # arriere ventre
        (-0.45, -0.04),  # dessous de la queue
    ]
    wing = [
        (0.14, 0.14),
        (0.26, 0.02),
        (0.12, -0.13),
        (-0.14, -0.14),
        (-0.36, -0.05),
        (-0.14, 0.11),
    ]
    body_shapes = [
        Shape('plumage', points=body_outline),
        Shape('wing', points=wing),
    ]

    # ----- Tete (repere museau : +x vers le bec, +y vers le haut) -----
    # Tete nettement agrandie (~1.35x) et arrondie, gros oeil avec reflet et
    # bec plus marque : le tout donne un regard bien plus vivant.
    head_shapes = [
        Shape(
            "beak",
            points=[
                (0.13, 0.075),
                (0.37, 0.005),
                (0.13, -0.075),
            ],
            facets=False,
        ),
        Shape(
            "plumage",
            points=[
                (-0.18, 0.03),
                (-0.15, 0.14),
                (-0.05, 0.20),
                (0.08, 0.19),
                (0.18, 0.09),
                (0.19, -0.03),
                (0.14, -0.15),
                (0.02, -0.21),
                (-0.10, -0.19),
                (-0.17, -0.09),
            ],
        ),
        # Zone claire autour de l'oeil pour l'agrandir visuellement.
        Shape("eye", kind="circle", center=(0.05, 0.06), radius=0.052, facets=False),
        # Reflet clair : donne du vivant au regard.
        Shape(
            "plumage", kind="circle", center=(0.075, 0.09), radius=0.02, facets=False
        ),
    ]

    # ----- Crete et barbillon : ressorts (EarSpec est generique) -----
    # Repositionnes/agrandis pour la tete plus grosse.
    comb = EarSpec(
        base_local=(-0.02, 0.17),
        points=[(-0.11, -0.01), (-0.07, 0.13), (-0.01, 0.05),
                (0.03, 0.16), (0.07, 0.05), (0.11, 0.12), (0.13, -0.02)],
        color='comb',
        stiffness=70.0, damping=8.0, react_gain=0.05, max_deflection=0.45,
    )
    wattle = EarSpec(
        base_local=(0.09, -0.11),
        points=[(-0.05, -0.07), (0.0, -0.17), (0.06, 0.0)],
        color='comb',
        stiffness=40.0, damping=6.0, react_gain=0.09, max_deflection=0.7,
    )
    # ears[0] est dessinee derriere le crane (crete), les suivantes devant.
    ears = [comb, wattle]

    # ----- Pattes : cuisse et pilon emplumes, tarse et orteils jaunes -----
    def leg_styles(side):
        return {
            f'{side}_thigh': LegStyle(hw_top=0.10, hw_bottom=0.06, color='plumage'),
            f'{side}_shin': LegStyle(hw_top=0.165, hw_bottom=0.065, color='plumage'),
            f'{side}_shank': LegStyle(hw_top=0.05, hw_bottom=0.03, color='legs_y'),
            f'{side}_foot': LegStyle(hw_top=0.025, hw_bottom=0.02, color='legs_y'),
        }

    legs = {**leg_styles('right'), **leg_styles('left')}
    leg_chains = [['right_thigh', 'right_shin', 'right_shank', 'right_foot']]
    far_leg_chains = [['left_thigh', 'left_shin', 'left_shank', 'left_foot']]

    # ----- Queue en eventail : chaine courte et rigide, elle fretille -----
    tail = TailSpec(
        anchor_bone='body',
        anchor_local=(-0.40, 0.14),
        segment_length=0.13,
        rest_angles_deg=[138, 122, 106],
        half_widths=[0.05, 0.10, 0.145, 0.06],
        tip_color='plumage_dark',
        tip_ratio=0.34,
        color='plumage',
        stiffness=55.0,
        damping=5.0,
        gravity=0.8,
    )

    return SkinSpec(
        palette=palette,
        facet_jitter=0.05,
        body_shapes=body_shapes,
        head_shapes=head_shapes,
        neck_bone='neck',
        neck_hw_base=0.14,
        neck_hw_top=0.10,
        neck_color='plumage',
        legs=legs,
        leg_chains=leg_chains,
        far_leg_chains=far_leg_chains,
        far_leg_darken=0.72,
        far_leg_offset=(0.05, 0.0),
        tail=tail,
        ears=ears,
    )


CHICKEN = AnimalDefinition(
    name='chicken',
    skeleton=_build_skeleton(),
    skin=_build_skin(),
    spawn_y=1.32,   # la poule apparait juste au-dessus du sol
    has_legacy_textures=False,
)
