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
DENSITY_BONE = 0.15
DENSITY_BACK = 0.1
THIGH_LEN = 0.22
SHIN_LEN = 0.22
SHANK_LEN = 0.18   # tarsometatarse : la partie fine et jaune de la patte
FOOT_LEN = 0.28
NECK_LEN = 0.28
HEAD_LEN = 0.22
MARGE = 0.05
HIP_X = 0.0      # x d'attache des pattes sur le corps (issu de la pose d'equilibre)
# Tete et cou alleges : etant en avant du corps, leur masse cree un bras de
# levier qui fait piquer la poule vers l'avant. Densites reduites pour limiter
# ce basculement (la tete surtout, plus eloignee du centre).
NECK_DENSITY = 0.05
HEAD_DENSITY = 0.15


def _leg_bones(side: str, hip_x: float):
    """Os d'une patte SIMPLIFIEE a 3 segments : cuisse -> mollet -> pied.

    Le tarse ('shank') a ete retire pour simplifier (moins d'articulations
    a coordonner). Le pied est desormais rattache directement au mollet.
    """
    # Pose d'EQUILIBRE (mesuree via debug_print puis recopiee ici) : positions
    # relatives au corps + angles d'apparition. La poule nait donc directement
    # accroupie et stable, sans secousse (les joints sont deja dans leurs limites).
    return [
        BoneDef(f'{side}_thigh', hip_x,         -0.12, WIDTH_BONE, THIGH_LEN, DENSITY_BONE, angle=-0.4),   # cuisse
        BoneDef(f'{side}_shin',  hip_x,         -0.30, WIDTH_BONE, SHIN_LEN,  DENSITY_BONE, angle=0.3),  # mollet
        # --- Segment retire (simplification 3 os) : le tarse ---
        # BoneDef(f'{side}_shank', hip_x, 0, WIDTH_BONE, SHANK_LEN, DENSITY_BONE),
        BoneDef(f'{side}_foot',  hip_x + 0.035, -0.52, WIDTH_BONE, FOOT_LEN, DENSITY_BONE, angle=0.6),   # pied
    ]


def _leg_muscles(side: str, hip_x: float):
    """Muscles d'une patte : memes limites que la patte arriere du renard."""
    return [
        MuscleDef(f'{side}_hip', 'body', f'{side}_thigh',
                  (hip_x, -WIDTH_BONE), (0, THIGH_LEN / 2 + MARGE),
                  -math.pi * 0.3, math.pi * 0.7, max_torque=3000, actuated=True),
        MuscleDef(f'{side}_knee', f'{side}_thigh', f'{side}_shin',
                  (0, -THIGH_LEN / 2 + MARGE), (0, SHIN_LEN / 2 + MARGE),
                  -math.pi * 0.7, 0, max_torque=3500, actuated=True),
        # --- Muscle retire (simplification 3 os) : cheville shin -> shank ---
        # MuscleDef(f'{side}_ankle', f'{side}_shin', f'{side}_shank',
        #           (0, -SHIN_LEN / 2 + MARGE), (0, SHANK_LEN / 2 + MARGE),
        #           -math.pi * 0.0, math.pi * 0.7, max_torque=3500, actuated=True),
        # Le pied est desormais rattache au MOLLET (shin), plus au tarse.
        MuscleDef(f'{side}_foot_joint', f'{side}_shin', f'{side}_foot',
                  (WIDTH_BONE, -(SHIN_LEN / 2 + MARGE)), (WIDTH_BONE, MARGE),
                  math.pi * 0.3, math.pi * 0.8, max_torque=1500, actuated=True),
    ]


def _build_skeleton() -> SkeletonDef:
    # Pose d'equilibre : le corps est legerement incline (0.11 rad), le cou et
    # la tete naissent dans leur orientation stabilisee (aucune secousse).
    bones = [
        BoneDef('body', 0.0, 0.0, BODY_LEN, WIDTH_BONE, DENSITY_BACK, angle=0),
        *_leg_bones('right', HIP_X),
        *_leg_bones('left', HIP_X),
        BoneDef('neck', 0.47, 0.24, WIDTH_BONE, NECK_LEN, NECK_DENSITY, angle=0),
        BoneDef('head', 0.52, 0.49, WIDTH_BONE, HEAD_LEN, HEAD_DENSITY, angle=0),
    ]

    # Les 8 muscles actionnes d'abord (patte droite puis gauche).
    muscles = [
        *_leg_muscles('right', HIP_X),
        *_leg_muscles('left', HIP_X),
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
    # Legerement agrandi (~8%) autour du centre de l'os body.
    body_outline = [
        (-0.43, 0.22),   # naissance de la queue
        (-0.16, 0.30),   # dos
        (0.16, 0.29),
        (0.41, 0.19),    # epaule vers le cou
        (0.51, 0.03),    # jabot haut
        (0.46, -0.18),   # poitrail
        (0.24, -0.31),   # dessous avant
        (-0.11, -0.32),  # ventre
        (-0.36, -0.23),  # arriere ventre
        (-0.49, -0.04),  # dessous de la queue
    ]
    wing = [
        (0.15, 0.15),
        (0.28, 0.02),
        (0.13, -0.14),
        (-0.15, -0.15),
        (-0.39, -0.05),
        (-0.15, 0.12),
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

    # ----- Pattes (3 os) : cuisse et mollet emplumes, pied jaune -----
    def leg_styles(side):
        return {
            f'{side}_thigh': LegStyle(hw_top=0.10, hw_bottom=0.06, color='plumage'),
            f'{side}_shin': LegStyle(hw_top=0.165, hw_bottom=0.065, color='plumage'),
            # f'{side}_shank': LegStyle(hw_top=0.05, hw_bottom=0.03, color='legs_y'),  # retire (3 os)
            f'{side}_foot': LegStyle(hw_top=0.025, hw_bottom=0.02, color='legs_y'),
        }

    legs = {**leg_styles('right'), **leg_styles('left')}
    leg_chains = [['right_thigh', 'right_shin', 'right_foot']]
    far_leg_chains = [['left_thigh', 'left_shin', 'left_foot']]

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
    spawn_y=1.08,  # hauteur du corps dans la pose d'equilibre (pieds au sol)
    has_legacy_textures=False,
)
