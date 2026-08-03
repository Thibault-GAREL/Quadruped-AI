# ============================================
# cat.py - Le chat (quadrupede au dos articule)
# ============================================
# Meme topologie de pattes que le renard, mais le tronc est coupe en DEUX os
# relies par un muscle de colonne vertebrale. C'est ce qui distingue un chat
# d'un renard en course : il plie et deplie le dos pour allonger sa foulee.
#
# Consequences a connaitre :
# - 9 muscles actionnes au lieu de 8, donc un reseau 25 -> 16 -> 9. Le chat
#   n'est PAS directement comparable au renard, qui a un tronc rigide.
# - `body_shapes` ne suit que l'os racine, la croupe est donc peinte via
#   `extra_shapes` (voir SkinSpec dans definition.py).
#
# Proportions : le renard a l'echelle 0.82, un chat etant plus petit et plus
# fin. Attention, une distance parcourue par le chat n'est pas comparable a
# celle du renard, ses pattes sont plus courtes.

import math

from src.animals.definition import (
    AnimalDefinition, BoneDef, MuscleDef, SkeletonDef,
    Shape, SkinSpec, LegStyle, TailSpec, EarSpec, BridgeSpec,
)

# ----- Proportions (renard x 0.82) -----
WIDTH_BONE = 0.045
BODY_SEG = 0.58        # longueur de CHAQUE moitie du tronc
HALF_SEG = BODY_SEG / 2
DENSITY_BONE = 0.5
THIGH_HEIGHT = 0.41
SHIN_HEIGHT_F = 0.25
SHIN_HEIGHT_B = 0.33
ANKLE_HEIGHT = 0.16
FOOT_HEIGHT = 0.16
NECK_HEIGHT = 0.33
MARGE = 0.04

# Plages du joint du pied. Son zero ne correspond PAS a un pied a plat : vu la
# geometrie des ancres, le pied repose a plat autour de +85 degres. La butee
# BASSE fixe donc la pose au repos (trop bas, le chat se met sur des echasses),
# la butee HAUTE donne la course disponible. L'avant et l'arriere ne se reglent
# pas pareil, le jarret arriere demandant nettement plus de course.
FOOT_MIN_F = math.pi * 0.12   # +22 deg, l'avant se pose vers +67 deg
FOOT_MAX_F = math.pi * 0.72   # +130 deg, soit 108 deg de course
FOOT_MIN_B = math.pi * 0.42   # +76 deg, sous cette valeur l'arriere se dresse
FOOT_MAX_B = math.pi * 0.72   # +130 deg, l'arriere se pose alors bien a plat


def _leg_bones(prefix: str, hip_x: float, shin_h: float):
    """Les 4 os d'une patte (cuisse, mollet, cheville, pied)."""
    return [
        BoneDef(f'{prefix}_thigh', hip_x, -0.41, WIDTH_BONE, THIGH_HEIGHT, DENSITY_BONE),
        BoneDef(f'{prefix}_shin', hip_x, -1.07, WIDTH_BONE, shin_h, DENSITY_BONE),
        BoneDef(f'{prefix}_ankle', hip_x, -1.15, WIDTH_BONE, ANKLE_HEIGHT, DENSITY_BONE),
        BoneDef(f'{prefix}_foot', hip_x - 0.08, -1.23, WIDTH_BONE, FOOT_HEIGHT, DENSITY_BONE),
    ]


def _leg_muscles(prefix: str, torso: str, hip_local_x: float, shin_h: float,
                 hip_min: float, hip_max: float, knee_min: float, knee_max: float,
                 ankle_min: float, ankle_max: float,
                 foot_min: float, foot_max: float):
    """Les 4 muscles ACTIONNES d'une patte, dans l'ordre hanche, genou, cheville, pied."""
    return [
        MuscleDef(f'{prefix}_hip', torso, f'{prefix}_thigh',
                  (hip_local_x, -WIDTH_BONE), (0, THIGH_HEIGHT / 2 + MARGE),
                  hip_min, hip_max, max_torque=3300, actuated=True),
        MuscleDef(f'{prefix}_knee', f'{prefix}_thigh', f'{prefix}_shin',
                  (0, -THIGH_HEIGHT / 2 + MARGE), (0, shin_h / 2 + MARGE),
                  knee_min, knee_max, max_torque=4100, actuated=True),
        MuscleDef(f'{prefix}_ankle_joint', f'{prefix}_shin', f'{prefix}_ankle',
                  (WIDTH_BONE, -(shin_h / 2 + MARGE)), (WIDTH_BONE, ANKLE_HEIGHT / 2 + MARGE),
                  ankle_min, ankle_max, max_torque=4100, actuated=True),
        # ATTENTION au zero de ce joint : il ne correspond PAS a un pied a plat.
        # Vu la geometrie des ancres, le pied repose a plat autour de +85 degres,
        # et un joint autorise a descendre sous +22 se met a la verticale, le chat
        # se retrouve sur des echasses. La butee BASSE fixe donc la pose au repos,
        # la butee HAUTE doit monter jusqu'a +130 pour que le pied ARRIERE se pose
        # lui aussi a plat (le jarret lui demande plus de course qu'a l'avant).
        # Mesure a l'appui : +22 a +130 pose les 4 pieds a +80 et +85 degres.
        MuscleDef(f'{prefix}_foot_joint', f'{prefix}_ankle', f'{prefix}_foot',
                  (WIDTH_BONE, -(ANKLE_HEIGHT / 2 + MARGE)), (WIDTH_BONE, MARGE),
                  foot_min, foot_max, max_torque=1600, actuated=True),
    ]


def _build_skeleton() -> SkeletonDef:
    # Les pattes gauche et droite d'une meme paire partent d'un x tres proche
    # mais pas identique : une symetrie parfaite donne des contacts degeneres
    # et une physique moins stable. L'ecart de 3 cm ne se voit pas de profil.
    bones = [
        # Tronc en deux segments. body_front est la racine : il porte le cou,
        # et la distance parcourue se mesure donc a l'avant de l'animal.
        BoneDef('body_front', 0.29, 0.0, BODY_SEG, WIDTH_BONE, DENSITY_BONE),
        BoneDef('body_back', -0.29, 0.0, BODY_SEG, WIDTH_BONE, DENSITY_BONE),

        *_leg_bones('front_right', 0.535, SHIN_HEIGHT_F),
        *_leg_bones('front_left', 0.505, SHIN_HEIGHT_F),
        *_leg_bones('back_right', -0.505, SHIN_HEIGHT_B),
        *_leg_bones('back_left', -0.535, SHIN_HEIGHT_B),

        BoneDef('neck', 0.62, 0.08, WIDTH_BONE, NECK_HEIGHT, DENSITY_BONE),
        BoneDef('head', 0.70, 0.08, WIDTH_BONE, NECK_HEIGHT, DENSITY_BONE),
    ]

    # IMPORTANT : les 17 muscles actionnes par l'IA d'abord (indices 0..16).
    # Ordre : avant droit, avant gauche, arriere droit, arriere gauche, colonne.
    muscles = [
        # Plages articulaires. Le genou et la cheville AVANT gardent 0 comme
        # minimum : c'est la patte tendue, la position d'appui qui porte le
        # poids. Leur donner une marge negative laisse l'avant-train s'effondrer
        # sous son propre poids (teste : le chat pique du nez et sa tete touche
        # le sol). Les pattes ARRIERE, elles, travaillent en flexion, une petite
        # marge de l'autre cote leur va bien.
        *_leg_muscles('front_right', 'body_front', 0.20, SHIN_HEIGHT_F,
                      -math.pi * 0.45, math.pi * 0.14, 0, math.pi * 0.78,
                      0, math.pi * 0.44, FOOT_MIN_F, FOOT_MAX_F),
        *_leg_muscles('front_left', 'body_front', 0.17, SHIN_HEIGHT_F,
                      -math.pi * 0.45, math.pi * 0.14, 0, math.pi * 0.78,
                      0, math.pi * 0.44, FOOT_MIN_F, FOOT_MAX_F),
        *_leg_muscles('back_right', 'body_back', -0.17, SHIN_HEIGHT_B,
                      -math.pi * 0.34, math.pi * 0.38, -math.pi * 0.72, math.pi * 0.08,
                      -math.pi * 0.72, math.pi * 0.08, FOOT_MIN_B, FOOT_MAX_B),
        *_leg_muscles('back_left', 'body_back', -0.20, SHIN_HEIGHT_B,
                      -math.pi * 0.34, math.pi * 0.38, -math.pi * 0.72, math.pi * 0.08,
                      -math.pi * 0.72, math.pi * 0.08, FOOT_MIN_B, FOOT_MAX_B),

        # 17e muscle : la colonne vertebrale. +/- 32 degres, l'amplitude d'un chat
        # au galop qui ramasse puis detend son dos. A +/- 20 degres l'articulation
        # ne servait presque a rien, au dela de 40 le chat se plie en deux et se
        # bloque. Le couple est le plus eleve du squelette, ce joint porte tout
        # le tronc.
        MuscleDef('spine', 'body_back', 'body_front',
                  (HALF_SEG, 0), (-HALF_SEG, 0),
                  -math.pi * 0.18, math.pi * 0.18, max_torque=6000, actuated=True),

        # Cou et tete : joints figes (non actionnes), comme les autres animaux.
        # Angles PLUS FERMES que ceux du renard : le chat porte la tete haute et
        # le museau a l'horizontale, un port de tete plongeant lui donne tout de
        # suite un air de canide qui renifle le sol.
        MuscleDef('neck_joint', 'body_front', 'neck',
                  (HALF_SEG + MARGE, WIDTH_BONE), (0, NECK_HEIGHT / 2),
                  math.pi * 0.70, math.pi * 0.70, max_torque=40),
        MuscleDef('head_joint', 'neck', 'head',
                  (WIDTH_BONE, -(NECK_HEIGHT / 2 + MARGE)), (WIDTH_BONE, -WIDTH_BONE),
                  math.pi * 0.88, math.pi * 0.88, max_torque=40),
    ]

    # self_collide=False : les pattes gauche et droite se superposent en vue de
    # profil, sans ca elles se repoussent en permanence (meme raison que la poule).
    return SkeletonDef(
        bones=bones, muscles=muscles, root='body_front', self_collide=False,
        foot_bones=['front_right_foot', 'front_left_foot',
                    'back_right_foot', 'back_left_foot'],
    )


def _build_skin() -> SkinSpec:
    palette = {
        "coat": (112, 122, 134),        # gris ardoise du dos
        "coat_dark": (74, 82, 93),      # gris sombre (ombres, croupe, oreille du fond)
        "white": (240, 240, 236),       # ventre, poitrail, museau, pattes
        "nose": (208, 145, 152),        # truffe rose
        "eye": (96, 148, 96),           # oeil vert
        "ear_inner": (198, 150, 155),   # interieur d'oreille rose
        "sun": (146, 158, 172),         # eclairage du dessus du dos
    }

    # ----- Croupe : repere de body_back (+x vers l'avant, x dans -0.29..0.29) -----
    # Les formes debordent LARGEMENT de l'os pour donner du volume : colle aux os
    # le chat parait squelettique, un felin est un tube de muscles autour d'une
    # colonne fine. Elles se chevauchent a la jonction pour qu'aucun trou
    # n'apparaisse quand le dos plie.
    # Le bord avant de la croupe s'enfonce LOIN sous l'avant-train (jusqu'a
    # +0.56 au lieu de +0.34). Bord a bord, les deux moities laissent voir une
    # cassure nette sur le dos, le flanc et le ventre des que la colonne plie.
    # Avec un large recouvrement, l'avant-train glisse par dessus la croupe et
    # la silhouette reste continue quel que soit l'angle du dos.
    # Contour dense : le rendu facettise en eventail autour du centroide, donc
    # chaque sommet devient une arete. Peu de points donnent une silhouette
    # anguleuse, des points intermediaires sur les courbes (croupe, hanche,
    # ventre) suffisent a l'arrondir sans quitter le style low poly.
    rump_outline = [
        (-0.46, 0.12),   # racine de la queue
        (-0.435, 0.23),
        (-0.38, 0.32),   # haut de la croupe
        (-0.28, 0.362),
        (-0.16, 0.377),
        (-0.05, 0.38),   # dos arriere
        (0.12, 0.377),
        (0.30, 0.37),    # dos, vers la jonction
        (0.40, 0.35),    # s'enfonce sous l'avant-train
        (0.40, -0.31),
        (0.30, -0.33),
        (0.15, -0.337),
        (-0.02, -0.34),  # ventre arriere
        (-0.18, -0.327),
        (-0.32, -0.30),  # bas de cuisse
        (-0.425, -0.205),
        (-0.48, -0.06),  # arriere de cuisse
    ]
    rump_sun = [
        (-0.38, 0.28), (-0.24, 0.325), (-0.05, 0.35), (0.14, 0.347),
        (0.30, 0.34), (0.40, 0.315),
        (0.40, 0.175), (0.30, 0.19), (0.13, 0.198), (-0.05, 0.20),
        (-0.22, 0.175), (-0.36, 0.14),
    ]
    rump_belly = [
        (0.40, -0.285), (0.30, -0.31), (0.15, -0.322), (-0.02, -0.33),
        (-0.18, -0.312), (-0.30, -0.28),
        (-0.275, -0.235), (-0.25, -0.188), (-0.02, -0.215), (0.15, -0.211),
        (0.30, -0.206), (0.40, -0.194),
    ]

    # ----- Avant-train : repere de body_front (l'os racine) -----
    # Symetrique de la croupe : l'avant-train deborde loin vers l'arriere et
    # recouvre la jonction. C'est lui qui est dessine en DERNIER (os racine),
    # donc c'est son bord arriere qui doit rester dans la silhouette.
    chest_outline = [
        (-0.40, 0.35),   # deborde sur la croupe
        (-0.30, 0.37),
        (-0.13, 0.377),
        (0.06, 0.38),    # dos avant
        (0.20, 0.357),
        (0.32, 0.31),    # garrot
        (0.395, 0.232),  # descente adoucie vers le cou
        (0.45, 0.13),    # base du cou
        (0.468, 0.028),
        (0.47, -0.08),   # poitrail haut
        (0.435, -0.197),
        (0.36, -0.31),   # poitrail bas
        (0.21, -0.341),
        (0.04, -0.35),   # dessous de poitrine
        (-0.14, -0.343),
        (-0.30, -0.33),
        (-0.40, -0.31),  # jonction basse, debordante
    ]
    chest_sun = [
        (-0.40, 0.315), (-0.30, 0.34), (-0.13, 0.353), (0.06, 0.36),
        (0.19, 0.335), (0.30, 0.29), (0.375, 0.213), (0.43, 0.12),
        (0.33, 0.115), (0.19, 0.163), (0.06, 0.21), (-0.13, 0.207),
        (-0.30, 0.20), (-0.40, 0.185),
    ]
    chest_belly = [
        (0.36, -0.29), (0.21, -0.318), (0.04, -0.33), (-0.14, -0.322),
        (-0.30, -0.31), (-0.40, -0.295),
        (-0.40, -0.198), (-0.30, -0.206), (-0.14, -0.218), (0.04, -0.226),
        (0.22, -0.196), (0.38, -0.162),
    ]

    extra_shapes = {
        'body_back': [
            Shape('coat', points=rump_outline),
            Shape('sun', points=rump_sun),
            Shape('white', points=rump_belly),
        ],
    }
    body_shapes = [
        Shape('coat', points=chest_outline),
        Shape('sun', points=chest_sun),
        Shape('white', points=chest_belly),
    ]

    # ----- Tete : repere "museau" (+x vers le nez, +y vers le haut) -----
    # Crane plus rond et museau plus court que le renard, c'est ce qui fait
    # lire "chat" plutot que "canide" en silhouette.
    # Recette du "mignon" : un crane rond et large, un museau court, et un oeil
    # gros place BAS sur la face. Ce sont les proportions d'un chaton, un museau
    # long et un oeil haut donnent immediatement une tete d'adulte severe.
    head_shapes = [
        Shape('coat', points=[
            (-0.30, 0.06),   # arriere du crane
            (-0.23, 0.24),   # haut arriere
            (-0.03, 0.30),   # sommet, entre les oreilles
            (0.18, 0.25),    # front bombe
            (0.31, 0.09),    # haut du museau (court)
            (0.36, -0.06),   # bout du nez
            (0.26, -0.20),   # dessous du museau
            (0.04, -0.27),   # bajoue pleine
            (-0.19, -0.26),  # joue
            (-0.31, -0.10),  # arriere bas
        ]),
        # Museau et menton blancs, bien larges pour arrondir le bas du visage.
        Shape('white', points=[
            (0.36, -0.06), (0.26, -0.19), (0.04, -0.25),
            # Bord haut du museau maintenu SOUS l'oeil : s'il remonte davantage
            # il vient lecher la paupiere basse et le regard perd sa lisibilite.
            (-0.13, -0.19), (-0.08, -0.078), (0.13, -0.062), (0.30, -0.038),
        ], facets=False),
        Shape('nose', kind='circle', center=(0.345, -0.056), radius=0.040, facets=False),
        # Oeil en AMANDE, pas un rond : de profil on ne voit qu'une partie du
        # globe, et le coin externe fuit vers l'arriere du crane. Un cercle
        # parfait avec sa pupille au centre se lit comme un oeil vu de face,
        # l'animal semble alors fixer le spectateur au lieu de regarder devant.
        Shape('eye', points=[
            (0.008, 0.022),   # coin arriere, effile
            (0.046, 0.086),   # paupiere haute
            (0.112, 0.090),
            (0.166, 0.030),   # coin avant, vers le museau
            (0.128, -0.046),
            (0.052, -0.044),  # paupiere basse
        ], facets=False),
        # Pupille fendue decalee vers l'AVANT du globe : c'est ce decalage qui
        # dirige le regard vers l'horizon plutot que vers l'observateur. Elle
        # reste ETROITE, une fente large redonne un oeil rond vu de face.
        Shape('coat_dark', points=[
            (0.114, 0.070), (0.130, 0.022), (0.114, -0.030), (0.098, 0.022),
        ], facets=False),
        # Reflet devant la pupille, du cote de la lumiere, plus un appoint discret.
        Shape('white', kind='circle', center=(0.140, 0.046), radius=0.015, facets=False),
        Shape('white', kind='circle', center=(0.058, -0.014), radius=0.011, facets=False),
    ]

    # ----- Oreilles : grandes et triangulaires, posees haut sur le crane rond -----
    ear_shape = [(-0.125, 0.0), (-0.013, 0.338), (0.113, 0.013)]
    ear_inner = [(-0.069, 0.025), (-0.013, 0.238), (0.056, 0.038)]
    ears = [
        EarSpec(base_local=(0.025, 0.213), points=ear_shape, inner_points=[],
                color='coat_dark'),
        EarSpec(base_local=(-0.19, 0.20), points=ear_shape, inner_points=ear_inner,
                color='coat'),
    ]

    # ----- Pattes : charnues en haut, fines au bas, chaussettes blanches -----
    # Une cuisse trop fine est ce qui donne l'air squelettique. Le haut de patte
    # porte le muscle, seul le bas (cheville et pied) reste sec.
    def leg_style(prefix, thigh_top, thigh_bottom, shin_top, shin_bottom):
        return {
            f'{prefix}_thigh': LegStyle(hw_top=thigh_top, hw_bottom=thigh_bottom, color='coat'),
            f'{prefix}_shin': LegStyle(hw_top=shin_top, hw_bottom=shin_bottom, color='coat'),
            f'{prefix}_ankle': LegStyle(hw_top=0.052, hw_bottom=0.042, color='white'),
            f'{prefix}_foot': LegStyle(hw_top=0.046, hw_bottom=0.038, color='white'),
        }

    legs = {
        **leg_style('front_right', 0.160, 0.088, 0.082, 0.056),
        **leg_style('front_left', 0.160, 0.088, 0.082, 0.056),
        # Cuisses arriere plus epaisses : c'est la que le felin a sa masse.
        **leg_style('back_right', 0.185, 0.100, 0.092, 0.062),
        **leg_style('back_left', 0.185, 0.100, 0.092, 0.062),
    }

    def chain(prefix):
        return [f'{prefix}_thigh', f'{prefix}_shin', f'{prefix}_ankle', f'{prefix}_foot']

    # Les 4 pattes sont REELLES (une chaine physique chacune), plus aucune n'est
    # une copie decorative. Le cote gauche part au fond, assombri et decale.
    far_leg_chains = [chain('back_left'), chain('front_left')]
    leg_chains = [chain('back_right')]
    front_leg_chains = [chain('front_right')]

    # ----- Queue : longue, ancree sur la croupe (donc sur body_back) -----
    # Une queue trop molle et d'epaisseur quasi constante donne un ruban de
    # tissu et non un appendice vivant. Trois reglages evitent ca : une base
    # nettement plus epaisse que la pointe (le lecteur voit une structure qui
    # part du corps), une raideur haute pour qu'elle garde sa courbe au lieu de
    # pendre, et un amortissement fort pour tuer les oscillations parasites.
    tail = TailSpec(
        anchor_bone='body_back',
        anchor_local=(-0.38, 0.10),
        segment_length=0.17,
        # Courbe DOUCE et etalee : des angles qui tournent trop vite enroulent la
        # queue sur elle-meme et la font lire comme un moignon. Elle part vers
        # l'arriere puis se releve progressivement, sans jamais revenir sur le corps.
        rest_angles_deg=[188, 179, 168, 154, 138, 122],
        half_widths=[0.100, 0.092, 0.082, 0.070, 0.057, 0.043, 0.027],
        tip_color='white',
        tip_ratio=0.26,
        color='coat',
        stiffness=52.0,
        damping=6.0,
        gravity=1.1,
    )

    # ----- Liaison du tronc : la piece qui rend la jonction invisible -----
    # Les deux moities du torse sont des formes RIGIDES, chacune sur son os.
    # Quand la colonne plie, leurs bords divergent et une couture apparait sur
    # le dos, le flanc et le ventre a la fois. Ces trois ponts sont tendus entre
    # les deux os et recalcules a chaque frame, donc ils suivent la flexion.
    #
    # Ils sont dessines EN DESSOUS des deux moities (voir procedural_skin) : ils
    # comblent le pli sans pouvoir deborder. Les peindre par dessus donnerait une
    # boite visible sur le dos, car un pont est un quadrilatere plat tendu entre
    # deux points de l'AXE des os, alors que le dos est bombe. Etant recouverts,
    # ils peuvent etre genereusement dimensionnes.
    # Un pont par couleur, dans le meme ordre que le torse : robe, lumiere, ventre.
    ANCHOR_BACK, ANCHOR_FRONT = (0.20, 0.0), (-0.20, 0.0)
    bridges = [
        BridgeSpec('body_back', 'body_front', ANCHOR_BACK, ANCHOR_FRONT,
                   top_a=0.42, top_b=0.42, bottom_a=0.38, bottom_b=0.38,
                   color='coat'),
        # Bande de lumiere : bottom negatif, elle ne descend donc pas jusqu'a
        # la ligne d'ancrage et reste collee au haut du dos.
        BridgeSpec('body_back', 'body_front', ANCHOR_BACK, ANCHOR_FRONT,
                   top_a=0.40, top_b=0.40, bottom_a=-0.20, bottom_b=-0.20,
                   color='sun'),
        # Ventre : symetrique, top negatif pour rester sous la ligne.
        BridgeSpec('body_back', 'body_front', ANCHOR_BACK, ANCHOR_FRONT,
                   top_a=-0.215, top_b=-0.215, bottom_a=0.36, bottom_b=0.36,
                   color='white'),
    ]

    return SkinSpec(
        palette=palette,
        facet_jitter=0.05,
        body_shapes=body_shapes,
        extra_shapes=extra_shapes,
        bridges=bridges,
        head_shapes=head_shapes,
        neck_bone='neck',
        neck_hw_base=0.22,
        neck_hw_top=0.17,
        neck_color='coat',
        legs=legs,
        leg_chains=leg_chains,
        front_leg_chains=front_leg_chains,
        far_leg_chains=far_leg_chains,
        far_leg_darken=0.68,
        far_leg_offset=(0.06, 0.0),
        tail=tail,
        ears=ears,
    )


CAT = AnimalDefinition(
    name='cat',
    skeleton=_build_skeleton(),
    skin=_build_skin(),
    spawn_y=2.6,   # plus bas que le renard, le chat est plus petit
)
