# ============================================
# alien.py - Creature alien dont le CORPS evolue
# ============================================
# Contrairement aux autres animaux, dont le squelette est ecrit a la main, cette
# creature construit le sien a partir de GENES. Le meme algorithme genetique fait
# donc evoluer le corps et le cerveau en meme temps, ce qui est le principe des
# creatures de Karl Sims.
#
# ---- Comment une topologie variable tient dans un genome de taille FIXE ----
#
# Un GA classique croise deux vecteurs de meme longueur. Si le nombre de muscles
# variait librement, deux individus n'auraient plus des genomes comparables et le
# croisement perdrait tout sens. Trois choix evitent ce probleme :
#
# 1. Le reseau est dimensionne pour le MAXIMUM de muscles (MAX_MUSCLES). Une
#    creature qui en a moins laisse les sorties du haut inutilisees, ce que le
#    moteur encaisse deja (set_muscle_activation ignore les indices hors bornes,
#    et build_input complete la proprioception par des zeros).
# 2. Les paires de pattes actives sont TOUJOURS les premieres, donc l'indice d'un
#    muscle ne depend pas du nombre de pattes. Sans cela, perdre une paire
#    decalerait tous les muscles suivants et le reseau devrait tout reapprendre.
# 3. Chaque patte garde TOUJOURS le meme nombre de segments, mais un segment peut
#    s'atrophier jusqu'a une longueur quasi nulle. La morphologie varie donc sans
#    jamais changer le nombre de muscles d'une patte.
#
# Genome complet : [MORPHO_GENES genes de corps | poids du reseau], longueur fixe.

import math
from typing import List

from src.animals.definition import (
    AnimalDefinition, BoneDef, MuscleDef, SkeletonDef,
    Shape, SkinSpec, LegStyle,
)

# ----- Bornes de la morphologie -----
MAX_PAIRS = 4          # jusqu'a 8 pattes
SEG_PER_LEG = 3        # segments par patte, constant (l'atrophie fait varier la forme)
MAX_MUSCLES = MAX_PAIRS * 2 * SEG_PER_LEG   # 24

GENES_GLOBAL = 3
GENES_PER_PAIR = 8
MORPHO_GENES = GENES_GLOBAL + MAX_PAIRS * GENES_PER_PAIR   # 35

WIDTH_BONE = 0.05
DENSITY_BONE = 0.5
MARGE = 0.04


def _map(gene: float, lo: float, hi: float) -> float:
    """Ramene un gene (issu d'une gaussienne centree) dans [lo, hi].

    tanh sature en douceur : une mutation forte ne fait pas exploser la valeur,
    elle la pousse simplement vers la borne. Un clip brutal creerait au contraire
    des plateaux ou la mutation n'a plus aucun effet visible.
    """
    return lo + (hi - lo) * 0.5 * (math.tanh(float(gene)) + 1.0)


def _pair_genes(genes, index: int):
    """Les 8 genes de la paire de pattes `index`, decodes en grandeurs physiques."""
    base = GENES_GLOBAL + index * GENES_PER_PAIR
    g = genes[base:base + GENES_PER_PAIR]
    return {
        'attach': _map(g[0], -0.42, 0.42),      # position le long du corps
        'seg': [
            _map(g[1], 0.20, 0.60),             # cuisse
            _map(g[2], 0.15, 0.55),             # tibia
            _map(g[3], 0.05, 0.45),             # tarse, peut s'atrophier
        ],
        'spread': _map(g[4], -0.55, 0.55),      # inclinaison d'implantation
        'lim_lo': _map(g[5], -0.80, -0.10),     # butee basse (rad / pi)
        'lim_hi': _map(g[6], 0.10, 0.80),       # butee haute
        'thick': _map(g[7], 0.030, 0.095),      # epaisseur visuelle
    }


# Le gene de topologie est AMPLIFIE avant saturation. Les genes sortent d'une
# gaussienne serree (init_std = 0.5), et tanh les ramenerait tous vers le centre
# de la plage : toutes les creatures naitraient avec le meme nombre de pattes et
# l'evolution n'aurait rien a choisir. Ce facteur etale la distribution sur les
# trois valeurs possibles.
TOPOLOGY_GAIN = 4.0


def decode(genes) -> dict:
    """Traduit le vecteur de genes en description lisible de la creature."""
    n_pairs = int(round(_map(genes[0] * TOPOLOGY_GAIN, 2.0, float(MAX_PAIRS))))
    n_pairs = max(2, min(MAX_PAIRS, n_pairs))
    return {
        'n_pairs': n_pairs,
        'body_len': _map(genes[1], 0.80, 1.80),
        'body_thick': _map(genes[2], 0.10, 0.26),
        'pairs': [_pair_genes(genes, i) for i in range(n_pairs)],
    }


# ============ SQUELETTE ============

def _build_skeleton(shape: dict) -> SkeletonDef:
    body_len = shape['body_len']
    bones = [BoneDef('body', 0.0, 0.0, body_len, WIDTH_BONE, DENSITY_BONE)]
    muscles: List[MuscleDef] = []
    feet: List[str] = []

    for p, pair in enumerate(shape['pairs']):
        attach_x = pair['attach'] * body_len
        for side, offset in (('r', 0.015), ('l', -0.015)):
            prefix = f'p{p}{side}'
            # Les segments descendent en cascade sous le point d'attache.
            depth = 0.0
            parent = 'body'
            for s, seg_len in enumerate(pair['seg']):
                depth += seg_len
                bones.append(BoneDef(
                    f'{prefix}_s{s}', attach_x + offset, -depth + seg_len / 2,
                    WIDTH_BONE, seg_len, DENSITY_BONE,
                    angle=pair['spread'] if s == 0 else 0.0,
                ))
                anchor_parent = ((attach_x, -WIDTH_BONE) if parent == 'body'
                                 else (0, -pair['seg'][s - 1] / 2 + MARGE))
                muscles.append(MuscleDef(
                    f'{prefix}_m{s}', parent, f'{prefix}_s{s}',
                    anchor_parent, (0, seg_len / 2 + MARGE),
                    math.pi * pair['lim_lo'], math.pi * pair['lim_hi'],
                    max_torque=2600, actuated=True,
                ))
                parent = f'{prefix}_s{s}'
            feet.append(f'{prefix}_s{SEG_PER_LEG - 1}')

    # self_collide=False : les pattes gauche et droite se superposent en vue de
    # profil, comme chez la poule et le chat.
    return SkeletonDef(bones=bones, muscles=muscles, root='body',
                       self_collide=False, foot_bones=feet)


# ============ PEAU (generee, car la morphologie n'est pas connue d'avance) ============

def _build_skin(shape: dict) -> SkinSpec:
    palette = {
        "chitin": (74, 92, 78),        # carapace vert sombre
        "chitin_dark": (46, 58, 50),   # creux entre les plaques
        "glow": (128, 214, 148),       # nervures luminescentes
        "membrane": (150, 128, 176),   # ventre violace
        "eye": (226, 240, 140),        # oeil compose, jaune acide
        "eye_dark": (40, 52, 42),
    }

    half = shape['body_len'] / 2
    t = shape['body_thick']

    # Carapace : une suite de plaques, dessinees comme un seul polygone dont le
    # dos ondule. C'est ce profil segmente qui fait lire "insecte" plutot que
    # "mammifere", bien plus que la couleur.
    top, bottom = [], []
    plates = 5
    for i in range(plates + 1):
        f = i / plates
        x = -half + f * shape['body_len']
        # Corps fusele : epais au milieu, effile aux deux bouts.
        taper = math.sin(math.pi * (0.18 + 0.64 * f))
        wave = 0.018 * (1 if i % 2 else -1)     # ondulation des plaques
        top.append((x, t * taper + wave))
        bottom.append((x, -t * taper * 0.82))
    carapace = top + list(reversed(bottom))

    glow_line = [(x, y * 0.55) for x, y in top]
    glow_line += [(x, y * 0.30) for x, y in reversed(top)]

    belly = [(x, y) for x, y in bottom]
    belly += [(x, y * 0.45) for x, y in reversed(bottom)]

    body_shapes = [
        Shape('chitin', points=carapace),
        Shape('glow', points=glow_line, facets=False),
        Shape('membrane', points=belly, facets=False),
        # Oeil compose unique, place a l'avant : signature alien immediate.
        Shape('eye', kind='circle', center=(half * 0.78, t * 0.42),
              radius=t * 0.46, facets=False),
        Shape('eye_dark', kind='circle', center=(half * 0.84, t * 0.50),
              radius=t * 0.17, facets=False),
    ]

    # Pattes : capsules fines et effilees, tres insectoides.
    legs, right_chains, left_chains = {}, [], []
    for p, pair in enumerate(shape['pairs']):
        th = pair['thick']
        for side in ('r', 'l'):
            prefix = f'p{p}{side}'
            chain = []
            for s in range(SEG_PER_LEG):
                # Chaque segment plus fin que le precedent : effet patte d'insecte.
                k = 1.0 - 0.26 * s
                legs[f'{prefix}_s{s}'] = LegStyle(
                    hw_top=th * k, hw_bottom=th * k * 0.62,
                    color='chitin' if s < SEG_PER_LEG - 1 else 'chitin_dark',
                )
                chain.append(f'{prefix}_s{s}')
            (right_chains if side == 'r' else left_chains).append(chain)

    return SkinSpec(
        palette=palette,
        facet_jitter=0.07,
        body_shapes=body_shapes,
        neck_bone='',          # ni cou ni tete : le corps porte l'oeil
        legs=legs,
        leg_chains=left_chains,        # cote gauche derriere le corps
        front_leg_chains=right_chains,  # cote droit devant
        far_leg_chains=left_chains,
        far_leg_darken=0.62,
        far_leg_offset=(0.05, 0.0),
    )


# ============ CONSTRUCTION DEPUIS UN GENOME ============

def build_alien(genes) -> AnimalDefinition:
    """Construit la creature decrite par `genes` (les MORPHO_GENES premiers)."""
    shape = decode(genes)
    skeleton = _build_skeleton(shape)

    # Hauteur d'apparition : la patte la plus longue, plus une marge. Trop bas la
    # creature nait dans le sol, trop haut elle passe ses premieres frames en
    # chute libre et le debut de l'episode ne mesure plus sa demarche.
    longest = max(sum(pair['seg']) for pair in shape['pairs'])
    return AnimalDefinition(
        name='alien',
        skeleton=skeleton,
        skin=_build_skin(shape),
        spawn_y=longest + 0.45,
    )


# Creature par defaut : genes nuls, donc toutes les grandeurs au MILIEU de leur
# plage (tanh(0) = 0). Sert a main.py, replay.py et a l'inspection du squelette,
# l'entrainement remplacant ces genes par ceux de chaque individu.
ALIEN = build_alien([0.0] * MORPHO_GENES)
