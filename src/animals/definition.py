# ============================================
# definition.py - Structures de definition d'un animal
# ============================================
# Un animal = un squelette physique (os + muscles Box2D) + une peau
# procedurale (polygones dessines par code, style low poly).
#
# Pour creer un nouvel animal : ecrire un module dans src/animals/
# (ex: fox.py) qui construit une AnimalDefinition. Aucun asset image
# n'est necessaire pour le rendu procedural.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vec2 = Tuple[float, float]
Color = Tuple[int, int, int]


# ============ SQUELETTE (physique Box2D) ============

@dataclass
class BoneDef:
    """Un os : boite Box2D positionnee relativement au point d'apparition."""
    name: str
    dx: float           # offset x par rapport au point d'apparition (metres)
    dy: float           # offset y
    width: float        # largeur de la boite Box2D
    height: float       # hauteur de la boite Box2D
    density: float = 0.5


@dataclass
class MuscleDef:
    """Un muscle : joint moteur revolute entre deux os.

    Les muscles actionnes (actuated=True) doivent etre declares EN PREMIER :
    l'IA controle les indices 0..N-1 de la liste quadruped.muscles.
    """
    name: str
    bone_a: str
    bone_b: str
    anchor_a: Vec2      # ancre locale sur bone_a
    anchor_b: Vec2      # ancre locale sur bone_b
    min_angle: float    # radians
    max_angle: float
    max_torque: float = 1000.0
    max_speed: float = 3.0
    actuated: bool = False


@dataclass
class SkeletonDef:
    bones: List[BoneDef]
    muscles: List[MuscleDef]
    root: str = "body"  # os principal (position, detection de chute)
    # False = les os de l'animal ne se collisionnent pas entre eux (necessaire
    # pour les bipedes dont les deux pattes se superposent en vue de profil).
    self_collide: bool = True


# ============ PEAU (rendu procedural low poly) ============

@dataclass
class Shape:
    """Une forme locale a un os : polygone (facettise) ou cercle.

    Les points sont exprimes dans le repere local de l'os porteur
    (pour la tete : repere "museau", +x vers le nez, +y vers le haut).
    """
    color: str                    # cle dans SkinSpec.palette
    points: List[Vec2] = field(default_factory=list)   # polygone
    kind: str = "poly"            # "poly" ou "circle"
    center: Vec2 = (0.0, 0.0)     # si kind == "circle"
    radius: float = 0.0
    facets: bool = True           # facettes low poly (variation de teinte)


@dataclass
class LegStyle:
    """Style d'un segment de patte : capsule effilee suivant l'os."""
    hw_top: float       # demi-largeur en haut du segment (metres)
    hw_bottom: float    # demi-largeur en bas
    color: str = "coat"


@dataclass
class TailSpec:
    """Queue purement esthetique : chaine de noeuds simulee (pas de Box2D).

    La chaine est ancree sur un os et suit ses mouvements avec inertie,
    ce qui donne un effet de fouet naturel quand l'animal bouge.
    """
    anchor_bone: str = "body"
    anchor_local: Vec2 = (0.0, 0.0)      # point d'ancrage local sur l'os
    segment_length: float = 0.15
    rest_angles_deg: List[float] = field(default_factory=list)  # angle de repos de chaque segment, relatif a l'os d'ancrage (0 = vers l'avant)
    half_widths: List[float] = field(default_factory=list)      # demi-largeur a chaque noeud (len = nb segments + 1)
    tip_color: str = "cream"
    tip_ratio: float = 0.3               # fraction finale coloree en tip_color
    color: str = "coat"
    stiffness: float = 30.0              # rappel vers la pose de repos
    damping: float = 3.5                 # amortissement (par seconde)
    gravity: float = 1.5                 # gravite apparente (m/s^2, vers le bas)


@dataclass
class EarSpec:
    """Oreille triangulaire montee sur ressort : reagit aux mouvements de la tete."""
    base_local: Vec2 = (0.0, 0.0)        # base de l'oreille (repere museau)
    points: List[Vec2] = field(default_factory=list)   # triangle relatif a la base
    inner_points: List[Vec2] = field(default_factory=list)  # interieur (optionnel)
    color: str = "coat"
    inner_color: str = "ear_inner"
    stiffness: float = 80.0
    damping: float = 9.0
    react_gain: float = 0.06             # sensibilite a la vitesse angulaire de la tete
    max_deflection: float = 0.6          # rad


@dataclass
class SkinSpec:
    """Description complete du rendu procedural d'un animal."""
    palette: Dict[str, Color]
    facet_jitter: float = 0.06           # amplitude de variation de teinte des facettes

    # Formes rigides attachees aux os (torse sur 'body', tete sur 'head'...).
    # Cle = nom de l'os porteur, valeur = liste de formes (ordre = ordre de dessin).
    body_shapes: List[Shape] = field(default_factory=list)
    head_shapes: List[Shape] = field(default_factory=list)

    # Cou : capsule dynamique entre le corps et la tete (suit l'os 'neck').
    neck_bone: str = "neck"
    neck_hw_base: float = 0.15
    neck_hw_top: float = 0.10
    neck_color: str = "coat"

    # Pattes : capsules suivant les os, cle = nom de l'os.
    legs: Dict[str, LegStyle] = field(default_factory=dict)
    # Chaines de pattes de devant (liste de listes de noms d'os).
    leg_chains: List[List[str]] = field(default_factory=list)
    # Chaines dessinees au fond, assombries et decalees. Un quadrupede en vue
    # de profil duplique ses 2 pattes physiques (leg_chains == far_leg_chains),
    # un bipede met sa patte gauche au fond et sa droite devant.
    far_leg_chains: List[List[str]] = field(default_factory=list)
    far_leg_darken: float = 0.68         # facteur de couleur des pattes du fond
    far_leg_offset: Vec2 = (0.07, 0.0)   # decalage visuel des pattes du fond (metres)

    tail: Optional[TailSpec] = None
    ears: List[EarSpec] = field(default_factory=list)


@dataclass
class AnimalDefinition:
    name: str
    skeleton: SkeletonDef
    skin: SkinSpec
    spawn_y: float = 3.0            # hauteur d'apparition (limite la chute au reset)
    has_legacy_textures: bool = False  # True si des PNG decoupes existent (renard)

    @property
    def num_actuated(self) -> int:
        """Nombre de muscles controles par l'IA (= taille de sortie du reseau)."""
        return sum(1 for m in self.skeleton.muscles if m.actuated)
