# ============================================
# physics.py - Gestion de la physique Box2D
# ============================================

from Box2D import b2World, b2PolygonShape, b2RevoluteJointDef
import math


class PhysicsWorld:
    """Gestionnaire du monde physique Box2D"""

    def __init__(self, gravity=(0, -10)):
        self.world = b2World(gravity=gravity, doSleep=True) # Do sleep pour ne pas calculer les objets qui ne bouge plus
        self.ground = None
        self.create_ground()

    def create_ground(self):
        """Crée le sol.

        Le sol est un UNIQUE corps statique : sa taille n'a aucun impact sur le
        coût de simulation (un rectangle de 2000 m coûte autant qu'un de 160 m).
        On le fait donc très large (demi-largeur 1000 -> terrain de 2000 m,
        centré sur l'origine, de -1000 à +1000 en x). L'animal apparaît à x=6
        et court vers +x : de quoi tenir de très longs épisodes et de longues
        visualisations sans jamais atteindre le bord. Rester sous ~1000 m garde
        la physique numériquement stable (positions en float32).
        """
        self.ground = self.world.CreateStaticBody(
            position=(0, -3.5),
            shapes=b2PolygonShape(box=(1000, 4))
        )
        self.ground.fixtures[0].friction = 0.8

    def step(self, time_step, vel_iterations=10, pos_iterations=10):
        """Avance la simulation d'un pas"""
        self.world.Step(time_step, vel_iterations, pos_iterations)


class Bone:
    """Représente un os du squelette"""

    def __init__(self, world, x, y, width, height, density=5.0, group_index=0, angle=0.0):
        self.body = world.CreateDynamicBody(
            position=(x, y),
            angle=angle
        )
        self.fixture = self.body.CreatePolygonFixture(
            box=(width / 2, height / 2),
            density=density,
            friction=0.5
        )
        # groupIndex négatif = les os d'un même animal ne se collisionnent pas
        # entre eux (indispensable aux bipèdes dont les pattes se croisent).
        if group_index:
            self.fixture.filterData.groupIndex = group_index
        self.width = width
        self.height = height


class Muscle:
    """Représente un muscle (joint moteur entre deux os)"""

    def __init__(self, world, body_a, body_b, anchor_a, anchor_b,
                 min_angle, max_angle, max_torque=1000, max_speed=3.0):

        joint_def = b2RevoluteJointDef(
            bodyA=body_a,
            bodyB=body_b,
            localAnchorA=anchor_a,
            localAnchorB=anchor_b,
            enableLimit=True,
            lowerAngle=min_angle,
            upperAngle=max_angle,
            enableMotor=True,
            maxMotorTorque=max_torque,
            motorSpeed=0
        )
        self.joint = world.CreateJoint(joint_def)
        self.target_speed = 0
        self.max_speed = max_speed
        self.body_a = body_a
        self.body_b = body_b
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b

    def contract(self, strength=1.0):
        """Contracter le muscle (flexion)"""
        self.target_speed = -self.max_speed * strength

    def extend(self, strength=1.0):
        """Étendre le muscle (extension)"""
        self.target_speed = self.max_speed * strength

    def relax(self):
        """Relâcher le muscle"""
        self.target_speed = 0

    def update(self):
        """Mettre à jour la vitesse du moteur"""
        self.joint.motorSpeed = self.target_speed

    def get_angle(self):
        """Retourne l'angle actuel du joint"""
        return self.joint.angle

    def get_speed(self):
        """Retourne la vitesse angulaire actuelle"""
        return self.joint.speed


class Quadruped:
    """Quadrupède générique construit depuis une AnimalDefinition.

    Les os, muscles et proportions viennent de la définition (src/animals/),
    plus rien n'est codé en dur ici. Par défaut : le renard (FOX).
    Les muscles actionnés par l'IA occupent les indices 0..num_actuated-1.
    """

    def __init__(self, physics_world, x=6, y=3, definition=None):
        if definition is None:
            # Import local pour éviter un import circulaire au chargement.
            from src.animals.fox import FOX
            definition = FOX

        self.definition = definition
        self.physics_world = physics_world
        world = physics_world.world

        # ----- Os -----
        group_index = 0 if definition.skeleton.self_collide else -1
        self.bones_by_name = {}
        for bd in definition.skeleton.bones:
            bone = Bone(world, x + bd.dx, y + bd.dy, bd.width, bd.height,
                        density=bd.density, group_index=group_index, angle=bd.angle)
            self.bones_by_name[bd.name] = bone
            # Accès direct type quadruped.front_thigh (compat overlay/replay).
            setattr(self, bd.name, bone)
        self.bones = list(self.bones_by_name.values())

        # Os racine (position de référence, détection de chute).
        self.body = self.bones_by_name[definition.skeleton.root]

        # ----- Muscles -----
        self.muscles = []
        self.muscles_by_name = {}
        for md in definition.skeleton.muscles:
            muscle = Muscle(
                world,
                self.bones_by_name[md.bone_a].body,
                self.bones_by_name[md.bone_b].body,
                md.anchor_a, md.anchor_b,
                md.min_angle, md.max_angle,
                max_torque=md.max_torque, max_speed=md.max_speed,
            )
            self.muscles.append(muscle)
            self.muscles_by_name[md.name] = muscle

        self.num_actuated = sum(1 for md in definition.skeleton.muscles if md.actuated)

    def control_muscles(self, muscle_index, action):
        """
        Contrôle un muscle spécifique
        muscle_index: 0, 1 ou 2
        action: 'contract', 'extend' ou 'relax'
        """
        if 0 <= muscle_index < len(self.muscles):
            if action == 'contract':
                self.muscles[muscle_index].contract(1.0)
            elif action == 'extend':
                self.muscles[muscle_index].extend(1.0)
            elif action == 'relax':
                self.muscles[muscle_index].relax()

    def set_muscle_activation(self, muscle_index, activation):
        """Contrôle CONTINU d'un muscle (pour l'IA neuroevolution).

        activation dans [-1, 1] :
            > 0  -> extension, l'amplitude module la vitesse du moteur
            < 0  -> contraction, idem
            ~ 0  -> relâché

        Contrairement à control_muscles (ternaire tout-ou-rien), l'intensité
        est proportionnelle, ce qui autorise des démarches beaucoup plus fluides.
        """
        # Idem control_muscles : seuls les muscles actionnes sont pilotables
        # (le cou/la tete restent hors de controle IA et humain).
        if 0 <= muscle_index < self.num_actuated:
            muscle = self.muscles[muscle_index]
            activation = max(-1.0, min(1.0, float(activation)))
            if activation > 0:
                muscle.extend(activation)
            elif activation < 0:
                muscle.contract(-activation)
            else:
                muscle.relax()

    def update(self):
        """Met à jour tous les muscles"""
        for muscle in self.muscles:
            muscle.update()

    def debug_print(self, time_s=None):
        """Affiche la position/angle de chaque os et l'angle de chaque muscle.

        Sert a trouver une pose d'equilibre : on lit ici les angles absolus
        des os (a recopier dans BoneDef(..., angle=...)) et les angles relatifs
        des joints (a comparer aux limites min/max des muscles).
        """
        tag = f" (t={time_s:.1f}s)" if time_s is not None else ""
        print(f"\n===== ETAT{tag} =====")
        print("  Os : position (x, y) | angle absolu")
        for name, bone in self.bones_by_name.items():
            p = bone.body.position
            a = bone.body.angle
            print(f"    {name:14s} pos=({p.x:6.2f},{p.y:6.2f})  "
                  f"angle={math.degrees(a):7.1f}deg ({a:+.3f} rad)")
        print("  Muscles : angle du joint")
        for name, m in self.muscles_by_name.items():
            a = m.get_angle()
            print(f"    {name:16s} = {math.degrees(a):7.1f}deg ({a:+.3f} rad)")

    def get_state(self):
        """Retourne l'état du quadrupède (pour l'IA plus tard)"""
        state = {
            'body_pos': self.body.body.position,
            'body_angle': self.body.body.angle,
            'body_velocity': self.body.body.linearVelocity,
            'muscle_angles': [m.get_angle() for m in self.muscles],
            'muscle_speeds': [m.get_speed() for m in self.muscles]
        }
        return state


    def is_upside_down(self, threshold=math.pi / 1.5):
        """
        Vérifie si le quadrupède est retourné (dos au sol)
        threshold: angle en radians (par défaut π/4 = 45°)
        Retourne True si l'angle du corps dépasse le seuil
        """
        angle = self.body.body.angle % (2 * math.pi)

        # Normaliser l'angle entre -π et π
        if angle > math.pi:
            angle -= 2 * math.pi

        # Le quadrupède est retourné si l'angle est proche de ±π (180°)
        return abs(abs(angle) - math.pi) < threshold
