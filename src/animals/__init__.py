# Package des definitions d'animaux (squelette physique + peau procedurale).
#
# Pour ajouter un animal : creer un module (ex: wolf.py) qui construit une
# AnimalDefinition, puis l'enregistrer dans get_animal() ci-dessous.
# La selection se fait via ANIMAL dans src/config.py.


def get_animal(name: str):
    """Retourne l'AnimalDefinition correspondant au nom (fr ou en accepte)."""
    key = name.strip().lower()
    if key in ('fox', 'renard'):
        from src.animals.fox import FOX
        return FOX
    if key in ('chicken', 'poule'):
        from src.animals.chicken import CHICKEN
        return CHICKEN
    if key in ('cat', 'chat'):
        from src.animals.cat import CAT
        return CAT
    if key in ('alien', 'insect', 'insecte'):
        # Squelette EVOLUTIF : celui-ci est la creature par defaut (genes nuls).
        # Pendant l'entrainement, chaque individu reconstruit le sien depuis ses
        # propres genes (voir evaluate.run_episode).
        from src.animals.alien import ALIEN
        return ALIEN
    raise ValueError(
        f"Animal inconnu : {name!r}. "
        "Valeurs supportees : 'fox', 'chicken', 'cat', 'alien'."
    )
