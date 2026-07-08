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
    raise ValueError(
        f"Animal inconnu : {name!r}. Valeurs supportees : 'fox', 'chicken'."
    )
