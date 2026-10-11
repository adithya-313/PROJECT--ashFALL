"""
WHAT: Demographic lifestyle persona definitions and random persona sampling.
WHY: Assigns behavioral profiles (spending tier, look-alike characteristics) to people.
TERM: persona: A demographic behavioral archetype governing spending magnitude and inbound payment habits.
TERM: mule look-alike: An innocent person (such as a tea stall owner or gig worker) whose legitimate routine
      receives dozens of small incoming transfers, superficially mimicking a money mule.
"""

from dataclasses import dataclass
import numpy as np

from world.placeholders import PERSONA_MIX


@dataclass(frozen=True)
class Persona:
    """
    WHAT: Represents a distinct socioeconomic behavioral template.
    WHY: Defines spending tiers ('low', 'mid', 'high') and flags whether normal activity mimics mule patterns.
    """

    name: str
    spend_level: str
    looks_like_mule_in_normal_life: bool


# The five baseline demographic personas in Level 11A.
# Small shopkeepers and gig workers receive many small payments from random senders in daily life,
# providing realistic adversarial distractors ("look-alikes") that fraud models must not wrongly freeze.
PERSONAS: dict[str, Persona] = {
    "salaried_employee": Persona("salaried_employee", "mid", False),
    "student": Persona("student", "low", False),
    "retiree": Persona("retiree", "low", False),
    "small_shopkeeper": Persona("small_shopkeeper", "high", True),
    "gig_worker": Persona("gig_worker", "low", True),
}


def get_persona(name: str) -> Persona:
    """
    WHAT: Retrieves the immutable Persona specification by its identifier string.
    WHY: Translates stored string labels into structured demographic definitions.
    """
    if name not in PERSONAS:
        raise ValueError(f"Unknown persona '{name}'; available: {list(PERSONAS.keys())}")
    return PERSONAS[name]


def choose_persona_name(rng: np.random.Generator) -> str:
    """
    WHAT: Samples one persona name at random according to the configured demographic proportions.
    WHY: Shapes the town's population distribution according to empirical persona mix ratios.
    """
    names = list(PERSONA_MIX.keys())
    weights = [PERSONA_MIX[n] for n in names]
    chosen = rng.choice(names, p=weights)
    return str(chosen)
