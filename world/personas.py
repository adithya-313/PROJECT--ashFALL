"""
WHAT: Demographic lifestyle persona definitions and random persona sampling.
WHY: Assigns behavioral profiles (spending tier, look-alike characteristics) to people.
TERM: persona: A demographic behavioral archetype governing spending magnitude and inbound payment habits,
      e.g. a student buying tea vs a salaried employee paying monthly rent.
TERM: mule look-alike: An innocent person whose normal legitimate routine receives dozens of small incoming transfers,
      e.g. a roadside tea stall owner or gig worker superficially mimicking a money mule.
TERM: spend level: A categorical classification of spending power ('low', 'mid', 'high'),
      e.g. 'low' for students with limited pocket money versus 'high' for established business proprietors.
"""

from dataclasses import dataclass

import numpy as np

from world.placeholders import PERSONA_MIX


@dataclass(frozen=True)
class Persona:
    """
    WHAT: Represents a distinct socioeconomic behavioral template.
    It defines spending tiers ('low', 'mid', 'high') and indicates whether normal activity mimics mule patterns.
    WHY: Fraud models must distinguish innocent high-velocity recipients from actual criminal money mules.
    Formalizing personas creates ground-truth behavioral benchmarks across diverse economic cohorts.
    Args:
        name: Unique string name of the persona archetype, e.g. 'gig_worker'.
        spend_level: String categorical spending power tier, one of 'low', 'mid', or 'high'.
        looks_like_mule_in_normal_life: Boolean flag indicating if legitimate daily routine receives
            many small payments.
    Example:
        persona = Persona("gig_worker", "low", True)
        assert persona.spend_level == "low" and persona.looks_like_mule_in_normal_life is True
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
    """WHAT: Retrieves the immutable Persona specification by its identifier string.
    It verifies that the requested name exists in the registered PERSONAS catalog.
    If valid, it returns the matching frozen Persona instance.
    WHY: Translates stored string labels into structured demographic definitions.
    Prevents invalid or misspelled persona archetypes from entering the simulation.
    Args: name: String identifier corresponding to a registered persona key.
    Returns: Frozen Persona instance defining spend level and mule look-alike status.
    Example:
        p = get_persona("gig_worker")
        assert p.name == "gig_worker" and p.spend_level == "low" and p.looks_like_mule_in_normal_life is True
    """
    # Verify the requested persona name is registered in the demographic catalog.
    if name not in PERSONAS:
        raise ValueError(f"Unknown persona '{name}'; available: {list(PERSONAS.keys())}")

    # Fetch and supply the matching frozen Persona instance.
    return PERSONAS[name]


def choose_persona_name(rng: np.random.Generator) -> str:
    """WHAT: Samples one persona name at random according to configured demographic proportions.
    It reads persona names and probability weights from the PERSONA_MIX configuration table.
    It then executes a weighted categorical choice using the provided random generator.
    WHY: Shapes the town population distribution according to empirical persona mix ratios.
    Ensures realistic proportions of students, retirees, workers, and merchants.
    Args: rng: NumPy pseudo-random number generator for reproducible sampling.
    Returns: String identifier of the chosen persona, e.g. 'small_shopkeeper'.
    Example:
        name = choose_persona_name(np.random.default_rng(42))
        assert name == "small_shopkeeper"
    """
    # Extract persona names and their corresponding population distribution weights.
    names = list(PERSONA_MIX.keys())
    weights = [PERSONA_MIX[n] for n in names]

    # Sample a persona name according to the demographic probability distribution.
    chosen = rng.choice(names, p=weights)

    # Deliver the sampled persona identifier string.
    return str(chosen)
