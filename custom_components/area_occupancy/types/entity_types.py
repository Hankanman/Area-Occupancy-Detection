""" Hardcoded definition of entity types"""

from enum import StrEnum

class EntityType(StrEnum):
    """Enum representing the different types of entities used."""
    MOTION = "motion"
    MEDIA = "media" 
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    LIGHT = "light"
    ENVIRONMENTAL = "environmental"
    WASP_IN_BOX = "wasp_in_box"

# Default configuration for each entity type
ENTITY_TYPE_CONFIGS = {
    EntityType.MOTION: {
        "prob_given_true": 0.8,
        "prob_given_false": 0.2,
        "default_prior": 0.5,
        "weight": 1.0
    },
    EntityType.MEDIA: {
        "prob_given_true": 0.7,
        "prob_given_false": 0.3,
        "default_prior": 0.5,
        "weight": 0.8
    },
    EntityType.APPLIANCE: {
        "prob_given_true": 0.6,
        "prob_given_false": 0.4,
        "default_prior": 0.5,
        "weight": 0.7
    },
    EntityType.DOOR: {
        "prob_given_true": 0.7,
        "prob_given_false": 0.3,
        "default_prior": 0.5,
        "weight": 0.6
    },
    EntityType.WINDOW: {
        "prob_given_true": 0.7,
        "prob_given_false": 0.3,
        "default_prior": 0.5,
        "weight": 0.6
    },
    EntityType.LIGHT: {
        "prob_given_true": 0.6,
        "prob_given_false": 0.4,
        "default_prior": 0.5,
        "weight": 0.5
    },
    EntityType.ENVIRONMENTAL: {
        "prob_given_true": 0.5,
        "prob_given_false": 0.5,
        "default_prior": 0.5,
        "weight": 0.4
    },
    EntityType.WASP_IN_BOX: {
        "prob_given_true": 0.9,
        "prob_given_false": 0.1,
        "default_prior": 0.5,
        "weight": 0.9
    }
}

