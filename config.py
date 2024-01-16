# Simulation:
START_TIME = 1 * 1000
ADDED_TIME = 0
POPULATION_SIZE = 10
GENERATIONS = 100
SHOW_RAYS = False
SHOW_NN = True
RAY_SPEED = 3

# Neural Network:
NUM_INPUTS = 5
NUM_MAX_HIDDEN = 7
NUM_OUTPUTS = 2

# Car:
CAR_MIN_SPEED = 3
CAR_MAX_SPEED = 8
CAR_FOV = 150
CAR_MAX_TURN = 2
CAR_MAX_VIEW_DISTANCE = 800
CAR_SPEED_MULTIPLIER = 0.1
CAR_STEER_MULTIPLIER = 0.1

# NEAT:
MUTATION_RATE_ADD_NODE = 0.1
MUTATION_RATE_ADD_EDGE = 0.1
MUTATION_RATE_UPDATE_PARAM = 1
SPECIES_SURVIVAL = 0.6
INDIVIDUAL_SURVIVAL = 0.2

# Compatability:
COMPATIBILITY_THRESHOLD = 1
C1 = 1
C2 = 1
C3 = 1

# Window:
FPS = 60
WIN_WIDTH, WIN_HEIGHT = 1800, 1000
WIN_TITLE = "Car evolution with NEAT"
TRACK_IMAGE = "track.png"
CAR_IMAGE = "car.png"
CLICK_RADIUS = 35


def species_score(species):
    """Function chosen by the user to evaluate a species' performance."""
    return sum(m.fitness**2 for m in species.members)
