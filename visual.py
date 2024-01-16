COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'blue': (0, 0, 255),
    'red': (255, 0, 0)
}

TEXT_COLOR = COLORS['white']
BACKGROUND_COLOR = COLORS['black']
PLOT_X, PLOT_HEIGHT, PLOT_LAYER_WIDTH = 600, 400, 200


def get_general_text(simulation_length, current_time, generation, fitnesses, population):
    """Returns the general information about the simulation as a list of strings.

    Args:
        simulation_length (float): The maximum time length of this generation.
        current_time (float): The amount of time that has passed this generation.
        generation (int): The number of the current generation.
        fitnesses (list): All fitnesses of individuals in the latest generation.
        population (int): Number of individuals in the simulation.

    Returns:
        list: List with strings to display.
    """
    return [f"Time remaining: {(simulation_length-current_time)/1000}",
            f"Generation: {generation}",
            f"Avg. fitness: {round(sum(fitnesses)/len(fitnesses))}",
            f"Median fitness: {round(fitnesses[len(fitnesses)//2])}",
            f"Max fitness: {round(max(fitnesses))}",
            f"Amount of species: {len(population)}",
            f"Individuals: {len([ind for species in population for ind in species.members])}"]


def get_selected_text(selected):
    """Returns specific information about the selected individual.

    Args:
        selected (Car): The selected individual

    Returns:
        list: List with strings to display.
    """
    return [f"Fitness: {round(selected.nn.fitness)}",
            f"Species ID: {sum(selected.color)}"]
