# TODO: Improve NEAT class
# TODO: Add ability to click on individual cars and see their NN
# TODO: Add control buttons
# TODO: Configs
# TODO: More settings
# TODO: Visualize improvement over generations
# TODO: Hyperparameter tuning
# TODO: Clean up code, remove colormask function
# TODO: Figure out starting angle
# TODO: Make gif and add to github

import pygame as pg
import math

from neat import NEAT, Layer
from car import Car

import config

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (238, 221, 130, 100)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BACKGROUND_COLOR = BLACK
TEXT_COLOR = WHITE


def color_mask(image, mask_color):
    mask_image = image.convert()
    mask_image.set_colorkey(mask_color)
    mask = pg.mask.from_surface(mask_image)
    return mask


def generate_cars(neat):
    cars = []
    for species in neat.population:
        for nn in species.members:
            cars.append(Car(x0, y0, 90, nn, species.color))
    return cars


pg.init()
pg.display.set_caption(config.WIN_TITLE)
win = pg.display.set_mode((config.WIN_WIDTH, config.WIN_HEIGHT))

# black track with white background and red starting pixel
TRACK_IMAGE = pg.image.load(config.TRACK_IMAGE).convert_alpha()
TRACK_WIDTH, TRACK_HEIGHT = TRACK_IMAGE.get_width(), TRACK_IMAGE.get_height()
GUI_HEIGHT = config.WIN_HEIGHT - TRACK_HEIGHT
track_rect = TRACK_IMAGE.get_rect()
track_surface = pg.Surface.convert_alpha(TRACK_IMAGE)
TRACK_IMAGE.set_colorkey(WHITE)
mask = color_mask(TRACK_IMAGE, WHITE)

width, height = track_surface.get_width(), track_surface.get_height()
# -- Set starting position at red pixels
for x in range(width):
    for y in range(height):
        pixel_color = track_surface.get_at((x, y))
        red, green, blue = pixel_color[0:3]
        if red > 150 and green < 100 and blue < 100:
            x0, y0 = x, y
if not x0:
    raise Exception("The track must have a red starting pixel")

current_time = 0
clock = pg.time.Clock()
simulation_length = config.SIMULATION

neat = NEAT(config.NUM_INPUTS, config.NUM_OUTPUTS, config.POPULATION_SIZE)
cars = generate_cars(neat)
visualized = cars[0]

font = pg.font.Font(None, 36)
fitnesses = [0]

running = True
while running:
    # -- Update:
    pg.display.flip()
    win.fill(BACKGROUND_COLOR)
    win.blit(mask.to_surface(), track_rect)

    for car in cars:
        car.update(mask, win)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            mouse_pos = pg.mouse.get_pos()
            for car in cars:
                if math.dist((car.x, car.y), mouse_pos) < config.CLICK_RADIUS:
                    visualized = car
                    break

    # -- Display text:
    general_text = [f"Time remaining: {(simulation_length-current_time)/1000}",
                    f"Generation: {neat.generation}",
                    f"Avg. fitness: {round(sum(fitnesses)/len(fitnesses))}",
                    f"Median fitness: {round(fitnesses[len(fitnesses)//2])}",
                    f"Max fitness: {round(max(fitnesses))}",
                    f"Amount of species: {len(neat.population)}"]

    general_pos = [(100, TRACK_HEIGHT+50*i)
                   for i in range(len(general_text))]
    for i, text in enumerate(general_text):
        win.blit(font.render(text, True, TEXT_COLOR), general_pos[i])

    # -- Display neural network:
    nn = visualized.nn

    pos = {}  # Positions of nodes on the screen
    plot_x, plot_y = 600, TRACK_HEIGHT
    dx, dy = 200, 30
    plot_height = 400

    nodes_placed = [0, 0, 0]
    for id, node in nn.nodes.items():
        layer = node.type.value
        new_x = plot_x + layer*dx
        dy = plot_height/(nn.layer_size[layer]*2)
        new_y = plot_y + nodes_placed[layer]*dy + dy/2

        pos[id] = (new_x, new_y)

        if node.type == Layer.INPUT:
            input_text = f"{round(visualized.inputs[nodes_placed[Layer.INPUT.value]])}"
            x2, y2 = pos[id]
            win.blit(font.render(input_text, True, TEXT_COLOR), (x2-50, y2))
        elif node.type == Layer.OUTPUT:
            output_text = f"{round(visualized.outputs[nodes_placed[Layer.OUTPUT.value]])}"
            x2, y2 = pos[id]
            win.blit(font.render(output_text, True, TEXT_COLOR), (x2+50, y2-10))

        nodes_placed[node.type.value] += 1

    for edge in nn.edges.values():
        pg.draw.line(win, WHITE, pos[edge.from_node], pos[edge.to_node])

    for id, node in nn.nodes.items():
        pg.draw.circle(win, RED, pos[id], radius=10)

    pg.draw.circle(win, BLUE, visualized.get_center(), radius=30, width=2)

    selected_text = [f"Fitness: {visualized.nn.fitness}",
                     f"Generation: {neat.generation}",
                     f"Avg. fitness: {round(sum(fitnesses)/len(fitnesses))}",
                     f"Median fitness: {round(fitnesses[len(fitnesses)//2])}",
                     f"Max fitness: {round(max(fitnesses))}",
                     f"Amount of species: {len(neat.population)}"]

    selected_pos = [(1300, TRACK_HEIGHT+50*i)
                    for i in range(len(selected_text))]
    for i, text in enumerate(selected_text):
        win.blit(font.render(text, True, TEXT_COLOR), selected_pos[i])

    # -- Perform evolution:
    clock.tick(config.FPS)
    current_time += clock.get_time()
    if current_time >= simulation_length or all([car.has_crashed for car in cars]):
        generation, fitnesses, population = neat.evolve(current_time)
        simulation_length += 500
        cars = generate_cars(neat)
        visualized = cars[0]
        current_time = 0

pg.quit()
