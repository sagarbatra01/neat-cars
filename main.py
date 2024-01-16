import pygame as pg
import math

import visual
import config as cf
from neat import NEAT, Layer
from car import Car

pg.init()
pg.display.set_caption(cf.WIN_TITLE)
WIN = pg.display.set_mode((cf.WIN_WIDTH, cf.WIN_HEIGHT))
TRACK_IMAGE = pg.image.load(cf.TRACK_IMAGE).convert_alpha()
TRACK_WIDTH, TRACK_HEIGHT = TRACK_IMAGE.get_width(), TRACK_IMAGE.get_height()
GUI_HEIGHT = cf.WIN_HEIGHT - TRACK_HEIGHT

# Load and process track.
track_rect = TRACK_IMAGE.get_rect()
track_surface = pg.Surface.convert_alpha(TRACK_IMAGE)
mask_image = TRACK_IMAGE.convert()
mask_image.set_colorkey(visual.COLORS['white'])
mask = pg.mask.from_surface(mask_image)

# Find starting point and angle.
red_pixels = []
width, height = track_surface.get_width(), track_surface.get_height()
for x in range(width):
    for y in range(height):
        pixel_color = track_surface.get_at((x, y))
        red, green, blue = pixel_color[0:3]
        if red > 150 and green < 100 and blue < 100:
            red_pixels.append((x, y))
if len(red_pixels) < 2:
    raise Exception("The track must have at least 2 red starting pixels.")
x0, y0 = red_pixels[len(red_pixels)//2]
x1, y1, x2, y2 = red_pixels[0] + red_pixels[-1]
start_angle = math.degrees(math.atan2(y2-y1, x2-x1))


def generate_individuals():
    """Generate and return a new set of individuals using the current neural
    networks in the given neat instance.

    Returns:
        list: List of newly generated individuals.
    """
    new_individuals = []
    for species in neat.population:
        for nn in species.members:
            new_individuals.append(Car(x0, y0, start_angle, nn, species.color))
    return new_individuals


time = 0
clock = pg.time.Clock()
time_limit = cf.START_TIME
font = pg.font.Font(None, 36)

neat = NEAT()
individuals = generate_individuals()
selected, fitnesses = individuals[0], [0]

running = True
while running:
    pg.display.flip()
    WIN.fill(visual.BACKGROUND_COLOR)
    WIN.blit(mask.to_surface(), track_rect)

    for individual in individuals:
        individual.update(mask, WIN)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            mouse_pos = pg.mouse.get_pos()
            for ind in individuals:
                if math.dist((ind.x, ind.y), mouse_pos) < cf.CLICK_RADIUS:
                    selected = ind
                    break

    # Display general text about the current state.
    general_text = visual.get_general_text(
        time_limit, time, neat.generation, fitnesses, neat.population)
    general_pos = [(100, TRACK_HEIGHT+50*i) for i in range(len(general_text))]
    for i, text in enumerate(general_text):
        WIN.blit(font.render(text, True, visual.TEXT_COLOR), general_pos[i])

    # Display the neural network of the selected individual:
    if cf.SHOW_NN:
        nn = selected.nn
        pos = {}
        nodes_placed = [0, 0, 0]
        for id, node in nn.nodes.items():
            layer = node.type.value
            x = visual.PLOT_X + layer*visual.PLOT_LAYER_WIDTH
            dy = visual.PLOT_HEIGHT/(nn.layer_size[layer]*2)
            y = TRACK_HEIGHT + nodes_placed[layer]*dy + dy/2
            pos[id] = (x, y)

            # Display input and output values.
            if node.type == Layer.INPUT:
                text = f"{round(selected.inputs[nodes_placed[node.type.value]])}"
                WIN.blit(font.render(text, True, visual.TEXT_COLOR), (x-50, y-12))
            elif node.type == Layer.OUTPUT:
                text = f"{round(selected.outputs[nodes_placed[node.type.value]])}"
                WIN.blit(font.render(text, True, visual.TEXT_COLOR), (x+50, y-12))

            nodes_placed[node.type.value] += 1

        # Display edges:
        for (from_, to), edge in nn.edges.items():
            pg.draw.line(WIN, visual.COLORS['white'], pos[from_], pos[to])

        # Display nodes:
        for id, node in nn.nodes.items():
            pg.draw.circle(WIN, visual.COLORS['red'], pos[id], radius=10)

    # Display which individual is selected.
    pg.draw.circle(WIN, selected.color, selected.get_center(), 30, 2)

    # Display information about the selected individual.
    text = visual.get_selected_text(selected)
    selected_pos = [(1300, TRACK_HEIGHT+50*i) for i in range(len(text))]
    for i, line in enumerate(text):
        WIN.blit(font.render(line, True, visual.TEXT_COLOR), selected_pos[i])

    # If done with current generation, evolve.
    if time >= time_limit or all([ind.has_crashed for ind in individuals]):
        for nn in neat.get_individuals():
            fitnesses.append(nn.fitness/(time/1000))
        neat.evolve()
        time_limit += cf.ADDED_TIME
        individuals = generate_individuals()
        selected = individuals[0]
        time = 0

    clock.tick(cf.FPS)
    time += clock.get_time()

pg.quit()
