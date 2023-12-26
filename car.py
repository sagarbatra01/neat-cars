import math
import pygame as pg
import numpy as np

import config


class Car:
    FOV = 150
    MAX_TURN = 1
    MAX_DISTANCE = 800
    VIEW_ANGLES = np.linspace(-FOV/2, FOV/2, config.NUM_INPUTS).tolist()

    def __init__(self, x, y, angle, neural_network, color):
        self.x, self.y = x, y
        self.angle = angle
        self.speed = 0
        self.nn = neural_network
        self.image = pg.image.load(config.CAR_IMAGE).convert_alpha()
        self.mask = pg.mask.from_surface(self.image)
        self.has_crashed = False
        self.steering = 0
        self.recolor(color)

    def update(self, track_mask, window):
        if not self.has_crashed:
            self.inputs = self.get_inputs(track_mask, window)
            self.speed, self.steering = self.nn.feed_forward(self.inputs)
            self.speed = min(config.CAR_MAX_SPEED, max(
                config.CAR_MIN_SPEED, self.speed/10))
            self.update_position()
            self.update_fitness()
            self.update_collision(track_mask)
            self.outputs = self.speed, self.steering
            print(self.inputs)
            # print("hi")
            # print(self.speed)
        self.draw(window)

    def update_collision(self, track_mask):
        if not self.has_crashed and not track_mask.overlap_area(self.mask, (self.x, self.y)) == self.mask.count():
            self.has_crashed = True

    def draw(self, window):
        rotated = pg.transform.rotate(self.image, self.angle-90)
        window.blit(rotated, (self.x, self.y))
        self.mask = pg.mask.from_surface(rotated)

    def update_position(self):
        vx, vy = self.get_components(self.speed, self.angle)
        self.x += vx
        self.y += vy

        if abs(self.steering) > Car.MAX_TURN:
            self.steering /= abs(self.steering)
            self.steering *= Car.MAX_TURN
        self.angle += self.steering

    def update_fitness(self):
        self.nn.fitness += self.speed  # - abs(steering)

    def get_components(self, vector, angle):
        """Return the horizontal and vertical components of a vector."""
        x = vector * math.cos(math.radians(angle))
        y = vector * -math.sin(math.radians(angle))
        return x, y

    def get_inputs(self, track_mask, window):
        return [self.get_ray_distance(angle, track_mask, window) for angle in Car.VIEW_ANGLES]

    def get_ray_distance(self, angle, track_mask, window):

        radians = math.radians(self.angle + angle)
        ray_direction = (math.cos(radians), -math.sin(radians))
        # print(angle, self.angle, ray_direction)

        distance = 0
        x, y = self.get_center()
        print("in ray")
        if not (0 <= x < window.get_size()[0] and 0 <= y < window.get_size()[1]):
            print("skip")

        while 0 <= x < window.get_size()[0] and 0 <= y < window.get_size()[1]:
            # print("in while")
            if track_mask.overlap_area(self.mask, (int(x), int(y))) != self.mask.count():
                return distance

            # No overlap, update distance and continue raycasting
            distance += 1
            x += ray_direction[0]
            y += ray_direction[1]
            pg.draw.circle(window, (255, 0, 0), (x, y), radius=1)
            # If no intersection, return a maximum distance
            if distance >= Car.MAX_DISTANCE:
                break

        # Return the calculated distance
        return distance

    def recolor(self, color):
        """Fill all pixels of the surface with color, preserve transparency."""
        w, h = self.image.get_size()
        r, g, b, _ = color
        for x in range(w):
            for y in range(h):
                a = self.image.get_at((x, y))[3]
                self.image.set_at((x, y), pg.Color(r, g, b, a))

    def get_center(self):
        x_offset, y_offset = self.image.get_rect().center
        return self.x + x_offset, self.y + y_offset
