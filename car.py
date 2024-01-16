import math
import pygame as pg
import numpy as np

import config as cf


class Car:
    HALF_FOV = cf.CAR_FOV/2
    VIEW_ANGLES = np.linspace(-HALF_FOV, HALF_FOV, cf.NUM_INPUTS).tolist()

    def __init__(self, x0, y0, start_angle, nn, color):
        self.x, self.y = x0, y0
        self.start_angle = start_angle
        self.angle = start_angle
        self.has_crashed = False
        self.steering = 0
        self.speed = 0
        self.nn = nn
        self.image = pg.image.load(cf.CAR_IMAGE).convert_alpha()
        self.mask = pg.mask.from_surface(self.image)
        self.color = color
        self.recolor(self.color)
        self.width = self.image.get_rect().center[0]

    def update(self, track_mask, window):
        """Update the car's state.

        Args:
            track_mask (pygame.Mask): Pygame mask of the track.
            window (pygame.Surface): The window to blit the car's image onto.
        """
        if not self.has_crashed:
            self.inputs = self.get_inputs(track_mask, window)
            self.speed, self.steering = self.nn.feed_forward(self.inputs)
            self.speed *= cf.CAR_SPEED_MULTIPLIER
            self.steering *= cf.CAR_STEER_MULTIPLIER
            if self.speed > cf.CAR_MAX_SPEED:
                self.speed = cf.CAR_MAX_SPEED
            if self.speed < cf.CAR_MIN_SPEED:
                self.speed = cf.CAR_MIN_SPEED
            self.update_pos_and_angle()
            self.update_fitness()
            self.update_collision(track_mask)
            self.outputs = self.speed, self.steering
        self.draw(window)

    def update_collision(self, track_mask):
        """Check for collision and update the has_crashed flag if a crash occurs.

        Args:
            track_mask (pygame.Mask): Pygame mask of the track.
        """
        if not track_mask.overlap_area(self.mask, (self.x, self.y)) == self.mask.count():
            self.has_crashed = True

    def draw(self, window):
        """Blit the car onto the window and update its mask.

        Args:
            window (pygame.Surface): The window to blit the car's image onto.
        """
        rotated = pg.transform.rotate(self.image, self.angle-self.start_angle)
        window.blit(rotated, (self.x, self.y))
        self.mask = pg.mask.from_surface(rotated)

    def update_pos_and_angle(self):
        """Use the car's velocity and steering to update the car's position and angle."""
        vx, vy = self.get_2D_components(self.speed, self.angle)
        self.x += vx
        self.y += vy

        if abs(self.steering) > cf.CAR_MAX_TURN:
            self.steering /= abs(self.steering)
            self.steering *= cf.CAR_MAX_TURN
        self.angle += self.steering

    def update_fitness(self):
        """Update the fitness of the car."""
        self.nn.fitness += self.speed  # Distance based fitness.
        # self.nn.fitness += 1 # Time based fitness.

    def get_2D_components(self, vector, angle):
        """Return the horizontal and vertical components of a 2D vector.

        Args:
            vector (tuple): The vector which's components are wanted.
            angle (float): The angle of the vector in the 2D plane in degrees.

        Returns:
            tuple: The x and y components of the 2D vector.
        """
        x = vector * math.cos(math.radians(angle))
        y = vector * -math.sin(math.radians(angle))
        return x, y

    def get_inputs(self, track_mask, window):
        """Get the inputs that should be passed onto the NN.

        Args:
            track_mask (pygame.Mask): Pygame mask of the track.
            window (pygame.Surface): The window to blit the car's image onto.

        Returns:
            list: Distances of each ray until collision (input to NN).
        """
        return [self.get_ray_distance(angle, track_mask, window) for angle in Car.VIEW_ANGLES]

    def get_ray_distance(self, angle, track_mask, window):
        """Get the distance to the edge for the ray at the given angle.

        Args:
            angle (float): The angle of the ray from the car's perspective in degrees.
            track_mask (pygame.Mask): Pygame mask of the track.
            window (pygame.Surface): The window to blit the car's image onto.

        Returns:
            _type_: _description_
        """

        radians = math.radians(self.angle + angle)
        ray_direction = (math.cos(radians), -math.sin(radians))
        half_mask_count = self.mask.count()/2
        x, y = self.get_center()
        dx, dy = np.subtract(self.get_center(), (self.x, self.y))
        distance = 0
        while True:
            if track_mask.overlap_area(self.mask, (int(x-dx), int(y-dy))) <= half_mask_count or distance >= cf.CAR_MAX_VIEW_DISTANCE:
                break
            distance += 1
            x += ray_direction[0] * cf.RAY_SPEED
            y += ray_direction[1] * cf.RAY_SPEED
            if cf.SHOW_RAYS:
                pg.draw.circle(window, self.color, (x, y), radius=1)
        return max(0, distance - math.ceil(self.width/2))

    def recolor(self, color):
        """Fill all pixels of the surface with color, preserve transparency.

        Args:
            color (tuple): The color to recolor with, RGBA format.
        """
        w, h = self.image.get_size()
        r, g, b, _ = color
        for x in range(w):
            for y in range(h):
                a = self.image.get_at((x, y))[3]
                self.image.set_at((x, y), pg.Color(r, g, b, a))

    def get_center(self):
        """Get the center position of the car in global coordinates.

        Returns:
            tuple: The center coordinates of the car in global coordinates.
        """
        x_offset, y_offset = self.image.get_rect().center
        return self.x + x_offset, self.y + y_offset
