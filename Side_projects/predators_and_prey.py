import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Prey and Hunters Simulation")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

# Entity settings
prey_size = 5
hunter_size = 7
num_prey = 20
num_hunters = 10
prey_reproduction_time = 5000  # milliseconds
hunter_reproduction_time = 5000  # milliseconds
vision_range = 50  # distance within which prey can see hunters and vice versa
prey_speed = 2
hunter_speed = 3

# Helper function to calculate distance
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Create classes for Prey and Hunters
class Prey:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, prey_size, prey_size)
        self.last_reproduce_time = pygame.time.get_ticks()
        self.moved_last_turn = False

    def move(self, hunters):
        closest_hunter = None
        closest_distance = vision_range
        for hunter in hunters:
            dist = distance(self.x, self.y, hunter.x, hunter.y)
            if dist < closest_distance:
                closest_hunter = hunter
                closest_distance = dist

        if closest_hunter:
            # Move away from the closest hunter
            dx = self.x - closest_hunter.x
            dy = self.y - closest_hunter.y
            if dx != 0 or dy != 0:
                length = math.sqrt(dx ** 2 + dy ** 2)
                dx /= length
                dy /= length
                self.x += int(dx * prey_speed)
                self.y += int(dy * prey_speed)
                self.moved_last_turn = True
        else:
            # Move randomly
            if random.choice([True, False]):
                self.x += random.choice([-1, 1])
                self.y += random.choice([-1, 1])
                self.moved_last_turn = True
            else:
                self.moved_last_turn = False

        self.rect.topleft = (self.x, self.y)

    def reproduce(self):
        if not self.moved_last_turn and pygame.time.get_ticks() - self.last_reproduce_time > prey_reproduction_time:
            self.last_reproduce_time = pygame.time.get_ticks()
            return Prey(self.x, self.y)
        return None

class Hunter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, hunter_size, hunter_size)
        self.last_reproduce_time = pygame.time.get_ticks()

    def move(self, preys):
        closest_prey = None
        closest_distance = vision_range
        for prey in preys:
            dist = distance(self.x, self.y, prey.x, prey.y)
            if dist < closest_distance:
                closest_prey = prey
                closest_distance = dist

        if closest_prey:
            # Move towards the closest prey
            dx = closest_prey.x - self.x
            dy = closest_prey.y - self.y
            if dx != 0 or dy != 0:
                length = math.sqrt(dx ** 2 + dy ** 2)
                dx /= length
                dy /= length
                self.x += int(dx * hunter_speed)
                self.y += int(dy * hunter_speed)
        else:
            # Move randomly
            self.x += random.choice([-1, 1])
            self.y += random.choice([-1, 1])

        self.rect.topleft = (self.x, self.y)

    def hunt(self, preys):
        for prey in preys:
            if self.rect.colliderect(prey.rect):
                preys.remove(prey)
                return True  # Indicates that the hunter has eaten a prey
        return False

    def reproduce(self):
        if pygame.time.get_ticks() - self.last_reproduce_time > hunter_reproduction_time:
            self.last_reproduce_time = pygame.time.get_ticks()
            return Hunter(self.x, self.y)
        return None

# Initialize entities
preys = [Prey(random.randint(0, width), random.randint(0, height)) for _ in range(num_prey)]
hunters = [Hunter(random.randint(0, width), random.randint(0, height)) for _ in range(num_hunters)]

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(black)

    # Move and draw preys
    new_preys = []
    for prey in preys:
        prey.move(hunters)
        pygame.draw.rect(screen, green, prey.rect)
        new_prey = prey.reproduce()
        if new_prey:
            new_preys.append(new_prey)
    preys.extend(new_preys)

    # Move and draw hunters
    new_hunters = []
    for hunter in hunters:
        hunter.move(preys)
        pygame.draw.rect(screen, red, hunter.rect)
        if hunter.hunt(preys):
            new_hunter = hunter.reproduce()
            if new_hunter:
                new_hunters.append(new_hunter)
    hunters.extend(new_hunters)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
