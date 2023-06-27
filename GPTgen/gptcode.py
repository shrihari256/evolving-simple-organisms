import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation


class Creature:
    def __init__(self, x, y, speed, movement_type, energy=100, map_width=100, map_height=100):
        self.x = x
        self.y = y
        self.speed = speed
        self.movement_type = movement_type
        self.energy = energy
        self.map_width = map_width
        self.map_height = map_height
        self.dead = False

    def move(self):
        if self.movement_type == 'linear':
            self.x += np.random.uniform(-self.speed, self.speed)
            self.y += np.random.uniform(-self.speed, self.speed)
        elif self.movement_type == 'angular':
            angle = np.random.uniform(0, 2 * np.pi)
            self.x += self.speed * np.cos(angle)
            self.y += self.speed * np.sin(angle)
        
        # Keep creature within bounds
        self.x = max(0, min(self.x, self.map_width))
        self.y = max(0, min(self.y, self.map_height))

        self.energy -= 1
        if self.energy <= 0:
            self.dead = True


    def eat(self, food):
        # Your logic to eat food and gain energy here
        pass

    def reproduce(self):
        # Your logic to reproduce and split into two here
        pass

    def mutate(self, mutation_rate):
        # Your logic to mutate creature's speed here
        pass


class Food:
    def __init__(self, x, y, energy=20):
        self.x = x
        self.y = y
        self.energy = energy


class Simulation:
    #def __init__(self, width, height, num_creatures, num_food, ticks, generations):
    def __init__(self, width, height, initial_food, initial_creatures, food_energy, creature_speed_variance,
                 food_spawn_rate, ticks, generations, max_food):
        self.width = width
        self.height = height
        self.creatures = []
        self.food = []
        self.ticks = ticks
        self.generations = generations
        self.current_tick = 0
        self.current_generation = 1
        self.max_food = max_food  # Maximum amount of food that can be present in the map 
        self.food_energy = food_energy  # Energy that each food gives to a creature
        self.food_spawn_rate = food_spawn_rate  # How often food spawns in ticks


        for _ in range(initial_creatures):
            creature = Creature(np.random.uniform(0, self.width), np.random.uniform(0, self.height), 
                    np.random.uniform(2.5, 3.5), random.choice(['linear', 'angular']),
                    map_width=self.width, map_height=self.height)

            self.creatures.append(creature)

        for _ in range(initial_food):
            food = Food(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            self.food.append(food)

    def run(self):
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()

            self.current_tick += 1
            if self.current_tick > self.ticks:
                self.current_tick = 0
                self.current_generation += 1

            for creature in self.creatures:
                creature.move()

                # Check if creature is near food
                for food in self.food:
                    distance = np.sqrt((creature.x - food.x)**2 + (creature.y - food.y)**2)

                    # If creature is near food, it eats it and gains energy
                    if distance < 5:  # You can adjust this value as needed
                        creature.energy += food.energy
                        self.food.remove(food)  # Remove food from simulation

                # Your logic here to reproduce
                # Also handle mutation here

            # Remove dead creatures
            self.creatures = [creature for creature in self.creatures if not creature.dead] 

            # Add your logic here to spawn food every n ticks

            # Spawn food every n ticks, up to maximum amount
            if self.current_tick % self.food_spawn_rate == 0 and len(self.food) < self.max_food:
                self.food.append(Food(np.random.uniform(0, self.width), np.random.uniform(0, self.height), self.food_energy))

            creature_x = [creature.x for creature in self.creatures]
            creature_y = [creature.y for creature in self.creatures]
            ax.scatter(creature_x, creature_y, c='blue')

            # annotate the energy levels
            for creature in self.creatures:
                ax.annotate(f"{creature.energy:.1f}", (creature.x, creature.y))

            food_x = [food.x for food in self.food]
            food_y = [food.y for food in self.food]
            ax.scatter(food_x, food_y, c='red')

            ax.set_title(f"Generation: {self.current_generation}, Tick: {self.current_tick}")

            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)


        ani = animation.FuncAnimation(fig, animate, frames=self.ticks*self.generations, interval=10)
        plt.show()


simulation = Simulation(
    width=100,
    height=100,
    initial_food=50,
    initial_creatures=10,
    food_energy=20,
    creature_speed_variance=1,
    food_spawn_rate=5,
    ticks=100,
    generations=10,
    max_food=100
)

simulation.run()
