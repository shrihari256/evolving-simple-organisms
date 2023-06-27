import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation


class Creature:
    def __init__(self, x, y, speed, movement_type, energy=100, map_width=100, map_height=100):
        self.x = x
        self.y = y
        #self.speed = speed
        self.speed = Gene(speed, 0.1, 0.1)
        self.movement_type = movement_type
        self.energy = energy
        self.map_width = map_width
        self.map_height = map_height
        self.dead = False


    def observe(self, food):
        # Your logic to observe food here
        pass

    def move(self):
        if self.movement_type == 'linear':
            # print(type(self.speed.value))
            # print(self.x + self.speed.value)
            # print(self.x + np.random.uniform(-(self.speed.value), self.speed.value))

            self.x = self.x + np.random.uniform(-self.speed.value, self.speed.value)
            self.y = self.y + np.random.uniform(-self.speed.value, self.speed.value)
            # self.x += np.random.uniform(-(self.speed.value), self.speed.value)
            # self.y += np.random.uniform(-(self.speed.value), self.speed.value)
        elif self.movement_type == 'angular':
            angle = np.random.uniform(0, 2 * np.pi)
            # print(type(self.speed.value))
            # print(self.speed.value)
            self.x += self.speed.value * np.cos(angle)
            self.y += self.speed.value * np.sin(angle)
        
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

class Gene:
    def __init__(self, value, mutation_rate, mutation_probability):
        self.value = value
        self.mutation_rate = mutation_rate
        self.mutation_probability = mutation_probability
        print(self.value)

    def mutate(self):
        if np.random.choice([True, False], p=[self.mutation_probability, 1-self.mutation_probability]):
            self.value += np.random.normal(scale=(self.value * self.mutation_rate))

class Simulation:
    #def __init__(self, width, height, num_creatures, num_food, ticks, generations):
    def __init__(self, width, height, initial_food, initial_creatures, food_energy, creature_speed_variance,
                 food_spawn_rate, ticks, generations, max_food,creature_avg_speed,eat_radius):
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
        self.total_tick = 0 # Total ticks that have passed
        self.creature_avg_speed = creature_avg_speed # Average speed of creatures
        self.eat_radius = eat_radius # Radius in which creatures can eat food


        self.pop_stats = []  # List to store population count
        self.food_stats = []  # List to store food count
        self.tick_stats = []  # List to store tick count

        for _ in range(initial_creatures):
            creature = Creature(np.random.uniform(0, self.width), np.random.uniform(0, self.height), 
                    np.random.uniform(creature_avg_speed-(creature_speed_variance/2), creature_avg_speed+(creature_speed_variance/2)), random.choice(['linear', 'angular']),
                    map_width=self.width, map_height=self.height)

            self.creatures.append(creature)

        for _ in range(initial_food):
            food = Food(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            self.food.append(food)

    def run(self):
        #fig, ax = plt.subplots()
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax3 = ax2.twinx()
        #fig2, ax2 = plt.subplots()
        ax2.set_xlabel("Time (ticks)")
        ax2.set_ylabel("Count")
        ax2.set_title("Population and Food over Time")

        pop_line, = ax2.plot([], [], label='Population')
        food_line, = ax2.plot([], [], label='Food')
        ax2.legend()

        def animate(i):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            self.current_tick += 1
            if self.current_tick > self.ticks:
                self.current_tick = 0
                self.current_generation += 1

            new_creatures = []
            
            for creature in self.creatures:
                creature.move()

                # Check if creature is near food
                for food in self.food:
                    distance = np.sqrt((creature.x - food.x)**2 + (creature.y - food.y)**2)

                    # If creature is near food, it eats it and gains energy
                    if distance < self.eat_radius:  # You can adjust this value as needed
                        creature.energy += food.energy
                        self.food.remove(food)  # Remove food from simulation

                # Your logic here to reproduce
                # Check if creature has enough energy to reproduce
                if creature.energy >= 150:
                    # Create a new creature at the same location
                    offspring = Creature(creature.x, creature.y, creature.speed, creature.movement_type, energy=75, map_width=self.width, map_height=self.height)
                    print("Born")
                    print(offspring.speed.value)
                    new_creatures.append(offspring)

                    # Divide energy between parent and offspring
                    creature.energy = 75
            
            # Add the new creatures to the simulation
            self.creatures.extend(new_creatures)

            

            # Also handle mutation here

            # Remove dead creatures
            self.creatures = [creature for creature in self.creatures if not creature.dead] 

            # Add your logic here to spawn food every n ticks

            # Spawn food every n ticks, up to maximum amount
            if self.current_tick % self.food_spawn_rate == 0 and len(self.food) < self.max_food:
                self.food.append(Food(np.random.uniform(0, self.width), np.random.uniform(0, self.height), self.food_energy))

            creature_x = [creature.x for creature in self.creatures]
            creature_y = [creature.y for creature in self.creatures]
            ax1.scatter(creature_x, creature_y, c='blue')

            # Record stats
            self.total_tick += 1

            self.tick_stats.append(self.total_tick)
            self.pop_stats.append(len(self.creatures))
            self.food_stats.append(len(self.food))

            # annotate the energy levels
            for creature in self.creatures:
                ax1.annotate(f"{creature.energy:.1f}", (creature.x, creature.y))

            food_x = [food.x for food in self.food]
            food_y = [food.y for food in self.food]
            ax1.scatter(food_x, food_y, c='red')

            ax1.set_title(f"Epoch: {self.current_generation}, Tick: {self.current_tick}, Population: {len(self.creatures)}, Food: {len(self.food)}")

            ax1.set_xlim(0, self.width)
            ax1.set_ylim(0, self.height)

            # Only plot the last 1000 data points if available
            if len(simulation.tick_stats) >= 100:
                ax2.plot(self.tick_stats[-100:], self.pop_stats[-100:], label='Population')
                ax3.plot(self.tick_stats[-100:], self.food_stats[-100:], label='Food', color='red')
    
                #show y labels
                ax2.set_ylabel("Population")
                ax3.set_ylabel("Food")

                #set y axis color
                ax2.tick_params(axis='y', colors='blue')
                ax3.tick_params(axis='y', colors='red')

                ax2.set_ylim(0, max(self.pop_stats)+2)
                ax3.set_ylim(0, max(self.food_stats)+2)


        # def init2():
        #     pop_line.set_data([], [])
        #     food_line.set_data([], [])
        #     return pop_line, food_line,
    
        # def animate2(i):
        #     pop_line.set_data(range(len(simulation.pop_stats[:i+1])), simulation.pop_stats[:i+1])
        #     food_line.set_data(range(len(simulation.food_stats[:i+1])), simulation.food_stats[:i+1])
        #     ax2.set_xlim(0, i+1)
        #     ax2.set_ylim(0, max(max(simulation.pop_stats), max(simulation.food_stats)))
        #     return pop_line, food_line,


        ani = animation.FuncAnimation(fig, animate, frames=self.ticks*self.generations, interval=10)
        # set the window size for the animation
        fig.set_size_inches(10, 20)
        # ani2 = animation.FuncAnimation(fig2, animate2, init_func=init2, frames=len(simulation.pop_stats), blit=True, interval=100)
        
        plt.show()


simulation = Simulation(
    width=1000,
    height=1000,
    initial_food=500,
    initial_creatures=1,
    food_energy=30,
    creature_avg_speed=30,
    creature_speed_variance=4,
    food_spawn_rate=1,
    ticks=100,
    generations=10,
    max_food=1000,
    eat_radius=10
)

simulation.run()
