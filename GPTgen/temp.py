import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation
import statistics as stat
import csv
import pandas as pd


class Creature:
    def __init__(self, x, y, speed, energy=100, vision=10, map_width=100, map_height=100):
        self.x = x
        self.y = y
        
        self.speed = Gene(speed, 0.1, 0.1)
        self.visibility = Gene(vision, 0.1, 0.1)
        self.energy = energy
        self.map_width = map_width
        self.map_height = map_height
        self.dead = False
        self.age = 0
        self.target_direction = None


    def observe(self, foods):
        closest_food = None
        min_distance = float('inf')
        for food in foods:
            distance = np.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_food = food
        
        # Implement fog of war
        if min_distance > self.visibility.value:
            closest_food = None

        if closest_food:
            angle_to_food = np.arctan2(closest_food.y - self.y, closest_food.x - self.x)
            self.target_direction = angle_to_food

        # Energy expenditure for observing
        self.energy -= 0.25+np.power(self.visibility.value,5)/(2 * np.power(25,5))

        pass
        


    def move(self):
        # if self.movement_type == 'linear':
            
        #     self.x = self.x + np.random.uniform(-self.speed.value, self.speed.value)
        #     self.y = self.y + np.random.uniform(-self.speed.value, self.speed.value)
            
        # elif self.movement_type == 'angular':
        #     angle = np.random.uniform(0, 2 * np.pi)
        #     # print(type(self.speed.value))
        #     # print(self.speed.value)
        #     self.x += self.speed.value * np.cos(angle)
        #     self.y += self.speed.value * np.sin(angle)

        if self.target_direction is not None:
            angle = self.target_direction
        else:
            angle = np.random.uniform(0, 2 * np.pi)

        self.x += np.random.uniform(0,self.speed.value) * np.cos(angle)
        self.y += np.random.uniform(0,self.speed.value) * np.sin(angle)
        
        # Keep creature within bounds
        self.x = max(self.speed.value/2, min(self.x, self.map_width))
        self.y = max(self.speed.value/2, min(self.y, self.map_height))


        #  lose energy according to speed
        self.energy -= 0.25+(np.square(self.speed.value)/3200)
        if self.energy <= 0:
            self.dead = True

        #   clear memory
        self.target_direction = None


    def eat(self, food):
        # Your logic to eat food and gain energy here
        pass

    def reproduce(self):
        # Your logic to reproduce and split into two here
        pass

    def mutate(self):
        # Call gene mutation functions
        self.speed.mutate()
        self.visibility.mutate()
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
        # print("in gene")
        # print(self.value)

    def mutate(self):
        if np.random.choice([True, False], p=[self.mutation_probability, 1-self.mutation_probability]):
            self.value += np.random.normal(scale=(self.value * self.mutation_rate))
            # print("Mutated, Value: " + str(self.value))

class Simulation:
    #def __init__(self, width, height, num_creatures, num_food, ticks, generations):
    def __init__(self, width, height, initial_food, initial_creatures, food_energy, creature_speed_variance,
                 food_spawn_rate, ticks, generations, max_food,creature_avg_speed,eat_radius, plot, write_csv, csv_loc):
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

        self.plotAnimation = plot  # Boolean to determine whether to plot animation or not
        self.write_csv = write_csv  # Boolean to determine whether to write to csv or not
        self.csv_loc = csv_loc  # Locaiton of csv file to write to
        self.writtenwaiter = False  # Boolean to make sure it writes only once
        

        self.meanspeed = []  # mean value of speeds of creatures in the simulation
        self.pop_stats = []  # List to store population count
        self.food_stats = []  # List to store food count
        self.tick_stats = []  # List to store tick count
        self.visibility_stats = []  # List to store visibility count
        self.oldest_age = 0  # Oldest age of a creature in the simulation
        self.mean_age_stats = []  # mean age of creatures in the simulation
        self.mean_age = 0  # mean age of creatures in the simulation

        for _ in range(initial_creatures):
            creature = Creature(np.random.uniform(0, self.width), np.random.uniform(0, self.height), 
                    np.random.uniform(creature_avg_speed-(creature_speed_variance/2), creature_avg_speed+(creature_speed_variance/2)),
                    vision=15, map_width=self.width, map_height=self.height)

            self.creatures.append(creature)

        for _ in range(initial_food):
            food = Food(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            self.food.append(food)

    def run(self):
        #fig, ax = plt.subplots()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax3 = ax2.twinx()
        ax4 = ax2.twinx()
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
            ax4.clear()

            self.current_tick += 1
            if self.current_tick > self.ticks:
                self.current_tick = 0
                self.current_generation += 1

            new_creatures = []
            
            for creature in self.creatures:
                
                creature.observe(self.food)

                if creature.age != 0:
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
                    offspring = Creature(creature.x, creature.y, creature.speed.value, energy=75, vision=creature.visibility.value, map_width=self.width, map_height=self.height)
                    offspring.mutate()
                    # print("Born")
                    print(offspring.visibility.value)
                    new_creatures.append(offspring)

                    # Divide energy between parent and offspring
                    creature.energy = 75
                
                creature.age += 1

                if creature.age > self.oldest_age:
                    self.oldest_age = creature.age

                # Mark creature visible range
                circle = plt.Circle((creature.x, creature.y), radius=creature.visibility.value, fill=False, color='blue')
                ax1.add_artist(circle)
            
            # Add the new creatures to the simulation
            self.creatures.extend(new_creatures)

            

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

            self.meanspeed.append(stat.mean([obj.speed.value for obj in self.creatures]))
            self.tick_stats.append(self.total_tick)
            self.pop_stats.append(len(self.creatures))
            self.food_stats.append(len(self.food))
            self.visibility_stats.append(stat.mean([obj.visibility.value for obj in self.creatures]))
            self.mean_age_stats.append(stat.mean([obj.age for obj in self.creatures]))

            # Write out stats every 1000 ticks

            
            if self.plotAnimation:
                # annotate the energy levels
                for creature in self.creatures:
                    ax1.annotate(f"{creature.energy:.1f}", (creature.x, creature.y))

                food_x = [food.x for food in self.food]
                food_y = [food.y for food in self.food]
                ax1.scatter(food_x, food_y, c='red')

                ax1.set_title(f"Epoch: {self.current_generation}, Tick: {self.current_tick}, Population: {len(self.creatures)}, Food: {len(self.food)}, Oldest: {self.oldest_age}")

                ax1.set_xlim(0, self.width)
                ax1.set_ylim(0, self.height)

                # Only plot the last 1000 data points if available
                if len(simulation.tick_stats) >= 10:
                    plothist = min(self.total_tick,1000)
                    ax2.plot(self.tick_stats[-plothist:], self.pop_stats[-plothist:], label='Population')
                    ax3.plot(self.tick_stats[-plothist:], self.food_stats[-plothist:], label='Food', color='red')
                    ax4.plot(self.tick_stats[-plothist:], self.meanspeed[-plothist:], label='meanSpeed', color = 'green')
        
                    #show y labels
                    ax2.set_ylabel("Population")
                    ax3.set_ylabel("Food")

                    #set y axis color
                    ax2.tick_params(axis='y', colors='blue')
                    ax3.tick_params(axis='y', colors='red')
                    ax4.tick_params(axis='y', colors='green')

                    ax2.set_ylim(0, max(self.pop_stats)+2)
                    ax3.set_ylim(0, max(self.food_stats)+2)

            
            if self.total_tick % 1000 == 50:
                self.writtenwaiter = False

            if self.write_csv & (self.total_tick % 1000 == 0) & (not self.writtenwaiter):
                data = {
                    "tick": self.tick_stats[-1000:],
                    "population": self.pop_stats[-1000:],
                    "food": self.food_stats[-1000:],
                    "meanSpeed": self.meanspeed[-1000:],
                    "meanVisibility": self.visibility_stats[-1000:],
                    "meanAge": self.mean_age_stats[-1000:]}
                    
                df = pd.DataFrame(data)
                print(self.total_tick,self.total_tick % 1000)
                if self.total_tick == 1:
                    print("Initialising CSV")
                    df.to_csv(f"{self.csv_loc}/output.csv", index=False , header=True)
                else:
                    # print("Writing CSV")
                    df.to_csv(f"{self.csv_loc}/output.csv", mode='a', header=False, index=False)
                    self.writtenwaiter = True


        if self.plotAnimation:
            ani = animation.FuncAnimation(fig, animate, frames=self.ticks*self.generations, interval=10)
            # set the window size for the animation
            fig.set_size_inches(20, 10)
            
            plt.show()

        


simulation = Simulation(
    width=1000,
    height=1000,
    initial_food=500,
    initial_creatures=10,
    food_energy=30,
    creature_avg_speed=15,
    creature_speed_variance=1,
    food_spawn_rate=1,
    ticks=100,
    generations=10,
    max_food=1000,
    eat_radius=10,
    plot = True,
    write_csv = False,
    # csv_loc = "D:/Simulations/outputs"
    csv_loc="/home/nator/Desktop"
)

simulation.run()