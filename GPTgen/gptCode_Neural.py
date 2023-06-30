import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation
import statistics as stat
import csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense



def create_brain():
    # Define your neural network model here
    model = Sequential()
    model.add(Dense(5, input_dim=4, activation='relu'))
    model.add(Dense(2, activation='linear'))
    return model

class Creature:
    def __init__(self, x, y, speed, movement_type, energy=100, map_width=100, map_height=100, brain=None):
        self.x = x
        self.y = y
        #self.speed = speed
        self.speed = Gene(speed, 0.1, 0.1)
        self.movement_type = movement_type
        self.energy = energy
        self.map_width = map_width
        self.map_height = map_height
        self.dead = False
        self.age = 0
        self.hasTarget = False
        self.target = []


    # Create a brain if none is provided
        if brain is None:
            self.brain = create_brain()
        else:
            self.brain = brain

    def observe(self, food):
        # Your logic to observe food here
        pass


    def decide_movement(self, food):
        # Your logic to decide movement here
        
        # The input to the brain should be a 1D array with 4 elements,
        # representing the x and y positions of the creature and the food
        brain_input = np.array([self.x, self.y, food.x, food.y])  # fill in with the actual values

        # The output of the brain is a 1D array with 2 elements,
        # representing the desired x and y components of the creature's movement
        target = self.brain.predict(np.array([brain_input]),verbose=0)[0]  # note the extra [] around brain_input
        self.hasTarget = True

        return target

        pass

    def decide_direction(self, target):

        target_theta = np.arctan2(target[1]-self.y, target[0]-self.x)
        target_distance = np.sqrt((target[1]-self.y)**2 + (target[0]-self.x)**2)
        return target_theta, target_distance


    def rand_move(self):
        if self.movement_type == 'linear':
        
            self.x = self.x + np.random.uniform(-self.speed.value, self.speed.value)
            self.y = self.y + np.random.uniform(-self.speed.value, self.speed.value)
            
        elif self.movement_type == 'angular':
            angle = np.random.uniform(0, 2 * np.pi)
            
            self.x += self.speed.value * np.cos(angle)
            self.y += self.speed.value * np.sin(angle)
        
        # Keep creature within bounds
        self.x = max(0, min(self.x, self.map_width))
        self.y = max(0, min(self.y, self.map_height))

        self.energy -= 1
        if self.energy <= 0:
            self.dead = True

    def intentional_move(self, target_theta, target_distance):
        self.x = self.x + self.speed.value * np.cos(target_theta)
        self.y = self.y + self.speed.value * np.sin(target_theta)
             
        # Keep creature within bounds
        self.x = max(0, min(self.x, self.map_width))
        self.y = max(0, min(self.y, self.map_height))

        self.energy -= 1
        if self.energy <= 0:
            self.dead = True

    def eat(self, food):
        # Your logic to eat food and gain energy here
        target = self.decide_movement(food)
        targ_theta, targ_dist = self.decide_direction(target)
        self.intentional_move(targ_theta, targ_dist)

        pass

    def reproduce(self):
        # Your logic to reproduce and split into two here
        pass

    def mutate(self):
        # Call gene mutation functions
        self.speed.mutate()
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
                 food_spawn_rate, ticks, generations, max_food,creature_avg_speed,eat_radius,see_radius, plot, write_csv, csv_loc):
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
        self.see_radius = see_radius # Radius in which creatures can see food

        self.plotAnimation = plot  # Boolean to determine whether to plot animation or not
        self.write_csv = write_csv  # Boolean to determine whether to write to csv or not
        self.csv_loc = csv_loc  # Locaiton of csv file to write to
        self.writtenwaiter = False  # Boolean to make sure it writes only once
        

        self.meanspeed = []  # mean value of speeds of creatures in the simulation
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
                
                if (creature.age != 0) & (creature.hasTarget == False):
                    creature.rand_move()

                # Check if creature is near food
                for food in self.food:
                    distance = np.sqrt((creature.x - food.x)**2 + (creature.y - food.y)**2)

                    # Check if the creature can see the food
                    if distance < self.see_radius:
                        creature.eat(food)


                    # If creature is near food, it eats it and gains energy
                    if distance < self.eat_radius:  # You can adjust this value as needed
                        creature.energy += food.energy
                        creature.hasTarget = False
                        self.food.remove(food)  # Remove food from simulation

                # Your logic here to reproduce
                # Check if creature has enough energy to reproduce
                if creature.energy >= 150:
                    # Create a new creature at the same location
                    offspring = Creature(creature.x, creature.y, creature.speed.value, creature.movement_type, energy=75, map_width=self.width, map_height=self.height)
                    offspring.mutate()
                    # print("Born")
                    # print(offspring.speed.value)
                    new_creatures.append(offspring)

                    # Divide energy between parent and offspring
                    creature.energy = 75
                
                creature.age += 1
            
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

            self.meanspeed.append(stat.mean([obj.speed.value for obj in self.creatures]))
            self.tick_stats.append(self.total_tick)
            self.pop_stats.append(len(self.creatures))
            self.food_stats.append(len(self.food))

            # Write out stats every 1000 ticks

            
            if self.plotAnimation:
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
                    "meanSpeed": self.meanspeed[-1000:]}
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
    width=100,
    height=100,
    initial_food=50,
    initial_creatures=5,
    food_energy=30,
    creature_avg_speed=15,
    creature_speed_variance=1,
    food_spawn_rate=1,
    ticks=100,
    generations=10,
    max_food=100,
    eat_radius=10,
    see_radius=50,
    plot = True,
    write_csv = True,
    csv_loc = "D:/Simulations/outputs"
)

simulation.run()