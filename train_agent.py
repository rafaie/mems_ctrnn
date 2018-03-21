"""
ga_function_optimization.py: Using genetic_algorithm for to find the best
parameters for the CTRNN

"""

import logging.config
import yaml
import numpy as np
import random
import time
import csv
import math
from visual_object import Line, Circle
from visual_agent import VisualAgent
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm

__author__ = "Mostafa Rafaie"
__license__ = "APLv2"

# CONSTANT
LINE = 1
CIRCLE = 2
STEP_SIZE = 0.1
MODEL_SIZE = 14


# dataset
dataset_path = 'dataset.csv'
dataset = []


def load_dataset():
    global dataset
    dataset = []

    with open('dataset.csv', 'r') as fi:
        csv_file = csv.reader(fi, delimiter=',',
                              quotechar="'", quoting=csv.QUOTE_MINIMAL)
        for row in csv_file:
            dataset.append([float(i) for i in row])


def create_agent(genom):
    agent = VisualAgent(MODEL_SIZE)
    nervous_system = agent.nervous_system
    nervous_system.set_circuit_size(MODEL_SIZE)

    nervous_system.set_neuron_time_constant(genom[:MODEL_SIZE])
    nervous_system.set_neuron_bias(genom[MODEL_SIZE * 1: 2 * MODEL_SIZE])
    nervous_system.set_neuron_gain(genom[MODEL_SIZE * 2: 3 * MODEL_SIZE])

    for i in range(MODEL_SIZE):
        for j in range(MODEL_SIZE):
            v = genom[(i + 3) * MODEL_SIZE + j]
            nervous_system.set_connection_weight(i, j, v)

    return agent


def run_process(data, agent, show_details=False):
    obj_id = data[0]
    x1 = data[1]
    y1 = data[2]
    x2 = data[3]
    y2 = data[4]
    goal_x = data[5]
    goal_y = data[6]

    if obj_id == LINE:
        obj = Line()
    else:
        obj = Circle()

    # Run the agent
    random.seed()
    agent.reset(0, y1)
    agent.set_positionX(x1)
    obj.set_positionX(x2)
    obj.set_positionY(y2)

    timer = 0
    status = 0
    start_time = time.time()

    t = 0
    if show_details is True:
        agent.nervous_system.print_model_abstract()

    while obj.positionY() > VisualAgent.BODY_SIZE/2:
        t += STEP_SIZE
        timer += 1
        if show_details is True:
            print("------------------")
            print(agent.positionX(), agent.positionY())
            print(obj.positionX(), obj.positionY())

        status = 1
        agent.step(STEP_SIZE, obj, show_details=show_details)
        obj.step(STEP_SIZE)
        if show_details is True:
            agent.nervous_system.print_model_abstract()

    status += 1
    end_time = time.time()
    if show_details is True:
        print('finished computation at', end_time, ', elapsed time: ',
              end_time - start_time)

    dist = math.sqrt((agent.positionX() - goal_x) ** 2 +
                     (agent.positionY() - goal_y) ** 2)

    dist2 = math.sqrt((agent.positionX() - obj.positionX()) ** 2 +
                      (agent.positionY() - obj.positionY()) ** 2)
    f = 300
    if obj_id == LINE:
        if dist2 > 30:
            f = (2000 - dist2) / 1000
        else:
            f = (30 - dist2) * 5
    else:
        if dist < 28:
            f = dist / 10
        else:
            f = (dist - 28) * 10

    return [agent.positionX(), agent.positionY(), obj.positionX(),
            obj.positionY(), dist, dist2, f]


# Using inverse function for fitness 1/F(x)
def calc_fitness(genom):
    fitness = []
    agent = create_agent(genom)
    data2 = []

    for data in dataset:
        o = run_process(data, agent)
        f = o[-1]
        fitness.append(f)
        data2.append(data + o)

    logger.info('data2 = {} '.format(data2))
    logger.info('mean = {} and median = {} '.format(np.mean(fitness),
                np.median(fitness)))
    return np.mean(fitness)


# Save the best 10 models!
def save_models(population):
    for i in range(10):
        agent = create_agent(population[i])
        agent.nervous_system.save('models/model_' + str(i) + '.ns')


if __name__ == "__main__":
    # Load logger
    global logger
    logging.config.dictConfig(
        yaml.load(open('logging.yaml')))
    logger = logging.getLogger(GeneticAlgorithm.LOGGER_HANDLER_NAME)

    path = 'genom_struct.csv'
    init_population_size = 6000
    population_size = 100
    mutation_rate = 0.20
    num_iteratitions = 100
    crossover_type = GeneticAlgorithm.TWO_POINT_CROSSOVER
    fitness_goal = 0.00001

    load_dataset()

    ga = GeneticAlgorithm(path)
    start_time = time.time()
    population = ga.run(init_population_size, population_size,
                        mutation_rate, num_iteratitions, crossover_type,
                        calc_fitness, fitness_goal,
                        cuncurrency=20,
                        reverse_fitness_order=False)
    save_models(population)
    end_time = time.time()
    print(population[:3].astype(float))
    print(population[:, -1].astype(float))
    print('Runtime :', end_time - start_time)
