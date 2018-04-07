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
import sys
from visual_object import Line, Circle
from visual_agent import VisualAgent
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm

__author__ = "Mostafa Rafaie"
__license__ = "APLv2"

# CONSTANT
LINE = 1
CIRCLE = 2
STEP_SIZE = 0.1
MEM_STEP_SIZE = 1
MODEL_SIZE = 14
MAX_DISTANCE = 100.0


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


def create_agent(genom, show_details=False):
    genom = list(genom)

    if len(genom) - 1 != MODEL_SIZE * (MODEL_SIZE + 4) + 4:
        raise NameError('There is an Error in training configuration!!')

    agent = VisualAgent(MODEL_SIZE, agent_vel_x=agent_vel_x,
                        stability_acc=stability_acc,
                        stability_hist_bucket=stability_hist_bucket,
                        stability_min_iteration=stability_min_iteration,
                        stability_max_iteration=stability_max_iteration)

    nervous_system = agent.nervous_system
    nervous_system.set_circuit_size(MODEL_SIZE)

    for i in range(MODEL_SIZE):
        nervous_system.hs[i] = genom[i * (MODEL_SIZE + 4)]
        nervous_system.v_biases[i] = genom[i * (MODEL_SIZE + 4) + 1]
        for j in range(MODEL_SIZE):
            v = genom[i * (MODEL_SIZE + 4) + j + 2]
            nervous_system.set_connection_weight(i, j, v)

        if i < 7:
            c = (MODEL_SIZE + 4) * i + MODEL_SIZE + 2
            nervous_system.inp_alpha[i] = genom[c]
            nervous_system.inp_beta[i] = genom[c+1]

    c = MODEL_SIZE * (MODEL_SIZE + 4)
    nervous_system.out_alpha = [genom[c], genom[c + 2]]
    nervous_system.out_beta = [genom[c + 1], genom[c + 3]]

    nervous_system.mem_L = float(MEMS_info['L'])
    nervous_system.mem_b = float(MEMS_info['b'])
    nervous_system.mem_g0 = float(MEMS_info['g0'])
    nervous_system.mem_d = float(MEMS_info['d'])
    nervous_system.mem_h = float(MEMS_info['h'])
    nervous_system.mem_E1 = float(MEMS_info['E1'])
    nervous_system.mem_nu = float(MEMS_info['nu'])
    nervous_system.mem_rho = float(MEMS_info['rho'])
    nervous_system.mem_c = float(MEMS_info['c'])
    nervous_system.mem_K = float(MEMS_info['K'])
    nervous_system.mem_ythr = float(MEMS_info['ythr'])
    nervous_system.mem_state_stopper = float(MEMS_info['state_stopper'])
    nervous_system.step_size = MEM_STEP_SIZE
    nervous_system.calc_params()

    if show_details is True:
        nervous_system.print_model_abstract()

    return agent


def run_process(data, agent, show_details=False, outfile_csv=None):
    obj_id = data[0]
    x1 = data[1]
    y1 = data[2]
    x2 = data[3]
    y2 = data[4]

    if obj_id == LINE:
        obj = Line(vy=obj_vel_y)
    else:
        obj = Circle(vy=obj_vel_y)

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

        if outfile_csv is not None:
            outfile_csv.writerow([obj_id, timer, STEP_SIZE, x1, y1, x2, y2,
                                  agent.positionX(), agent.positionY(),
                                  obj.positionX(), obj.positionY(), status])
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

    dist = min(math.fabs(agent.positionX() - obj.positionX()), MAX_DISTANCE)

    f = dist/MAX_DISTANCE
    if obj_id == CIRCLE:
        f = 1 - f
    f = math.pow(f, 1.5)

    dist2 = 0

    if outfile_csv is not None:
        outfile_csv.writerow([obj_id, timer, STEP_SIZE, x1, y1, x2, y2,
                              agent.positionX(), agent.positionY(),
                              obj.positionX(), obj.positionY(), status])

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
    logger.info('mean = {} and median = {} => {}'.format(np.mean(fitness),
                np.median(fitness), fitness))
    return np.mean(fitness)


# Save the best 10 models!
def save_models(population):
    for i in range(saved_model_count):
        agent = create_agent(population[i])
        agent.nervous_system.save('models/model_' + str(i) + '.ns')
        logger.info('i = {} and genom = {}'.format(i, population[i]))

    fi_name = population_np_path
    np.save(fi_name, np.array(population))

    p = np.load(fi_name)
    logger.info(p)


def load_config(path):
    print(path)
    global STEP_SIZE, genom_struct_path, init_population_size
    global population_size, mutation_rate, num_iteratitions
    global crossover_type, fitness_goal, STEP_SIZE, log_enable
    global cuncurrency, saved_model_count, MEMS_info, mid_neurons_count
    global MODEL_SIZE, population_np_path, reload_np_population_rate
    global agent_vel_x, obj_vel_y
    global stability_acc, stability_hist_bucket, stability_min_iteration
    global stability_max_iteration

    with open(path, 'r') as fi:
        yaml_data = yaml.load(fi)

        MEMS_info = yaml_data['MEMS']

        training_conf = yaml_data['Training']

        genom_struct_path = training_conf['genom_struct_path']
        init_population_size = training_conf['init_population_size']
        population_size = training_conf['population_size']
        mutation_rate = training_conf['mutation_rate']
        num_iteratitions = training_conf['num_iteratitions']
        crossover_type = training_conf['crossover_type']
        fitness_goal = training_conf['fitness_goal']
        STEP_SIZE = training_conf['STEP_SIZE']
        cuncurrency = training_conf['cuncurrency']
        log_enable = training_conf['log_enable']
        saved_model_count = training_conf['saved_model_count']
        mid_neurons_count = int(training_conf['mid_neurons_count'])
        MODEL_SIZE = 7 + mid_neurons_count + 2

        population_np_path = training_conf['population_np_path']
        reload_np_population_rate = float(training_conf[
                                        'reload_np_population_rate'])

        agent_vel_x = float(training_conf['agent_vel_x'])
        obj_vel_y = float(training_conf['obj_vel_y'])

        stability_acc = float(training_conf['stability_acc'])
        stability_hist_bucket = int(training_conf['stability_hist_bucket'])
        stability_min_iteration = int(training_conf['stability_min_iteration'])
        stability_max_iteration = int(training_conf['stability_max_iteration'])


def do_training():
    print (genom_struct_path, MEMS_info, cuncurrency)

    ga = GeneticAlgorithm(genom_struct_path)
    start_time = time.time()
    population = ga.run(init_population_size, population_size,
                        mutation_rate, num_iteratitions, crossover_type,
                        calc_fitness, fitness_goal,
                        cuncurrency=cuncurrency,
                        reverse_fitness_order=True,
                        population_np_path=population_np_path,
                        reload_np_population_rate=reload_np_population_rate)

    save_models(population)
    end_time = time.time()
    print(population[:3].astype(float))
    print(population[:, -1].astype(float))
    print('Runtime :', end_time - start_time)


def calc_fitness_for_model(model_path):
    fitness = []
    agent = VisualAgent(MODEL_SIZE, agent_vel_x=agent_vel_x)
    agent.nervous_system.load(model_path)
    agent.nervous_system.print_model_abstract()
    agent.nervous_system.print_model()

    data2 = []

    with open('output.csv', 'w') as outfile:
        outfile_csv = csv.writer(outfile, delimiter=',',
                                 quotechar="'", quoting=csv.QUOTE_MINIMAL)
        outfile_csv.writerow(['obj_type', 'timer', 'step_size', 'X1', 'Y1',
                              'X2', 'Y2', 'agent_X', 'agent_Y', 'obj_X',
                              'obj_Y', 'status'])

        for data in dataset:
            o = run_process(data, agent, outfile_csv=outfile_csv)
            f = o[-1]
            fitness.append(f)
            data2.append(data + o)

    logger.info('data2 = {} '.format(data2))
    logger.info('=> {}'.format(fitness))
    logger.info('mean = {} and median = {}'.format(np.mean(fitness),
                np.median(fitness)))


if __name__ == "__main__":
    config_path = ''
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    load_config(config_path)

    if log_enable is True:
        # Load logger
        global logger
        logging.config.dictConfig(
            yaml.load(open('logging.yaml')))
        logger = logging.getLogger(GeneticAlgorithm.LOGGER_HANDLER_NAME)

    load_dataset()

    if len(sys.argv) <= 2:
        do_training()
    else:
        model_path = sys.argv[2]
        calc_fitness_for_model(model_path)
