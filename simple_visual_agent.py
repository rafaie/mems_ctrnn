from visual_object import Line, Circle
from visual_agent import VisualAgent
import os
import sys
import random
import time
import csv

RANDOM_SEED = 1


# The main program
def run_process(outfile_csv, step_size, X1, Y1, X2, Y2, is_circle=False,
                show_details=False, path=None):
    agent = VisualAgent()
    obj = Line()
    obj_id = 1
    if is_circle is True:
        obj = Circle()
        obj_id = 2

    if path is None:
        path = "categorize.ns"

    if os.path.exists(path) is False:
        print('The network file is Not exit')
        sys.exit(1)

    agent.nervous_system.load(path)

    # Run the agent
    random.seed()
    agent.reset(0, Y1)
    agent.set_positionX(X1)
    obj.set_positionX(X2)
    obj.set_positionY(Y2)

    timer = 0
    status = 0
    start_time = time.time()

    t = 0
    if show_details is True:
        agent.nervous_system.print_model_abstract()

    while obj.positionY() > VisualAgent.BODY_SIZE/2:
        t += step_size
        timer += 1
        if show_details is True:
            print("------------------")
            print(agent.positionX(), agent.positionY())
            print(obj.positionX(), obj.positionY())
        outfile_csv.writerow([obj_id, timer, step_size, X1, Y1, X2, Y2,
                              agent.positionX(), agent.positionY(),
                              obj.positionX(), obj.positionY(), status])
        status = 1
        agent.step(step_size, obj, show_details=show_details)
        obj.step(step_size)
        if show_details is True:
            agent.nervous_system.print_model_abstract()

    status += 1
    end_time = time.time()
    if show_details is True:
        print('finished computation at', end_time, ', elapsed time: ',
              end_time - start_time)

    outfile_csv.writerow([obj_id, timer, step_size, X1, Y1, X2, Y2,
                          agent.positionX(), agent.positionY(),
                          obj.positionX(), obj.positionY(), status])


if __name__ == "__main__":
    outfile = open('output.csv', 'w')
    outfile_csv = csv.writer(outfile, delimiter=',',
                             quotechar="'", quoting=csv.QUOTE_MINIMAL)
    outfile_csv.writerow(['obj_type', 'timer', 'step_size', 'X1', 'Y1',
                          'X2', 'Y2', 'agent_X', 'agent_Y', 'obj_X',
                          'obj_Y', 'status'])

    dataset = []
    with open('dataset.csv', 'r') as fi:
        csv_file = csv.reader(fi, delimiter=',',
                              quotechar="'", quoting=csv.QUOTE_MINIMAL)
        for row in csv_file:
            dataset.append([float(i) for i in row])

    for d in dataset:
        print(d)
        is_circle = True if int(d[0]) == 2 else False
        run_process(outfile_csv, 0.1, d[1], d[2], d[3], d[4],
                    show_details=False, is_circle=is_circle,
                    path="models/sample.ns")

    # run_process(outfile_csv, 0.1, -20, 0, 0, 200, show_details=True)
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # run_process(outfile_csv, 0.1, -20, 0, 0, 200, is_circle=True,
    #             show_details=True)

    outfile.close()
