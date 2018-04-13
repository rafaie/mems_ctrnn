import random
import csv

BODY_SIZE = 30.0
ENV_WIDTH = 400.0
ENV_MIN_HEIGHT = 200.0
ENV_MAX_HEIGHT = 400.0
ENV_BASE_HEIGHT = 200.0
ENV_MIN_X = -30
ENV_MAX_X = 30
ENV_OBJ_Y_MOVEMENT = 30
ENV_OBJ_X_MOVEMENT = 30
ENV_AGENT_X_MOVEMENT = 100

LINE = 1
CIRCLE = 2

HIGHT_DIST = 1
WIDTH_DIST = 7
WIDTH_DIST2 = 3
RANDOM_DATA_COUNT = 0


def gen_data(x1, x2, y2, type=CIRCLE):
    y1 = 0
    if type == CIRCLE:
        return (CIRCLE, x1, y1, x2, y2, x2, 0)
    else:
        if abs(x2 - x1) > BODY_SIZE * 2:
            return (LINE, x1, y1, x2, y2, x1, 0)
        elif x2 > 0:
            return (LINE, x1, y1, x2, y2, -int(ENV_WIDTH / 2 - BODY_SIZE/2), 0)
        else:
            return (LINE, x1, y1, x2, y2, int(ENV_WIDTH / 2 - BODY_SIZE/2), 0)


def get_random_X12Y():
    x1 = random.randint(-ENV_WIDTH / 2 + BODY_SIZE/2,
                        ENV_WIDTH / 2 - BODY_SIZE/2)
    x2 = random.randint(-ENV_WIDTH / 2 + BODY_SIZE/2,
                        ENV_WIDTH / 2 - BODY_SIZE/2)
    y = random.randint(ENV_MAX_HEIGHT - ENV_MIN_HEIGHT, ENV_MAX_HEIGHT)

    return (x1, x2, y)


if __name__ == '__main__':
    with open('dataset.csv', 'w') as fi:
        outfile_csv = csv.writer(fi, delimiter=',',
                                 quotechar="'", quoting=csv.QUOTE_MINIMAL)
        agent_step = 0 if WIDTH_DIST <= 1 else \
            ENV_AGENT_X_MOVEMENT / (WIDTH_DIST-1.0)
        obj_step_x = 0 if WIDTH_DIST2 <= 1 else \
            ENV_OBJ_X_MOVEMENT / (WIDTH_DIST2-1.0)
        obj_step_y = 0 if HIGHT_DIST <= 1 else \
            ENV_OBJ_Y_MOVEMENT / (HIGHT_DIST-1.0)

        for i in range(WIDTH_DIST):
            x1 = int(-ENV_AGENT_X_MOVEMENT / 2.0 + i * agent_step)

            for j in range(WIDTH_DIST2):
                x2 = int(-ENV_OBJ_X_MOVEMENT / 2.0 + j * obj_step_x)

                for k in range(HIGHT_DIST):
                    y = ENV_BASE_HEIGHT if HIGHT_DIST <= 1 else \
                        int(ENV_BASE_HEIGHT - ENV_OBJ_Y_MOVEMENT / 2.0 +
                            k * obj_step_y)
                    print(x1, x2, y)
                    outfile_csv.writerow(gen_data(x1, x2, y, LINE))
                    outfile_csv.writerow(gen_data(x1, x2, y, CIRCLE))

        for i in range(RANDOM_DATA_COUNT):
            x1, x2, y = get_random_X12Y()
            print('Random', x1, x2, y)
            outfile_csv.writerow(gen_data(x1, x2, y, LINE))
            outfile_csv.writerow(gen_data(x1, x2, y, CIRCLE))
