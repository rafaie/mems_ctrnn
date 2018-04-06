from mems_ctrnn import MEMS_CTRNN
import matplotlib.pyplot as plt
import numpy as np
import csv

max_variation = 0.001


def gen_input(t):
    i1 = 80 if t > 0.0020 else 0
    i2 = 180 if t > 0.00400 else 0
    i2 = i2 if t < 0.00800 else 0

    return (i1, i2)


def calculate_with_stablizing(c, a, b, l, i, i1, i2,
                              step_size, normalized):
    c.set_neuron_external_input(0, i1)
    c.set_neuron_external_input(1, i2)
    c.euler_step(step_size, normalized=normalized)

    # print(abs(a[(i - l) % l] - c.states[-2]),
    #       abs(b[(i - l) % l] - c.states[-1]),
    #       i, l, a, b)

    if i >= l and \
       abs(a[(i - l) % l] - c.states[-2]) < max_variation and \
       abs(b[(i - l) % l] - c.states[-1]) < max_variation:
        t = 'nondimensional' if normalized is False else 'dimensional'
        print('The MEMCTRN {} is stablized in iteration {} for '.format(t, i) +
              'input1 = {}, input2 = {} and max_variation= {}'.
              format(i1, i2, max_variation))

        return True
    elif i >= l:
        a[(i - l) % l] = c.states[-2]
        b[(i - l) % l] = c.states[-1]

    return False


if __name__ == "__main__":
    f_name = 'sample_2n.ns'

    c = MEMS_CTRNN()
    c.load(f_name)

    c1 = MEMS_CTRNN()
    c1.load(f_name)

    c2 = MEMS_CTRNN()
    c2.load(f_name)

    c.print_model()
    c.print_model_abstract()
    mem_wm = 131675.65242702136
    run_duration = 0.01
    step_size = 1/mem_wm

    with open('debug.csv', 'w') as fi:
        csv_file = csv.writer(fi, delimiter=',')
        csv_file.writerow(['time', 'c_type', 'input1',
                           'input2', 'state1', 'state2', 'stable'])

        in1 = []
        in2 = []
        out1 = []
        out2 = []
        out1_1 = []
        out2_1 = []
        out1_2 = []
        out2_2 = []

        l = 3
        a1 = [0] * l
        b1 = [0] * l
        a2 = [0] * l
        b2 = [0] * l

        it1 = 0
        it2 = 0

        c1_stable = False
        c2_stable = False

        i1_temp = -100000
        i2_temp = -10000

        # Normal situation
        for time in np.arange(0.0, run_duration + step_size, step_size):
            i1, i2 = gen_input(time)

            in1.append((time, i1))
            in2.append((time, i2))

            # Normal calculation
            c.set_neuron_external_input(0, i1)
            c.set_neuron_external_input(1, i2)
            c.euler_step(1)
            out1.append((time, c.states[0]))
            out2.append((time, c.states[1]))
            csv_file.writerow([time, 'Normal', i1, i2, c.states[0],
                               c.states[1], -1])

            # Calculation for non dimentional
            if i1 != i1_temp or i2 != i2_temp or c1_stable is False:
                c1_stable = calculate_with_stablizing(c1, a1, b1, l, it1,
                                                      i1, i2, 1, False)
                if c1_stable is True:
                    it1 = 0
                    a1 = [0] * l
                    b1 = [0] * l
                else:
                    it1 += 1

            out1_1.append((time, c1.states[0]))
            out2_1.append((time, c1.states[1]))
            csv_file.writerow([time, 'Non-dimensional', i1, i2, c1.states[0],
                               c1.states[1],
                               1 if c1_stable is True else 0])

            # Calculation for dimentional
            if i1 != i1_temp or i2 != i2_temp or c2_stable is False:
                c2_stable = calculate_with_stablizing(c2, a2, b2, l, it2,
                                                      i1, i2, step_size, True)
                if c2_stable is True:
                    it2 = 0
                    a2 = [0] * l
                    b2 = [0] * l
                else:
                    it2 += 1

            out1_2.append((time, c2.states[0]))
            out2_2.append((time, c2.states[1]))
            csv_file.writerow([time, 'Dimensional', i1, i2, c2.states[0],
                               c2.states[1],
                               1 if c2_stable is True else 0])

            i1_temp = i1
            i2_temp = i2

        in1_np = np.array(in1)
        in2_np = np.array(in2)
        out1_np = np.array(out1)
        out2_np = np.array(out2)
        out1_1_np = np.array(out1_1)
        out2_1_np = np.array(out2_1)
        out1_2_np = np.array(out1_2)
        out2_2_np = np.array(out2_2)

        plt.subplot(411)
        plt.plot(in1_np[:, 0], in1_np[:, 1], "g-", label='Input 1')
        plt.plot(in2_np[:, 0], in2_np[:, 1], "y-", label='Input 2')
        plt.title('Input - The time is in Second and max_variation ' +
                  str(max_variation))
        plt.legend()
        plt.subplot(412)
        plt.plot(out1_np[:, 0], out1_np[:, 1], "r-", label='Output 1 - Normal')
        plt.plot(out2_np[:, 0], out2_np[:, 1], "b-", label='Output 2 - Normal')
        plt.legend()
        plt.subplot(413)
        plt.plot(out1_1_np[:, 0], out1_1_np[:, 1], "r-",
                 label='Output 1 - Non dim')
        plt.plot(out2_1_np[:, 0], out2_1_np[:, 1], "b-",
                 label='Output 2 - Non dim')
        plt.legend()
        plt.subplot(414)
        plt.plot(out1_2_np[:, 0], out1_2_np[:, 1], "r-",
                 label='Output 1 - dim')
        plt.plot(out2_2_np[:, 0], out2_2_np[:, 1], "b-",
                 label='Output 2 - dim')
        plt.legend()

    plt.show()
