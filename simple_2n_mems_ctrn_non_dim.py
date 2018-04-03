from mems_ctrnn import MEMS_CTRNN
import matplotlib.pyplot as plt
import numpy as np
import csv


def gen_input(t, step_size):
    # i1 = 80 if t > 0.2 else 0
    # i2 = 180 if t > 0.4 else 0
    # i2 = i2 if t < 0.8 else 0

    i1 = 400 * t - 80 if t > 0.2 and t <= 0.4 else 0
    i1 = 186.69 - 266.8 * t if t > 0.4 and t < 0.7 else i1
    i2 = 0

    return (i1, i2)


if __name__ == "__main__":
    f_name = 'sample_2n.ns'

    c = MEMS_CTRNN()
    c.load(f_name)

    c.print_model()
    c.print_model_abstract()
    run_duration = 300
    step_size = 0.01 # 0.1/131675.65242702136

    in1 = []
    in2 = []
    out1 = []
    out2 = []
    v_mem1 = []
    v_mem2 = []

    with open('debug.csv', 'w') as fi:
        csv_file = csv.writer(fi, delimiter=',')
        csv_file.writerow(['time', 'input1', 'input2', 'state1', 'state2'])

        for time in np.arange(0.0, 1 + step_size, step_size):
            i1, i2 = gen_input(time, step_size)
            c.set_neuron_external_input(0, i1)
            c.set_neuron_external_input(1, i2)
            c.euler_step(step_size)

            in1.append((time, i1))
            in2.append((time, i2))
            out1.append((time, c.states[0]))
            out2.append((time, c.states[1]))

            v_mem = c.external_inputs[0] + c.v_biases[0]
            for j in range(c.size):
                v_mem += c.weights[j][0] * c.v_outs[j]
            v_mem1.append((time, v_mem))

            v_mem = c.external_inputs[1] + c.v_biases[1]
            for j in range(c.size):
                v_mem += c.weights[j][1] * c.v_outs[j]
            v_mem2.append((time, v_mem))

            csv_file.writerow([time, i1, i2, c.states[0], c.states[1]])

    print(out1, out2)
    in1_np = np.array(in1)
    in2_np = np.array(in2)
    out1_np = np.array(out1)
    out2_np = np.array(out2)
    v_mem1_np = np.array(v_mem1)
    v_mem2_np = np.array(v_mem2)
    plt.subplot(311)
    plt.plot(in1_np[:, 0], in1_np[:, 1], "g-", label='Input 1')
    plt.plot(in2_np[:, 0], in2_np[:, 1], "y-", label='Input 2')
    plt.legend()
    plt.subplot(312)
    plt.plot(out1_np[:, 0], out1_np[:, 1], "r-", label='Output 1')
    plt.plot(out2_np[:, 0], out2_np[:, 1], "b-", label='Output 2')
    plt.legend()
    plt.subplot(313)
    plt.plot(v_mem1_np[:, 0], v_mem1_np[:, 1], "r-", label='v_mem 1')
    plt.plot(v_mem2_np[:, 0], v_mem2_np[:, 1], "b-", label='v_mem 2')
    plt.legend()
    plt.show()
