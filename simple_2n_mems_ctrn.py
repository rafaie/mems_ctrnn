from mems_ctrnn import MEMS_CTRNN
import matplotlib.pyplot as plt
import numpy as np


def gen_input(t):
    i1 = 1 if t > 0.001 else 0
    i2 = 1 if t > 0.005 else 1

    return (i1, i2)


if __name__ == "__main__":
    f_name = 'sample_2n.ns'

    c = MEMS_CTRNN()
    c.load(f_name)

    c.print_model()
    c.print_model_abstract()
    run_duration = 30
    step_size = 0.0005

    out1 = []
    out2 = []

    for time in np.arange(0.0, run_duration, step_size):
        i1, i2 = gen_input(time)
        c.set_neuron_external_input(0, i1)
        c.set_neuron_external_input(0, i2)
        c.euler_step(step_size)

        out1.append((time, c.states[0]))
        out2.append((time, c.states[1]))

    print(out1, out2)
    out1_np = np.array(out1)
    out2_np = np.array(out2)

    plt.plot(out1_np[:, 0], out1_np[:, 1], "b-", label='Output 1')
    plt.plot(out2_np[:, 0], out2_np[:, 1], "r-", label='Output 2')
    plt.legend()
    plt.show()
