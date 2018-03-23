from mems_ctrnn import MEMS_CTRNN


if __name__ == "__main__":
    f_name = 'sample_2n.ns'

    c = MEMS_CTRNN()
    c.load(f_name)

    c.print_model()
    c.print_model_abstract()
    # run_duration = 250
    # for time in range(int(run_duration/step_size)):
    #     c.euler_step(step_size)
    #     print(round(time*step_size, 2), c.neuron_output(0), c.neuron_output(1))
