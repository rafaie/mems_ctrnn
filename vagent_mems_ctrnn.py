from mems_ctrnn import MEMS_CTRNN
import numpy as np
import csv
import time


class VAgent_MEMS_CTRNN(MEMS_CTRNN):

    def __init__(self, new_size=0, stability_acc=0.001,
                 stability_hist_bucket=3, stability_min_iteration=7,
                 stability_max_iteration=150):

        self.stability_acc = stability_acc
        self.stability_hist_bucket = stability_hist_bucket
        self.stability_min_iteration = stability_min_iteration
        self.stability_max_iteration = stability_max_iteration

        MEMS_CTRNN.__init__(self, new_size)

    def print_vagent_variables(self):
        i_a = ', '.join([str(i) for i in self.inp_alpha])
        i_b = ', '.join([str(i) for i in self.inp_beta])
        o_a = ', '.join([str(i) for i in self.out_alpha])
        o_b = ', '.join([str(i) for i in self.out_beta])
        print("inp_alpha:", i_a)
        print("inp_beta:", i_b)
        print("out_alpha:", o_a)
        print("out_beta:", o_b)

    # Show the Model details
    def print_model(self):
        MEMS_CTRNN.print_model(self)
        self.print_vagent_variables()

    # Show the Model details
    def print_model_abstract(self):
        MEMS_CTRNN.print_model_abstract(self)
        self.print_vagent_variables()

    def set_circuit_size(self, new_size):
        MEMS_CTRNN.set_circuit_size(self, new_size)
        self.inp_alpha = np.full(7, 1.0, dtype=float)
        self.inp_beta = np.full(7, 0.0, dtype=float)
        self.out_alpha = np.full(2, 1.0, dtype=float)
        self.out_beta = np.full(2, 0.0, dtype=float)
        self.outputs = np.full(2, 0.0, dtype=float)

    def euler_step_with_stability(self, step_size=None, use_dim_equation=False,
                                  save_detail=False,
                                  use_defelection_feedback=False,
                                  return_states_info=False):
        states_info = []

        if save_detail is True:
            outfile = open('duration_analysis.csv', 'a')
            outfile_csv = csv.writer(outfile, delimiter=',',
                                     quotechar="'", quoting=csv.QUOTE_MINIMAL)

        a = [0] * self.stability_hist_bucket
        b = [0] * self.stability_hist_bucket
        l = self.stability_hist_bucket

        t = time.time()

        for i in range(self.stability_max_iteration):
            MEMS_CTRNN.euler_step(self, step_size, use_dim_equation,
                                  use_defelection_feedback)
            if save_detail is True:
                outfile_csv.writerow([t, i, step_size,
                                      self.states[-2], self.states[-1], '-',
                                      a[(i - l) % l],
                                      b[(i - l) % l], '-',
                                      (i - l) % l, '-',
                                      a[(i - l) % l] - b[(i - l) % l], '-',
                                      a, b])

            if return_states_info is True:
                states_info.append(list(self.states))

            if i >= self.stability_min_iteration and \
               abs(a[(i - l) % l] - self.states[-2]) < self.stability_acc and \
               abs(b[(i - l) % l] - self.states[-1]) < self.stability_acc:
                # print(i)
                break
            elif i >= l:
                a[(i - l) % l] = self.states[-2]
                b[(i - l) % l] = self.states[-1]

        if i >= 10:
            print(i)

        if save_detail is True:
            outfile.close()

        return states_info

    # Integrate a circuit one step using 4th-order Runge-Kutta.
    def euler_step(self, step_size=None, use_dim_equation=False,
                   save_detail=False, use_defelection_feedback=False,
                   return_states_info=False):
        for i in range(7):
            self.external_inputs[i] = self.external_inputs[i] * \
                 self.inp_alpha[i] + self.inp_beta[i]

        state_info = self.euler_step_with_stability(step_size,
                                                    use_dim_equation,
                                                    save_detail,
                                                    use_defelection_feedback,
                                                    return_states_info)

        for i in range(2):
            self.outputs[i] = self.states[self.size - 2 + i] * \
                self.out_alpha[i] + self.out_beta[i]

        return state_info

    # Input and output from file
    def load(self, path):
        with open(path, 'r') as fi:
            lines = fi.readlines()

            # Read the size
            self.size = int(lines[0])
            self.set_circuit_size(self.size)
            self.step_size = float(lines[2])

            # Read Mems Parameteres
            self.mem_L = float(lines[4])
            self.mem_b = float(lines[6])
            self.mem_g0 = float(lines[8])
            self.mem_d = float(lines[10])
            self.mem_h = float(lines[12])
            self.mem_E1 = float(lines[14])
            self.mem_nu = float(lines[16])
            self.mem_rho = float(lines[18])
            self.mem_c = float(lines[20])
            self.mem_K = float(lines[22])
            self.mem_ythr = float(lines[24])
            self.mem_state_stopper = float(lines[26])

            # Read the time constants
            d = lines[28].split()
            for i in range(self.size):
                self.taus[i] = d[i]
                self.Rtaus[i] = 1/self.taus[i]

            # Read the v_biases
            d = lines[30].split()
            for i in range(self.size):
                self.v_biases[i] = d[i]

            # Read the h's
            d = lines[32].split()
            for i in range(self.size):
                self.hs[i] = d[i]

            # Read the weights
            for i in range(self.size):
                d = lines[34+i].split()
                for j in range(self.size):
                    self.weights[i][j] = d[j]

            n = 34 + self.size + 1
            print(n)
            # Read the inp_alpha
            d = lines[n].split()
            for i in range(7):
                self.inp_alpha[i] = d[i]

            # Read the inp_beta
            d = lines[n + 2].split()
            for i in range(7):
                self.inp_beta[i] = d[i]

            # Read the out_alpha
            d = lines[n + 4].split()
            for i in range(2):
                self.out_alpha[i] = d[i]

            # Read the out_beta
            d = lines[n + 6].split()
            for i in range(2):
                self.out_beta[i] = d[i]

            self.calc_params()

    def save(self, path):
        with open(path, 'w') as fi:
            # Write the size
            fi.write(str(self.size) + '\n\n')
            fi.write(str(self.step_size) + '\n\n')

            # Write the Mems Parameteres
            fi.write(str(self.mem_L) + '\n\n')
            fi.write(str(self.mem_b) + '\n\n')
            fi.write(str(self.mem_g0) + '\n\n')
            fi.write(str(self.mem_d) + '\n\n')
            fi.write(str(self.mem_h) + '\n\n')
            fi.write(str(self.mem_E1) + '\n\n')
            fi.write(str(self.mem_nu) + '\n\n')
            fi.write(str(self.mem_rho) + '\n\n')
            fi.write(str(self.mem_c) + '\n\n')
            fi.write(str(self.mem_K) + '\n\n')
            fi.write(str(self.mem_ythr) + '\n\n')
            fi.write(str(self.mem_state_stopper) + '\n\n')

            # Write the time constants
            fi.write(' '.join([str(i) for i in self.taus]) + '\n\n')

            # Write the biases
            fi.write(' '.join([str(i) for i in self.v_biases]) + '\n\n')

            # Write the gains
            fi.write(' '.join([str(i) for i in self.hs]) + '\n\n')

            # Write the weights
            for i in range(self.size):
                fi.write(' '.join([str(i) for i in self.weights[i]]) +
                         '\n')
            fi.write('\n')

            # Write the inp_alpha
            fi.write(' '.join([str(i) for i in self.inp_alpha]) + '\n\n')

            # Write the inp_beta
            fi.write(' '.join([str(i) for i in self.inp_beta]) + '\n\n')

            # Write the out_alpha
            fi.write(' '.join([str(i) for i in self.out_alpha]) + '\n\n')

            # Write the out_beta
            fi.write(' '.join([str(i) for i in self.out_beta]) + '\n\n')
