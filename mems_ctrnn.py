import numpy as np
import math


class MEMS_CTRNN:
    def __init__(self, new_size=0):
        self.set_circuit_size(new_size)

    def calc_params(self):
        # Cross-sectional area
        self.mem_A = self.mem_d * self.mem_b

        # Effective young modulus
        self.mem_E = self.mem_E1 / (1-self.mem_nu ** 2)

        # Second moment of area I_yy
        self.mem_Iyy = self.mem_b * self.mem_d ** 3 / 12

        # Emessivity constant
        self.mem_eps = self.mem_K * 8.845187817620e-12

        # Straight beam natural frequency
        self.mem_wm = 22.3733 * math.sqrt(self.mem_E * self.mem_Iyy /
                                          self.mem_rho / self.mem_A /
                                          self.mem_L ** 4)

        self.mem_Sigma = self.mem_c / (self.mem_rho * self.mem_A)
        self.mem_Kstar = 1.0378584523852825 * self.mem_wm ** 2 / \
            self.mem_Sigma ** 2
        self.mem_K3Old = 0.06486615327408016 * self.mem_A * self.mem_wm ** 2\
            / self.mem_Iyy / self.mem_Sigma ** 2
        self.mem_K3 = self.mem_g0**2 * self.mem_K3Old
        self.mem_win = 2 / 3.0 * self.mem_b * self.mem_eps \
                         / (self.mem_A * self.mem_rho *
                            self.mem_Sigma ** 2 * self.mem_g0 ** 3)

    # Print MEMS Parameteres
    def print_mems_param(self):
        print(f'mem_L = {self.mem_L}, mem_b = {self.mem_b}, ' +
              f'mem_g0 = {self.mem_g0}, mem_d = {self.mem_d}')
        print(f'mem_h = {self.mem_h}, mem_E1 = {self.mem_E1}, ' +
              f'mem_nu = {self.mem_nu}, mem_rho = {self.mem_rho}, ')
        print(f'mem_h = {self.mem_c}, mem_E1 = {self.mem_K}' +
              f'mem_nu = {self.mem_ythr}, mem_rho = {self.mem_state_stopper}')
        print(f'mem_A = {self.mem_A}, mem_E = {self.mem_E}, ' +
              f'mem_Iyy = {self.mem_Iyy}, mem_eps = {self.mem_eps}')
        print(f'mem_wm = {self.mem_wm}, mem_Sigma = {self.mem_Sigma}, ' +
              f'mem_Kstar = {self.mem_Kstar}, mem_K3Old = {self.mem_K3Old}')
        print(f'mem_K3 = {self.mem_K3}, mem_win = {self.mem_win}')

    # Show the Model details
    def print_model(self):
        self.print_mems_param()
        print('-----------------------------------------')
        for i in range(self.size):
            print('Neuron Number :', i)
            print('taus:', self.taus[i])
            print('v_biases:', self.v_biases[i])
            print('hs:', self.hs[i])
            print('It\'s the Weights:')
            for j in range(self.size):
                print('Weight: ({}, {}) = {}'.format(i, j,
                                                     self.weights[i][j]))
            print('-----------------------------------------')

    # Show the Model details
    def print_model_abstract(self):
        t = ''
        r = ''
        v = ''
        h = ''
        e = ''
        s = ''
        w = ''

        for i in range(self.size):
            t += str(round(self.taus[i], 9)) + ', '
            r += str(round(self.Rtaus[i], 9)) + ', '
            v += str(round(self.v_biases[i], 9)) + ', '
            h += str(round(self.hs[i], 9)) + ', '
            e += str(round(self.external_inputs[i], 9)) + ', '
            s += str(round(self.states[i], 9)) + ', '
            for j in range(self.size):
                w += str(round(self.weights[i][j], 9)) + ', '
            w += '\n'

        self.print_mems_param()
        print("taus:", t)
        print("Rtaus:", r)
        print("v_biases:", v)
        print("hs:", h)
        print("external_inputs:", e)
        print("states:", s)
        print("weight:\n", w)

    # Accessors
    def circuit_size(self):
        return self.size

    def set_circuit_size(self, new_size):
        self.size = new_size
        self.states = np.full(new_size, 0.0, dtype=float)
        self.outputs = np.full(new_size, 0.0, dtype=float)
        self.v_biases = np.full(new_size, 0.0, dtype=float)
        self.v_outs = np.full(new_size, 0.0, dtype=float)
        self.hs = np.full(new_size, 1.0, dtype=float)
        self.taus = np.full(new_size, 1.0, dtype=float)
        self.Rtaus = np.full(new_size, 1.0, dtype=float)
        self.external_inputs = np.full(new_size, 0.0, dtype=float)
        self.weights = np.full((new_size, new_size), 0.0, dtype=float)

    def neuron_time_constant(self, i):
        return self.taus[i]

    def set_neuron_time_constant(self, i, value=None):
        if value is None:
            for j, v in enumerate(i):
                self.taus[j] = v
                self.Rtaus[j] = 1/v
        else:
            self.taus[i] = value
            self.Rtaus[i] = 1/value

    def neuron_external_input(self, i):
        return self.external_inputs[i]

    def set_neuron_external_input(self, i, value):
        self.external_inputs[i] = value

    # Integrate a circuit one step using 4th-order Runge-Kutta.
    def euler_step(self):
        # Calculate the v_0
        for i in range(self.size):
            if self.states[i] < self.mem_ythr:
                self.v_outs[i] = self.external_inputs[i] + self.v_biases[i]
            else:
                self.v_outs[i] = 0

        # Update the state of all neurons.
        for i in range(self.size):
            v_mem = self.external_inputs[i] + self.v_biases[i]
            for j in range(self.size):
                v_mem += self.weights[j][i] * self.v_outs[j]

            mem_theta = 1.0378584523852825 * self.hs[i] * \
                self.mem_wm ** 2 / self.mem_Sigma ** 2 / self.mem_g0
            k1 = self.mem_Kstar - self.hs[i] ** 2 * self.mem_K3Old

            self.states[i] += self.step_size * self.Rtaus[i] * \
                (-k1 * self.states[i] - self.mem_k3 * self.states[i] ** 3 +
                 mem_theta + self.mem_win * v_mem ** 2 /
                 math.sqrt((1 + self.states[i]) ** 3))

            if self.states[i] > self.mem_state_stopper:
                self.states[i] = self.mem_state_stopper

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
