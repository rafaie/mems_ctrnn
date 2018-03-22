from utils import sigmoid, inverse_sigmoid
import numpy as np
import yaml
import math


class MEMS_CTRNN:
    def __init__(self, conf_path, new_size=0):
        with open(conf_path, 'r') as fi:
            conf = yaml.load(fi)
            self.load_data(conf)
            self.calc_params()
        self.set_circuit_size(new_size)

    def load_data(self, conf):
        conf_mems = conf['MEMS']
        self.mem_L = conf_mems['L']
        self.mem_b = conf_mems['b']
        self.mem_g0 = conf_mems['g0']
        self.mem_h = conf_mems['h']
        self.mem_E1 = conf_mems['E1']
        self.mem_nu = conf_mems['nu']
        self.mem_rho = conf_mems['rho']
        self.mem_c = conf_mems['c']
        self.mem_K = conf_mems['K']

        self.size = conf['Network']['size']

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
        # self.mem_theta = 1.0378584523852825 * self.mem_h * \
        #     self.mem_wm ** 2 / self.mem_Sigma ** 2 / self.mem_g0
        # self.mem_Kstar = 1.0378584523852825 * self.mem_wm ** 2 / \
        #     self.mem_Sigma ** 2
        # self.mem_K3Old = 0.06486615327408016 * self.mem_A * self.mem_wm ** 2\
        #     / self.mem_Iyy / self.mem_Sigma ** 2
        # self.mem_K1 = self.mem_Kstar - self.mem_h ** 2 * self.mem_K3Old
        # self.mem_K3 = self.mem_g0**2 * self.mem_K3Old

    # Show the Model details
    def print_model(self):
        for i in range(self.size):
            print('Neuron Number :', i)
            print('taus:', self.taus[i])
            print('biases:', self.biases[i])
            print('gains:', self.gains[i])
            print('It\'s the Weights:')
            for j in range(self.size):
                print('Weight: ({}, {}) = {}'.format(i, j,
                                                     self.weights[i][j]))
            print('-----------------------------------------')

    # Show the Model details
    def print_model_abstract(self):
        o = ''
        t = ''
        b = ''
        g = ''
        w = ''
        r = ''
        e = ''
        s = ''
        for i in range(self.size):
            o += str(round(self.outputs[i], 9)) + ', '
            t += str(round(self.taus[i], 9)) + ', '
            r += str(round(self.Rtaus[i], 9)) + ', '
            b += str(round(self.biases[i], 9)) + ', '
            g += str(round(self.gains[i], 9)) + ', '
            e += str(round(self.external_inputs[i], 9)) + ', '
            s += str(round(self.states[i], 9)) + ', '
            for j in range(self.size):
                w += str(round(self.weights[i][j], 9)) + ', '
            w += '\n'

        print("Output:", o)
        print("taus:", t)
        print("Rtaus:", r)
        print("biases:", b)
        print("gain:", g)
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
        self.biases = np.full(new_size, 0.0, dtype=float)
        self.gains = np.full(new_size, 1.0, dtype=float)
        self.taus = np.full(new_size, 1.0, dtype=float)
        self.Rtaus = np.full(new_size, 1.0, dtype=float)
        self.external_inputs = np.full(new_size, 0.0, dtype=float)
        self.weights = np.full((new_size, new_size), 0.0, dtype=float)
        self.temp_states = np.full(new_size, 0.0, dtype=float)
        self.temp_outputs = np.full(new_size, 0.0, dtype=float)
        self.k1 = np.full(new_size, 0.0, dtype=float)
        self.k2 = np.full(new_size, 0.0, dtype=float)
        self.k3 = np.full(new_size, 0.0, dtype=float)
        self.k4 = np.full(new_size, 0.0, dtype=float)

    def neuron_state(self, i):
        return self.states[i]

    def set_neuron_state(self, i, value):
        self.states[i] = value
        self.outputs[i] = sigmoid(self.gains[i]*(self.states[i] +
                                  self.biases[i]))

    def neuron_output(self, i):
        return self.outputs[i]

    def set_neuron_output(self, i, value):
        self.outputs[i] = value
        self.states[i] = inverse_sigmoid(value)/self.gains[i] - \
            self.iases[i]

    def neuron_bias(self, i):
        return self.biases[i]

    def set_neuron_bias(self, i, value=None):
        if value is None:
            self.biases = i
        else:
            self.biases[i] = value

    def neuron_gain(self, i):
        return self.gains[i]

    def set_neuron_gain(self, i, value=None):
        if value is None:
            self.gains = i
        else:
            self.gains[i] = value

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

    def connection_weight(self, i, j):
        return self.weights[i][j]

    def set_connection_weight(self, i, j, value):
        self.weights[i][j] = value

    def lesion_neuron(self, n):
        for i in range(self.size):
            self.set_connection_weight(i, n, 0)
            self.set_connection_weight(n, i, 0)

    def set_center_crossing(self):
        for i in range(self.size):
            # Sum the input weights to this neuron
            input_weight = 0
            for j in range(self.size):
                input_weight += self.connection_weight(i, j)

            # Compute the corresponding ThetaStar
            tetha_star = -input_weight/2
            self.set_neuron_bias(i, tetha_star)

    def randomize_circuit_state(self, lb, ub, rs=None):
        if rs is None:
            for i in range(self.size):
                self.set_neuron_state(i, np.random.uniform(lb, ub))
        else:
            for i in range(self.size):
                self.set_neuron_state(i, rs.uniform(lb, ub))

    def randomize_circuit_output(self, lb, ub, rs=None):
        if rs is None:
            for i in range(self.size):
                self.set_neuron_output(i, np.random.uniform(lb, ub))
        else:
            for i in range(self.size):
                self.set_neuron_output(i, rs.uniform(lb, ub))

    # Integrate a circuit one step using 4th-order Runge-Kutta.
    def euler_step(self, step_size):
        # Update the state of all neurons.
        for i in range(self.size):
            inp = self.external_inputs[i]
            for j in range(self.size):
                inp += self.weights[j][i] * self.outputs[j]
            self.states[i] += step_size * self.Rtaus[i] * \
                (inp - self.states[i])

        # Update the outputs of all neurons.
        for i in range(self.size):
            self.outputs[i] = sigmoid(self.gains[i] *
                                      (self.states[i] + self.biases[i]))

    def RK4_step(self, step_size):
        # The first step.
        for i in range(self.size):
            inp = self.externalinputs[i]
            for j in range(self.size):
                inp += self.weights[j][i] * self.outputs[j]
            self.k1[i] = step_size * self.Rtaus[i] * (inp - self.states[i])
            self.temp_states[i] = self.states[i] + 0.5 * self.k1[i]
            self.temp_outputs[i] = sigmoid(self.gains[i] *
                                           (self.temp_states[i] +
                                            self.biases[i]))

        # The second step
        for i in range(self.size):
            inp = self.external_inputs[i]
            for j in range(self.size):
                inp += self.weights[j][i] * self.temp_outputs[j]
            self.k2[i] = step_size * self.Rtaus[i] * \
                (inp - self.temp_states[i])
            self.temp_states[i] = self.states[i] + 0.5 * self.k2[i]

        for i in range(self.size):
            self.temp_outputs[i] = sigmoid(self.gains[i] *
                                           (self.temp_states[i] +
                                            self.biases[i]))

        # The third step.
        for i in range(self.size):
            inp = self.external_inputs[i]
            for j in range(self.size):
                inp += self.weights[j][i] * self.temp_outputs[j]
            self.k3[i] = step_size * self.Rtaus[i] * \
                (inp - self.temp_states[i])
            self.temp_states[i] = self.states[i] + self.k3[i]

        for i in range(self.size):
            self.temp_outputs[i] = sigmoid(self.gains[i] * (
                                           self.temp_states[i] +
                                           self.biases[i]))

        # The fourth step.
        for i in range(self.size):
            inp = self.external_inputs[i]
            for j in range(self.size):
                inp += self.weights[j][i] * self.temp_outputs[j]
            self.k4[i] = step_size * self.Rtaus[i] * \
                (inp - self.temp_states[i])
            self.states[i] += (1.0/6.0) * self.k1[i] + (1.0/3.0) * \
                self.k2[i] + (1.0/3.0) * self.k3[i] + (1.0/6.0) * \
                self.k4[i]
            self.outputs[i] = sigmoid(self.gains[i] *
                                      (self.states[i] + self.biases[i]))

    # Input and output from file
    def load(self, path):
        with open(path, 'r') as fi:
            lines = fi.readlines()

            # Read the size
            self.size = int(lines[0])
            self.set_circuit_size(self.size)

            # Read the time constants
            d = lines[2].split()
            for i in range(self.size):
                self.taus[i] = d[i]
                self.Rtaus[i] = 1/self.taus[i]

            # Read the biases
            d = lines[4].split()
            for i in range(self.size):
                self.biases[i] = d[i]

            # Read the gains
            d = lines[6].split()
            for i in range(self.size):
                self.gains[i] = d[i]

            # Read the weights
            for i in range(self.size):
                d = lines[8+i].split()
                for j in range(self.size):
                    self.weights[i][j] = d[j]

    def save(self, path):
        with open(path, 'w') as fi:
            # Write the size
            fi.write(str(self.size) + '\n\n')

            # Write the time constants
            fi.write(' '.join([str(i) for i in self.taus]) + '\n\n')

            # Write the biases
            fi.write(' '.join([str(i) for i in self.biases]) + '\n\n')

            # Write the gains
            fi.write(' '.join([str(i) for i in self.gains]) + '\n\n')

            # Write the weights
            for i in range(self.size):
                fi.write(' '.join([str(i) for i in self.weights[i]]) +
                         '\n')
