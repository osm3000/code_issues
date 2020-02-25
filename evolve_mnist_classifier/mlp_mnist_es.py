import models
import torch
import torch.nn as nn
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import pygmo as pg

batch_size = 64
version = "1"
data_parameters = {}
data_parameters['window_size'] = 3
data_parameters['step_size'] = 0
data_parameters['horizonal_vertical'] = 'horizonal'
dataset_name = "mnist_784"
# dataset_name = "Fashion-MNIST"
X, y = pickle.load(open(f"{dataset_name}.pkl", 'rb'))
y = np.asarray(y, np.int16) # needed for pytorch

# shuffle and select only a tiny subset from the data (100 example per category for training)
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

learning_model = models.Simple_MLP(input_size=X_train.shape[1], hidden_size=50, nb_layers=1, output_size=10)


print ("Learning Model: \n", learning_model)

count_parameters = 0
for param in learning_model.parameters():
    count_parameters += int(np.product(param.size()))
print (f'nb of param = {count_parameters}')

loss_fn = torch.nn.NLLLoss()

class nn_evolve:
    def __init__(self, dim=28*28, min_limit=0, max_limit=1):
        self.dim = dim
        self.min_limit = min_limit
        self.max_limit = max_limit

    # @profile
    def fitness(self, weights_x):
        global learning_model
        global X
        global y
        nn.utils.vector_to_parameters(torch.from_numpy(weights_x).float(), learning_model.parameters())
        X_train_batch, y_train_batch = shuffle(X_train, y_train, n_samples=batch_size)
        X_train_batch = torch.tensor(X_train_batch, requires_grad=False).float()
        y_train_batch = torch.tensor(y_train_batch, requires_grad=False).long()

        output_train = learning_model(X_train_batch)
        train_loss = loss_fn(output_train, y_train_batch.view(-1))
        output_max = output_train.max(dim=1)[1]
        acc = (output_max==y_train_batch.view(-1)).sum().numpy()


        X_test_batch, y_test_batch = shuffle(X_test, y_test, n_samples=batch_size)
        X_test_batch = torch.tensor(X_test_batch, requires_grad=False).float()
        y_test_batch = torch.tensor(y_test_batch, requires_grad=False).long()
        output = learning_model(X_test_batch)
        test_loss = loss_fn(output, y_test_batch.view(-1))
        output_max = output.max(dim=1)[1]
        acc = (output_max==y_test_batch.view(-1)).sum().numpy()
        # print (f"TRAIN: Loss = {train_loss.data} -- Acc: {100 * acc / X_train_batch.size(0)}")
        # print (f"TEST: Loss = {test_loss.data} -- Acc: {100 * acc / X_test_batch.size(0)}")
        # print("------------------------------------------")
        return [float(train_loss.data)]

    def evaluate_debug(self, weights_x):
        global learning_model
        global X
        global y
        print("Evaluate the best")
        nn.utils.vector_to_parameters(torch.from_numpy(weights_x).float(), learning_model.parameters())

        X_test_batch, y_test_batch = shuffle(X_test, y_test, n_samples=batch_size*2)
        X_test_batch = torch.tensor(X_test_batch, requires_grad=False).float()
        y_test_batch = torch.tensor(y_test_batch, requires_grad=False).long()
        output = learning_model(X_test_batch)
        test_loss = loss_fn(output, y_test_batch.view(-1))
        output_max = output.max(dim=1)[1]
        acc = (output_max==y_test_batch.view(-1)).sum().numpy()
        print (f"Evaluation: Loss = {test_loss.data} -- Acc: {100 * acc / X_test_batch.size(0)}")
        # return mem_address_logs, left_right_header_log, memory_data_header_log, memory_tap_log

    def get_bounds(self):
        return ([self.min_limit]*self.dim, [self.max_limit]*self.dim)

problem_class = nn_evolve(dim=count_parameters, min_limit=-0.1, max_limit=0.1)
prob = pg.problem(problem_class)
# algo = pg.algorithm(pg.cmaes(gen = 10000))
algo = pg.algorithm(pg.sea(gen = 10000))
algo.set_verbosity(1)
pop = pg.population(prob, 300)
pop = algo.evolve(pop)

problem_class.evaluate_debug(pop.champion_x)
pickle.dump(pop.champion_x, open(f"sdmm_mnist_best_solution_v{version}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# algo = pg.algorithm(pg.sga(gen = 1, mutation='gaussian'))
# for i in range(50000):
#     print(f'Generation: {i}')
#     pop = algo.evolve(pop)
#     print(f'Best performance: {pop.champion_f}')
#     problem_class.evaluate_debug(pop.champion_x)
# pickle.dump(pop.champion_x, open(f"sdmm_mnist_best_solution_v{version}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
# mem_address_logs, left_right_header_log, memory_data_header_log, memory_tap_log = evaluate_debug(pop.champion_x)
# left_right_header_log = np.array(left_right_header_log)
# mem_address_logs = np.array(mem_address_logs)
# memory_data_header_log = np.array(memory_data_header_log)
# memory_tap_log = np.array(memory_tap_log)
# pickle.dump([mem_address_logs, left_right_header_log, memory_data_header_log, memory_tap_log], open("reasoning_mnist_best_solution_logs.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
