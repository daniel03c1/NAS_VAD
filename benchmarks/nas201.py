# For use of the NAS-Bench-201 dataset version NAS-Bench-201-v1_0-e61699.pth

import os
import numpy as np
import networkx as nx
from benchmarks.objectives import ObjectiveFunction
from nas_201_api import NASBench201API as API
import ConfigSpace
import random


class NAS201(ObjectiveFunction):

    def __init__(self, data_dir, task='cifar10-valid', log_scale=True, negative=True,
                 use_12_epochs_result=False,
                 seed=None):
        """
        data_dir: data directory that contains NAS-Bench-201-v1_0-e61699.pth file
        task: the target image tasks. Options: cifar10-valid, cifar100, ImageNet16-120
        log_scale: whether output the objective in log scale
        negative: whether output the objective in negative form
        use_12_epochs_result: whether use the statistics at the end of training of the 12th epoch instead of all the
                              way till the end.
        seed: set the random seed to access trained model performance: Options: 0, 1, 2
              seed=None will select the seed randomly
        """

        self.api = API(os.path.join(data_dir, 'NAS-Bench-201-v1_1-096897.pth'))
        if isinstance(task, list):
            task = task[0]
        self.task = task
        self.use_12_epochs_result = use_12_epochs_result

        if task == 'cifar10-valid':
            best_val_arch_index = 6111
            best_val_acc = 91.60666665039064 / 100
            best_test_arch_index = 1459
            best_test_acc = 91.52333333333333 / 100
        elif task == 'cifar100':
            best_val_arch_index = 9930
            best_val_acc = 73.49333323567708 / 100
            best_test_arch_index = 9930
            best_test_acc = 73.51333326009114 / 100
        elif task == 'ImageNet16-120':
            best_val_arch_index = 10676
            best_val_acc = 46.766666727701825 / 100
            best_test_arch_index = 857
            best_test_acc = 47.311111097547744 / 100
        else:
            raise NotImplementedError("task" + str(task) + " is not implemented in the dataset.")

        if log_scale:
            best_val_acc = np.log(best_val_acc)

        best_val_err = 1. - best_val_acc
        best_test_err = 1. - best_test_acc
        if log_scale:
            best_val_err = np.log(best_val_err)
            best_test_err = np.log(best_val_err)
        if negative:
            best_val_err = -best_val_err
            best_test_err = -best_test_err

        self.best_val_err = best_val_err
        self.best_test_err = best_test_err
        self.best_val_acc = best_val_acc
        self.best_test_acc = best_test_acc

        super(NAS201, self).__init__(dim=None, optimum_location=best_test_arch_index, optimal_val=best_test_err,
                                     bounds=None)

        self.log_scale = log_scale
        self.seed = seed
        self.X = []
        self.y_valid_acc = []
        self.y_test_acc = []
        self.costs = []
        self.negative = negative
        # self.optimal_val =   # lowest mean validation error
        # self.y_star_test =   # lowest mean test error

    def _retrieve(self, G, budget, which='eval'):
        #  set random seed for evaluation
        if which == 'test':
            seed = 3
        else:
            seed_list = [777, 888, 999]
            if self.seed is None:
                seed = random.choice(seed_list)
            elif self.seed >= 3:
                seed = self.seed
            else:
                seed = seed_list[self.seed]

        # find architecture index
        arch_str = G.name
        # print(arch_str)

        try:
            arch_index = self.api.query_index_by_arch(arch_str)
            acc_results = self.api.query_by_index(arch_index, self.task, use_12epochs_result=self.use_12_epochs_result,)
            if seed is not None and 3 <= seed < 777:
                # some architectures only contain 1 seed result
                acc_results = self.api.get_more_info(arch_index, self.task, None,
                                                     use_12epochs_result=self.use_12_epochs_result,
                                                     is_random=False)
                val_acc = acc_results['valid-accuracy'] / 100
                test_acc = acc_results['test-accuracy'] / 100
            else:
                try:
                    acc_results = self.api.get_more_info(arch_index, self.task, None,
                                                         use_12epochs_result=self.use_12_epochs_result,
                                                         is_random=seed)
                    val_acc = acc_results['valid-accuracy'] / 100
                    test_acc = acc_results['test-accuracy'] / 100
                    # val_acc = acc_results[seed].get_eval('x-valid')['accuracy'] / 100
                    # if self.task == 'cifar10-valid':
                    #     test_acc = acc_results[seed].get_eval('ori-test')['accuracy'] / 100
                    # else:
                    #     test_acc = acc_results[seed].get_eval('x-test')['accuracy'] / 100
                except:
                    # some architectures only contain 1 seed result
                    acc_results = self.api.get_more_info(arch_index, self.task, None,
                                                         use_12epochs_result=self.use_12_epochs_result,
                                                         is_random=False)
                    val_acc = acc_results['valid-accuracy'] / 100
                    test_acc = acc_results['test-accuracy'] / 100

            auxiliary_info = self.api.query_meta_info_by_index(arch_index,
                                                               use_12epochs_result=self.use_12_epochs_result)
            cost_info = auxiliary_info.get_compute_costs(self.task)

            # auxiliary cost results such as number of flops and number of parameters
            cost_results = {'flops': cost_info['flops'], 'params': cost_info['params'],
                            'latency': cost_info['latency']}

        except FileNotFoundError:
            val_acc = 0.01
            test_acc = 0.01
            print('missing arch info')
            cost_results = {'flops': None, 'params': None,
                            'latency': None}

        # store val and test performance + auxiliary cost information
        self.X.append(arch_str)
        self.y_valid_acc.append(val_acc)
        self.y_test_acc.append(test_acc)
        self.costs.append(cost_results)

        if which == 'eval':
            err = 1 - val_acc
        elif which == 'test':
            err = 1 - test_acc
        else:
            raise ValueError("Unknown query parameter: which = " + str(which))

        if self.log_scale:
            y = np.log(err)
        else:
            y = err
        if self.negative:
            y = -y
        return y

    def eval(self, G, budget=199, n_repeat=1):
        # input is a list of graphs [G1,G2, ....]
        if n_repeat == 1:
            return self._retrieve(G, budget, 'eval'),  [np.nan]
        return np.mean(np.array([self._retrieve(G, budget, 'eval') for _ in range(n_repeat)])), [np.nan]

    def test(self, G, budget=199, n_repeat=1):
        return np.mean(np.array([self._retrieve(G, budget, 'test') for _ in range(n_repeat)]))

    def get_results(self, ignore_invalid_configs=False):

        regret_validation = []
        regret_test = []
        costs = []
        model_graph_specs = []

        inc_valid = 0
        inc_test = 0

        for i in range(len(self.X)):

            if inc_valid < self.y_valid_acc[i]:
                inc_valid = self.y_valid_acc[i]
                inc_test = self.y_test_acc[i]

            regret_validation.append(float(self.best_val_acc - inc_valid))
            regret_test.append(float(self.best_test_acc - inc_test))
            model_graph_specs.append(self.X[i])
            costs.append(self.costs[i])

        res = dict()
        res['regret_validation'] = regret_validation
        res['regret_test'] = regret_test
        res['costs'] = costs
        res['model_graph_specs'] = model_graph_specs

        return res

    @staticmethod
    def get_configuration_space():
        # for unpruned graph
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
        for i in range(6):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, ops_choices))
        return cs


class NAS201edge(NAS201):

    def _retrieve(self, G, budget, which='eval'):
        #  set random seed for evaluation
        seed_list = [777, 888, 999]
        if self.seed is None:
            seed = random.choice(seed_list)
        elif self.seed >= 3:
            seed = self.seed
        else:
            seed = seed_list[self.seed]

        # find architecture index
        arch_str = G.name
        # print(arch_str)

        try:
            arch_index = self.api.query_index_by_arch(arch_str)
            acc_results = self.api.query_by_index(arch_index, self.task)
            if seed >= 3:
                # some architectures only contain 1 seed result
                acc_results = self.api.get_more_info(arch_index, self.task, None, self.use_12_epochs_result, False)
                val_acc = acc_results['valid-accuracy'] / 100
                test_acc = acc_results['test-accuracy'] / 100
            else:
                try:
                    val_acc = acc_results[seed].get_eval('x-valid')['accuracy'] / 100
                    if self.task == 'cifar10-valid':
                        test_acc = acc_results[seed].get_eval('ori-test')['accuracy'] / 100
                    else:
                        test_acc = acc_results[seed].get_eval('x-test')['accuracy'] / 100
                except:
                    # some architectures only contain 1 seed result
                    acc_results = self.api.get_more_info(arch_index, self.task, None, self.use_12_epochs_result, False)
                    val_acc = acc_results['valid-accuracy'] / 100
                    test_acc = acc_results['test-accuracy'] / 100

            auxiliary_info = self.api.query_meta_info_by_index(arch_index)
            cost_info = auxiliary_info.get_compute_costs(self.task)

            # auxiliary cost results such as number of flops and number of parameters
            cost_results = {'flops': cost_info['flops'], 'params': cost_info['params'],
                            'latency': cost_info['latency']}

        except FileNotFoundError:
            val_acc = 0.01
            test_acc = 0.01
            print('missing arch info')
            cost_results = {'flops': None, 'params': None,
                            'latency': None}

        # store val and test performance + auxiliary cost information
        self.X.append(arch_str)
        self.y_valid_acc.append(val_acc)
        self.y_test_acc.append(test_acc)
        self.costs.append(cost_results)

        if which == 'eval':
            err = 1. - val_acc
        elif which == 'test':
            err = 1. - test_acc
        else:
            raise ValueError("Unknown query parameter: which = " + str(which))

        if self.log_scale:
            y = np.log(err)
        else:
            y = err
        if self.negative:
            y = -y
        return y


class NAS201MultiTask(ObjectiveFunction):

    def __init__(self, data_dir, tasks,
                 task_weights=None,
                 log_scale=True, negative=True, use_12_epochs_result=False,
                 seed=None):
        self.tasks = tasks
        # Normalize the task_weights so that they sum to 1
        if task_weights is None:
            self.task_weights = np.array([1. / len(tasks)] * len(tasks))
        else:
            self.task_weights = np.array(task_weights).flatten() / np.sum(np.array(task_weights))
        assert len(self.tasks) == self.task_weights.shape[0], " mismatch between the task_weights and tasks!"
        self.nas201 = [NAS201(data_dir, t, log_scale, negative, use_12_epochs_result, seed) for t in tasks]
        super(NAS201MultiTask, self).__init__(dim=None, optimum_location=None, optimal_val=None, bounds=None)

    def eval(self, G, budget=100, n_repeat=1, scalarize=True):
        evals = np.array([n.eval(G, budget, n_repeat) for n in self.nas201])
        return np.sum(evals * self.task_weights) if scalarize else evals

    def test(self, G, budget=100, n_repeat=1, scalarize=True):
        tests = np.array([n.test(G, budget, n_repeat) for n in self.nas201])
        return np.sum(tests * self.task_weights) if scalarize else tests

    def __getattr__(self, item):
        method = getattr(self.nas201, item)
        return [method() for n in self.nas201]


if __name__ == '__main__':
    import pickle
    from bayesopt.generate_test_graphs import create_nasbench201_graph

    output_path = '../data/'
    op_node_labelling = ['nor_conv_3x3', 'none', 'avg_pool_3x3', 'skip_connect', 'nor_conv_3x3', 'skip_connect']
    G = create_nasbench201_graph(op_node_labelling)
    nascifar10 = NAS201(data_dir=output_path, task='cifar10-valid', seed=0, negative=False, log_scale=False)
    f = nascifar10.eval
    result = f(G)
