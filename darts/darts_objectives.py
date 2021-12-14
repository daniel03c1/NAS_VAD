# This file provides the conversion between of our API the DARTS interface
import logging

import torch
from torch.multiprocessing import Pool

from benchmarks.objectives import ObjectiveFunction
from darts.arch_trainer import DARTSTrainer
from darts.utils import *

INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'


def get_ops(search_space):
    if search_space == 'darts':
        return ['max_pool_3x3',
                'avg_pool_3x3',
                'skip_connect_original',
                'sep_conv_3x3_original',
                'sep_conv_5x5_original',
                'dil_conv_3x3',
                'dil_conv_5x5',
                ], 4
    elif search_space == 'nasnet':
        return ['max_pool_3x3',
                'max_pool_5x5',
                'sep_conv_3x3',
                'sep_conv_5x5',
                'sep_conv_7x7',
                'skip_connect',
               ], 7
    elif search_space == 'vadA':
        return [# 'sep_conv_3x3',
                # 'sep_conv_5x5',
                'MBConv_3x3_x2',
                'MBConv_3x3_x4',
                # 'MBConv_5x5_x2',
                # 'MBConv_5x5_x4',
                'SE_0.25',
                'SE_0.5',
                # 'FFN2D_0.5', # new~
                # 'FFN2D_1', # new~
                # 'FFN2D_2', # new~
                # 'GLU2D_3', # new~
                # 'GLU2D_5', # new
                # 'MHA2D_2_TO', # new~
                # 'MHA2D_2_FO', # new~
                'MHA2D_4_TO', # new~
                'MHA2D_4_FO', # new~
                'skip_connect',
                'zero'], 2 # 4 # 3 # 3 # 4
    elif search_space == 'vadB':
        return ['MHA2D_2',
                'MHA2D_4',
                'FFN2D_0.5',
                'FFN2D_1',
                'FFN2D_2',
                'GLU2D_3',
                'GLU2D_5',
                # 'sep_conv_3x3', # new~
                # 'sep_conv_5x5', # new~
                'MBConv_3x3_x2', # new~
                'MBConv_3x3_x4', # new~
                # 'MBConv_5x5_x2', # new~
                # 'MBConv_5x5_x4', # new~
                'SE_0.25', # new~
                'SE_0.5', # new~
                'skip_connect',
                'zero'], 7 # 4 # 6 # 4


class DARTSObjective(ObjectiveFunction):
    def __init__(self,
                 data_path: str,
                 save_path: str,
                 dataset: str = 'cifar10',
                 cutout=False,
                 log_scale=True,
                 negative=True,
                 query_policy='best',
                 seed=None,
                 n_gpu='all', epochs=50,
                 dummy_eval=True,
                 search_space='darts',
                 auxiliary=True,
                 use_1d=False,
                 time_average=False):
        super(DARTSObjective, self).__init__()
        assert query_policy in ['best', 'last', 'last5']
        self.query_policy = query_policy
        self.cutout = cutout
        self.log_scale = log_scale
        self.negative = negative
        self.seed = seed
        self.dummy_eval = dummy_eval
        self.dataset = dataset
        self.data_path = data_path
        self.save_path = save_path
        self.n_gpu = n_gpu if isinstance(n_gpu, int) else torch.cuda.device_count()
        self.epochs = epochs
        self.search_space = search_space
        self.auxiliary = auxiliary
        self.use_1d = use_1d
        self.time_average = time_average

    def eval(self, X, *args):
        if self.dummy_eval:
            return self.dummy_eval_(X, *args)
        return self.eval_(X, *args)

    def eval_(self, X, *args):
        """
        Evaluate a number of DARTS architecture in parallel. X should be a list of Genotypes defined by DARTS API.
        """
        from math import ceil
        n_parallel = min(len(X), self.n_gpu)
        res = []
        diag_stats = []

        if n_parallel == 0:
            raise ValueError("No GPUs available!")
        elif n_parallel == 1:
            for i, genotype in enumerate(X):
                t = DARTSTrainer(self.data_path, self.save_path, genotype,
                                 self.dataset, cutout=self.cutout,
                                 epochs=self.epochs, 
                                 eval_policy=self.query_policy,
                                 use_1d=self.use_1d,
                                 time_average=self.time_average)
                print('Start training: ', i + 1, "/ ", len(X))
                t.train()  # bottleneck
                result = t.retrieve()
                res.append(1. - result[0])  # Turn into error
                diag_stats.append(result[1])
        else:
            gpu_ids = range(n_parallel)
            num_reps = ceil(len(X) / float(n_parallel))

            for i in range(num_reps):
                x = X[i * n_parallel: min((i + 1) * n_parallel,
                                          len(X))]  # select the number of parallel archs to evaluate
                selected_gpus = gpu_ids[:len(x)]
                other_arg = [self.data_path, self.save_path, self.dataset, self.cutout, self.epochs, self.query_policy]
                args = list(map(list, zip(x, selected_gpus)))
                args = [a + other_arg for a in args]
                pool = Pool(processes=len(x))
                current_res = pool.starmap(parallel_eval, args)
                pool.close()
                pool.join()
                res.extend([i for i in current_res if i >= 0])  # Filter out the negative results due to errors

        res = np.array(res).flatten()
        if self.log_scale:
            res = np.log(res)
        if self.negative:
            res = -res
        return res, diag_stats

    def dummy_eval_(self, X: list, *args):
        # Evaluate a dummy variable for fast debugging
        # The dummy variable is a linear function of the number of 'dil_conv_5x5' units -- the objective
        # is to maximise the number of such operations.
        res = []
        for i, x in enumerate(X):
            count = 1.
            for i in x.nodes:
                if x.nodes[i]['op_name'] == 'sep_conv_3x3':
                    count -= 0.1
                # elif x.nodes[i]['op_name'] == 'sep_conv_3x3':
                #     count -= 0.1
                # elif x.nodes[i]['op_name'] == 'dil_conv_5x5':
                #     count -= 0.2
            res.append(max(count, 0.01))
        res = np.array(res).flatten()
        if self.log_scale:
            res = np.log(res)
        if self.negative:
            res = -res
        return res, None


def parallel_eval(*args):
    """The function to be used for parallelism"""
    genotype, gpuid, data_path, save_path, dataset, cutout, epochs, policy = args
    print(args)
    t = DARTSTrainer(data_path, save_path, genotype, dataset,
                     cutout=cutout, epochs=epochs, gpu_id=gpuid, eval_policy=policy)
    try:
        t.train()
    except:
        logging.error("Error occurred in training on of the archs. The child process is terminated")
        return -1
    return 1. - t.retrieve() / 100.


def random_sample_darts(n, search_space, same_arch=True, second_search_space='vad'):
    # TODO: same_arch, and different OPS for each type of cell
    """
    n: number of random samples to yield
    same_arch (bool): whether to use the same architecture for the normal cell and the reduction cell
    """
    # if not same_arch:
    #     raise NotImplementedError

    """Generate a list of 2 tuples, consisting of the random DARTS Genotype and DiGraph"""
    if second_search_space is None:
        second_search_space = search_space
    OPS, n_tower = get_ops(search_space)
    OPS_2, n_tower_2 = get_ops(second_search_space)

    def _sample():
        normal = []
        reduction = []
        if same_arch:
            for i in range(n_tower):
                # input nodes for reduce
                ops = np.random.choice(range(len(OPS)), 2) # n_tower)
                nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
                nodes_in_reduce = nodes_in_normal

                normal.extend([(OPS[ops[0]], nodes_in_normal[0]),
                               (OPS[ops[1]], nodes_in_normal[1])])
                reduction.extend([(OPS[ops[0]], nodes_in_reduce[0],),
                                  (OPS[ops[1]], nodes_in_reduce[1])])
        else:
            for i in range(n_tower):
                ops = np.random.choice(range(len(OPS)), 2)
                nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)

                normal.extend([(OPS[ops[0]], nodes_in_normal[0]),
                               (OPS[ops[1]], nodes_in_normal[1])])

            for i in range(n_tower_2):
                ops = np.random.choice(range(len(OPS_2)), 2)
                nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

                reduction.extend([(OPS_2[ops[0]], nodes_in_reduce[0]),
                                  (OPS_2[ops[1]], nodes_in_reduce[1])])

        darts_genotype = Genotype(normal=normal,
                                  normal_concat=range(2, 2 + n_tower),
                                  reduce=reduction,
                                  reduce_concat=range(2, 2 + n_tower_2))
        darts_digraph = darts2graph(darts_genotype)

        return darts_digraph, darts_genotype

    res = []
    for i in range(n):
        r = _sample()
        if is_valid_darts(r[1]):
            res.append(r)
    return res


def _mutate(arch: Genotype, search_space, edits, same_arch=True,
            second_search_space='vad') -> (nx.DiGraph, Genotype):
    """BANANAS style mutation on a DARTS-style Genotype"""
    if second_search_space is None:
        second_search_space = search_space
    OPS, n_towers = get_ops(search_space)
    OPS_2, n_tower_2 = get_ops(second_search_space)

    def _mutate_cell(cell, edits, ops):
        # convert tuple to list to allow edits
        mutable_cell = [list(i) for i in cell]

        for _ in range(edits):
            num = np.random.choice(2)
            if num == 1: # Mutate the ops
                op_to_mutate = np.random.choice(len(cell))
                op_chosen = np.random.choice(ops)
                mutable_cell[op_to_mutate][0] = op_chosen
            else: # Mutate the wiring
                inputs = len(mutable_cell) // 2 + 2
                while True:
                    op_to_mutate = np.random.choice(len(cell))
                    choice = np.random.choice(op_to_mutate//2 + 2)

                    # two edges directing the same node
                    if op_to_mutate % 2 == 0 \
                            and mutable_cell[op_to_mutate + 1][1] == choice:
                        continue
                    elif op_to_mutate % 2 == 1 \
                            and mutable_cell[op_to_mutate - 1][1] == choice:
                        continue
                    elif mutable_cell[op_to_mutate][1] == choice:
                        continue
                    else:
                        mutable_cell[op_to_mutate][1] = choice
                        break

        mutated_cell = [tuple(i) for i in mutable_cell]
        return mutated_cell

    if same_arch:
        mutated = _mutate_cell(arch.normal, edits, OPS)
        mutated_genotype = Genotype(normal=mutated, normal_concat=arch.normal_concat,
                                    reduce=mutated, reduce_concat=arch.reduce_concat)
    else:
        # choice = np.random.choice(2)
        # if choice == 1:  # Mutate the main
        #     mutated_normal = _mutate_cell(arch.normal, edits)
        #     mutated_reduce = arch.reduce
        # else:
        #     mutated_normal = arch.normal
        #     mutated_reduce = _mutate_cell(arch.reduce, edits)
        mutated_normal = _mutate_cell(arch.normal, edits, OPS)
        mutated_reduce = _mutate_cell(arch.reduce, edits, OPS_2)

        mutated_genotype = Genotype(normal=mutated_normal, normal_concat=arch.normal_concat,
                                    reduce=mutated_reduce, reduce_concat=arch.reduce_concat)

    mutated_digraph = darts2graph(mutated_genotype)

    return mutated_digraph, mutated_genotype


def mutation_darts(n, search_space, parents, edits, same_arch=True, n_rand=None,
                   second_search_space='vad'):
    if second_search_space is None:
        second_search_space = search_space

    res = []
    if n_rand is None:
        n_rand = n

    while len(res) < n:
        # Randomly choose a parent
        parent_arch_idx = np.random.choice(len(parents))
        parent_arch = parents[parent_arch_idx]

        # Mutate the parent
        child_arch, child_genotype = _mutate(
            parent_arch, search_space, edits, same_arch,
            second_search_space=second_search_space)

        while not is_valid_darts(child_genotype):
            child_arch, child_genotype = _mutate(
                parent_arch, search_space, edits, same_arch,
                second_search_space=second_search_space)

        res.append((child_arch, child_genotype))

    if n_rand > 0:
        rand_archs = random_sample_darts(
            n_rand, search_space, same_arch=same_arch,
            second_search_space=second_search_space)
        res.extend(rand_archs)

    return res

