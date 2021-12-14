import collections
import copy
import logging
import random
from copy import deepcopy

import ConfigSpace
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np

from kernels import GraphKernels, WeisfilerLehman
from .gp import GraphGP

# === For NASBench-101 ====
MAX_EDGES = 9
VERTICES = 7
OPS = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']

# === For NASBench-201 ===
MAX_EDGES_201 = None
VERTICES_201 = None
OPS_201 = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']


def prune(original_matrix, ops):
    """
    Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(original_matrix)[0]
    new_matrix = copy.deepcopy(original_matrix)
    new_ops = copy.deepcopy(ops)
    # DFS forward from input
    visited_from_input = {0}
    frontier = [0]
    while frontier:
        top = frontier.pop()
        for v in range(top + 1, num_vertices):
            if original_matrix[top, v] and v not in visited_from_input:
                visited_from_input.add(v)
                frontier.append(v)

    # DFS backward from output
    visited_from_output = {num_vertices - 1}
    frontier = [num_vertices - 1]
    while frontier:
        top = frontier.pop()
        for v in range(0, top):
            if original_matrix[v, top] and v not in visited_from_output:
                visited_from_output.add(v)
                frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
        new_matrix = None
        new_ops = None
        valid_spec = False
        return

    new_matrix = np.delete(new_matrix, list(extraneous), axis=0)
    new_matrix = np.delete(new_matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
        del new_ops[index]

    return new_matrix, new_ops


def get_nas101_configuration_space():
    nas101_cs = ConfigSpace.ConfigurationSpace()

    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", OPS))
    for i in range(VERTICES * (VERTICES - 1) // 2):
        nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
    return nas101_cs


# === For NASBench-201 ====
def create_nasbench201_graph(op_node_labelling, edge_attr=False):
    assert len(op_node_labelling) == 6
    # the graph has 8 nodes (6 operation nodes + input + output)
    G = nx.DiGraph()
    if edge_attr:
        edge_list = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        G.add_edges_from(edge_list)
        edge_attribute = {}
        remove_edge_list = []
        for i, edge in enumerate(edge_list):
            edge_attribute[edge] = {'op_name': op_node_labelling[i]}
            if op_node_labelling[i] == 'none':
                remove_edge_list.append(edge)
        nx.set_edge_attributes(G, edge_attribute)
        G.remove_edges_from(remove_edge_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 3 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        # Assign dummy variables as node attributes:
        for i in G.nodes:
            G.nodes[i]['op_name'] = "1"
        G.graph_type = 'edge_attr'
    else:
        edge_list = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7)]
        G.add_edges_from(edge_list)

        # assign node attributes and collate the information for nodes to be removed
        # (i.e. nodes with 'skip_connect' or 'none' label)
        node_labelling = ['input'] + op_node_labelling + ['output']
        nodes_to_remove_list = []
        remove_nodes_list = []
        edges_to_add_list = []
        for i, n in enumerate(node_labelling):
            G.nodes[i]['op_name'] = n
            if n == 'none' or n == 'skip_connect':
                input_nodes = [edge[0] for edge in G.in_edges(i)]
                output_nodes = [edge[1] for edge in G.out_edges(i)]
                nodes_to_remove_info = {'id': i, 'input_nodes': input_nodes, 'output_nodes': output_nodes}
                nodes_to_remove_list.append(nodes_to_remove_info)
                remove_nodes_list.append(i)

                if n == 'skip_connect':
                    for n_i in input_nodes:
                        edges_to_add = [(n_i, n_o) for n_o in output_nodes]
                        edges_to_add_list += edges_to_add

        # reconnect edges for removed nodes with 'skip_connect'
        G.add_edges_from(edges_to_add_list)

        # remove nodes with 'skip_connect' or 'none' label
        G.remove_nodes_from(remove_nodes_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 7 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        G.graph_type = 'node_attr'

    # create the arch string for querying nasbench dataset
    arch_query_string = f'|{op_node_labelling[0]}~0|+' \
                        f'|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+' \
                        f'|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|'

    G.name = arch_query_string
    return G


def get_nas201_configuration_space():
    # for unpruned graph
    cs = ConfigSpace.ConfigurationSpace()
    ops_choices = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
    for i in range(6):
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, ops_choices))
    return cs


# Regularised evolution to generate new graphs
def mutate_arch(parent_arch, benchmark, return_unpruned_arch=True):
    if benchmark == 'nasbench101':
        # get parent_arch node label and adjacency matrix
        child_arch = deepcopy(parent_arch)
        node_labeling = list(nx.get_node_attributes(child_arch, 'op_name').values())
        adjacency_matrix = np.array(nx.adjacency_matrix(child_arch).todense())

        parent_node_labeling = deepcopy(node_labeling)
        parent_adjacency_matrix = deepcopy(adjacency_matrix)

        dim_op_labeling = (len(node_labeling) - 2)
        dim_adjacency_matrix = adjacency_matrix.shape[0] * (adjacency_matrix.shape[0] - 1) // 2

        mutation_failed = True

        while mutation_failed:
            # pick random parameter
            dim = np.random.randint(dim_op_labeling + dim_adjacency_matrix)

            if dim < dim_op_labeling:
                choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                node_number = int(dim + 1)
                parent_choice = node_labeling[node_number]

                # drop current values from potential choices
                choices.remove(parent_choice)

                # flip parameter
                choice_idx = np.random.randint(len(choices))
                node_labeling[node_number] = choices[choice_idx]

            else:
                choices = [0, 1]
                # find the corresponding row and colum in adjacency matrix
                idx = np.triu_indices(adjacency_matrix.shape[0], k=1)
                edge_i = int(dim - dim_op_labeling)
                row = idx[0][edge_i]
                col = idx[1][edge_i]
                parent_choice = adjacency_matrix[row, col]

                # drop current values from potential choices
                choices.remove(parent_choice)

                # flip parameter
                choice_idx = np.random.randint(len(choices))
                adjacency_matrix[row, col] = choices[choice_idx]

            try:
                pruned_adjacency_matrix, pruned_node_labeling = prune(adjacency_matrix, node_labeling)
                mutation_failed = False
            except:
                continue

        child_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)

        for i, n in enumerate(pruned_node_labeling):
            child_arch.nodes[i]['op_name'] = n
        if return_unpruned_arch:
            child_arch_unpruned = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
            for i, n in enumerate(node_labeling):
                child_arch_unpruned.nodes[i]['op_name'] = n

    elif benchmark == 'nasbench201':
        arch_str_list = parent_arch.name.split('|')
        op_label_rebuild = []
        for str_i in arch_str_list:
            if '~' in str_i:
                op_label_rebuild.append(str_i[:-2])

        # pick random parameter
        dim = np.random.randint(len(op_label_rebuild))
        parent_choice = op_label_rebuild[dim]

        # drop current values from potential choices
        ops_choices = OPS_201[:]
        ops_choices.remove(parent_choice)

        # flip parameter
        choice_idx = np.random.randint(len(ops_choices))
        op_label_rebuild[dim] = ops_choices[choice_idx]
        child_arch = create_nasbench201_graph(op_label_rebuild, edge_attr=parent_arch.graph_type == 'edge_attr')

        if return_unpruned_arch:
            child_arch_unpruned = child_arch

    if return_unpruned_arch:
        return child_arch, child_arch_unpruned

    return child_arch, None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def regularized_evolution(acquisition_func,
                          observed_archs,
                          observed_archs_unpruned=None,
                          benchmark='nasbench101',
                          pool_size=200, cycles=40, n_mutation=10, batch_size=1,
                          mutate_unpruned_arch=True):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    """
    # Generate some random archs into the evaluation pool
    if mutate_unpruned_arch and observed_archs_unpruned is None:
        raise ValueError("When mutate_unpruned_arch option is toggled on, you need to supplied the list of unpruned "
                         "observed architectures.")
    if observed_archs_unpruned is not None:
        assert len(observed_archs_unpruned) == len(observed_archs), " unequal length between the pruned/unpruned " \
                                                                    "architecture lists"

    n_random_archs = pool_size - len(observed_archs)
    if mutate_unpruned_arch:
        (random_archs, _, random_archs_unpruned) = random_sampling(pool_size=n_random_archs, benchmark=benchmark,
                                                                   return_unpruned_archs=True)
        population_unpruned = observed_archs_unpruned + random_archs_unpruned
    else:
        (random_archs, _, _) = random_sampling(pool_size=n_random_archs, benchmark=benchmark, )
        population_unpruned = None
    population = observed_archs + random_archs

    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    population_performance = []
    for i, archs in enumerate(population):
        arch_acq = acquisition_func.eval(archs, asscalar=True)
        population_performance.append(arch_acq)

    # Carry out evolution in cycles. Each cycle produces a bat model and removes another.
    k_cycle = 0

    while k_cycle < cycles:
        # Sample randomly chosen models from the current population based on the acquisition function values
        pseudo_prob = np.array(population_performance) / (np.sum(population_performance))
        if mutate_unpruned_arch:
            samples = random.choices(population_unpruned, weights=pseudo_prob, k=30)
            sample_indices = [population_unpruned.index(s) for s in samples]
        else:
            samples = random.choices(population, weights=pseudo_prob, k=30)
            sample_indices = [population.index(s) for s in samples]
        sample_performance = [population_performance[idx] for idx in sample_indices]

        # The parents is the best n_mutation model in the sample. skip 2-node archs
        top_n_mutation_archs_indices = np.argsort(sample_performance)[-n_mutation:]  # argsort>ascending
        parents_archs = [samples[idx] for idx in top_n_mutation_archs_indices if len(samples[idx].nodes) > 3]

        # Create the child model and store it.
        for parent in parents_archs:
            child, child_unpruned = mutate_arch(parent, benchmark)
            # skip invalid architectures whose number of edges exceed the max limit of 9
            if np.sum(nx.to_numpy_array(child)) > MAX_EDGES:
                continue
            if iso.is_isomorphic(child, parent):
                continue

            skip = False
            for prev_edit in population:
                if iso.is_isomorphic(child, prev_edit, ):
                    skip = True
                    break
            if skip: continue
            child_arch_acq = acquisition_func.eval(child, asscalar=True)
            population.append(child)
            if mutate_unpruned_arch:
                population_unpruned.append(child_unpruned)
            population_performance.append(child_arch_acq)

        # Remove the worst performing model and move to next evolution cycle
        worst_n_mutation_archs_indices = np.argsort(population_performance)[:n_mutation]
        for bad_idx in sorted(worst_n_mutation_archs_indices, reverse=True):
            population.pop(bad_idx)
            population_performance.pop(bad_idx)
            if mutate_unpruned_arch:
                population_unpruned.pop(bad_idx)
            # print(f'len pop = {len(population)}')
        k_cycle += 1

    # choose batch_size archs with highest acquisition function values to be evaluated next
    best_archs_indices = np.argsort(population_performance)[-batch_size:]
    recommended_pool = [population[best_idx] for best_idx in best_archs_indices]
    if mutate_unpruned_arch:
        recommended_pool_unpruned = [population_unpruned[best_idx] for best_idx in best_archs_indices]
        return (recommended_pool, recommended_pool_unpruned), (population, population_unpruned, population_performance)
    return (recommended_pool, None), (population, None, population_performance)


class Model(object):
    """A class representing a model.
    """

    def __init__(self):
        self.arch = None
        self.unpruned_arch = None
        self.error = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def mutation(observed_archs, observed_errors,
             n_best=10,
             n_mutate=None,
             pool_size=250,
             allow_isomorphism=False,
             patience=50,
             benchmark='nasbench101', observed_archs_unpruned=None):
    """
    BANANAS-style mutation.
    The main difference with the previously implemented evolutionary algorithms is that it allows unlimited number of
    edits based on the parent architecture. For previous implementations, we only allow 1-edit distance mutation.
    (although *in expectation*, there is only one edit.)
    """
    if n_mutate is None:
        n_mutate = int(0.5 * pool_size)
    assert pool_size >= n_mutate, " pool_size must be larger or equal to n_mutate"

    def _banana_mutate(arch, mutation_rate=1.0, benchmark='nasbench101'):
        """Bananas Style mutation of the cell"""
        if benchmark == 'nasbench201':
            parent_arch_list = arch.name.split("|")
            op_label_rebuild = [str_i[:-2] for str_i in parent_arch_list if "~" in str_i]
            mutation_prob = mutation_rate / len(OPS_201)
            child = None
            while True:
                try:
                    for idx, parent_choice in enumerate(op_label_rebuild):
                        ops_choices = OPS_201[:]
                        ops_choices.remove(parent_choice)
                        # Flip parameter with probability of mutation prob
                        if random.random() < mutation_prob:
                            choice_idx = np.random.randint(len(ops_choices))
                            op_label_rebuild[idx] = ops_choices[choice_idx]
                    child = create_nasbench201_graph(op_label_rebuild, edge_attr=arch.graph_type == 'edge_attr')
                    break
                except:
                    continue
            child_unpruned = child
            return child, child_unpruned
        elif benchmark == 'nasbench101':
            while True:
                new_ops = list(nx.get_node_attributes(arch, 'op_name').values())
                new_matrix = np.array(nx.adjacency_matrix(arch).todense())
                vertice = min(VERTICES, new_matrix.shape[0])

                if vertice > 2:
                    edge_mutation_prob = mutation_rate / vertice
                    for src in range(0, vertice - 1):
                        for dst in range(src + 1, vertice):
                            if random.random() < edge_mutation_prob:
                                new_matrix[src, dst] = 1 - new_matrix[src, dst]

                    op_mutation_prob = mutation_rate / (vertice - 2)
                    for ind in range(1, vertice - 1):
                        if random.random() < op_mutation_prob:
                            available = [o for o in OPS if o != new_ops[ind]]
                            new_ops[ind] = random.choice(available)
                try:
                    pruned_adjacency_matrix, pruned_node_labeling = prune(new_matrix, new_ops)
                    child_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
                    child_arch_unpruned = nx.from_numpy_array(new_matrix, create_using=nx.DiGraph)
                    for i, n in enumerate(pruned_node_labeling):
                        child_arch.nodes[i]['op_name'] = n
                    for i, n in enumerate(new_ops):
                        child_arch_unpruned.nodes[i]['op_name'] = n
                    return child_arch, child_arch_unpruned
                except:
                    continue
        else:
            raise NotImplementedError("Search space " + str(benchmark) + " not implemented!")

    population = collections.deque()
    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    for i, archs in enumerate(observed_archs):
        model = Model()
        model.arch = archs
        model.unpruned_arch = observed_archs_unpruned[i] if observed_archs_unpruned is not None else None
        model.error = observed_errors[i]
        population.append(model)

    best_archs = [arch for arch in sorted(population, key=lambda i: -i.error)][:n_best]
    evaluation_pool, evaluation_pool_unpruned = [], []
    per_arch = n_mutate // n_best
    for arch in best_archs:
        n_child = 0
        patience_ = patience
        while n_child < per_arch and patience_ > 0:
            child = Model()
            child.arch, child.unpruned_arch = _banana_mutate(arch.unpruned_arch if observed_archs_unpruned is not None
                                                             else arch.arch, benchmark=benchmark, )
            # skip 2-node archs
            if benchmark == 'nasbench101':
                if len(child.arch.nodes) <= 3:  # Skip input-output 3-graphs in nasbench101 search space
                    patience_ -= 1
                    continue
                if np.sum(np.array(nx.adjacency_matrix(child.arch).todense())) > MAX_EDGES:
                    patience_ -= 1
                    continue
            elif benchmark == 'nasbench201':
                if len(child.arch.nodes) == 0:  # Skip empty graphs in nasbench201 search space
                    patience_ -= 1
                    continue

            skip = False
            if not allow_isomorphism:
                # if disallow isomorphism, we enforce that each time, we mutate n distinct graphs. For now we do not
                # check the isomorphism in all of the previous graphs though
                if benchmark == 'nasbench201':
                    name = child.arch.name
                    if name == arch.arch.name:
                        patience_ -= 1
                        continue
                    prev_pool_names = (prev_edit.name for prev_edit in evaluation_pool)
                    if name in prev_pool_names:
                        patience_ -= 1
                        continue
                elif benchmark == 'nasbench101':
                    if iso.is_isomorphic(child.arch, arch.arch):
                        #  and desirable to simply do Weisfiler-Lehman Isomorphism test, since we are based on
                        #  Weisfeiler-Lehman graph kernel already and the isomorphism test is essentially free.
                        patience_ -= 1
                        continue

                    for prev_edit in evaluation_pool:
                        if iso.is_isomorphic(child.arch, prev_edit, ):
                            skip = True
                            break
                    if skip:
                        patience_ -= 1
                        continue

            child.error = 0
            evaluation_pool.append(child.arch)
            evaluation_pool_unpruned.append(child.unpruned_arch)
            n_child += 1

    # Add some random archs into the evaluation pool, if either 1) patience is reached or 2)
    nrandom_archs = max(pool_size - len(evaluation_pool), 0)
    if nrandom_archs:
        random_evaluation_pool, _, random_evaluation_pool_unpruned = random_sampling(pool_size=nrandom_archs,
                                                                                     benchmark=benchmark,
                                                                                     return_unpruned_archs=True)
        evaluation_pool += random_evaluation_pool
        evaluation_pool_unpruned += random_evaluation_pool_unpruned
    if observed_archs_unpruned is None:
        return evaluation_pool, [None] * len(evaluation_pool)
    return evaluation_pool, evaluation_pool_unpruned


# Random to generate new graphs
def random_sampling(pool_size=100, benchmark='nasbench101',
                    save_config=False,
                    edge_attr=False,
                    return_unpruned_archs=False):
    """
    Return_unpruned_archs: bool: If True, both the list of pruned architectures and unpruned architectures will be
        returned.

    """
    evaluation_pool = []
    pruned_labeling_list = []
    unpruned_evaluation_pool = []  # The unpruned architectures. These might contain invalid architectures
    nasbench201_op_label_list = []
    nasbench_config_list = []
    # attribute_name_list = []
    while len(evaluation_pool) < pool_size:
        if benchmark == 'nasbench101':
            # generate random architecture for nasbench101
            if edge_attr:
                logging.warning("NAS-Bench-101 dataset search space is node-attributed. edge_attr option is not "
                                "applicable.")

            nas101_cs = get_nas101_configuration_space()
            config = nas101_cs.sample_configuration()

            adjacency_matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
            idx = np.triu_indices(adjacency_matrix.shape[0], k=1)
            for i in range(VERTICES * (VERTICES - 1) // 2):
                row = idx[0][i]
                col = idx[1][i]
                adjacency_matrix[row, col] = config["edge_%d" % i]

            labeling = [config["op_node_%d" % i] for i in range(5)]
            labeling = ['input'] + list(labeling) + ['output']

            try:
                pruned_adjacency_matrix, pruned_labeling = prune(adjacency_matrix, labeling)
            except:
                continue

            # skip only duplicating 2-node architecture
            if len(pruned_labeling) == 2 and pruned_labeling in pruned_labeling_list:
                continue

            # skip invalid architectures whose number of edges exceed the max limit of 9
            if np.sum(pruned_adjacency_matrix) > MAX_EDGES or np.sum(pruned_adjacency_matrix) == 0:
                continue

            rand_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
            rand_arch.graph_type = 'node_attr'

            for i, n in enumerate(pruned_labeling):
                rand_arch.nodes[i]['op_name'] = n
            if return_unpruned_archs:
                unpruned_rand_arch = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
                unpruned_rand_arch.graph_type = 'node_attr'
                for i, n in enumerate(labeling):
                    unpruned_rand_arch.nodes[i]['op_name'] = n

            pruned_labeling_list.append(pruned_labeling)
            nasbench_config_list.append(config)

        elif benchmark == 'nasbench201':
            # generate random architecture for nasbench201

            nas201_cs = get_nas201_configuration_space()
            config = nas201_cs.sample_configuration()
            op_labeling = [config["edge_%d" % i] for i in range(len(config.keys()))]
            # skip only duplicating architecture
            if op_labeling in nasbench201_op_label_list:
                continue

            nasbench201_op_label_list.append(op_labeling)
            rand_arch = create_nasbench201_graph(op_labeling, edge_attr)
            nasbench_config_list.append(config)

            # IN Nasbench201, it is possible that invalid graphs consisting entirely from None and skip-line are
            # generated; remove these invalid architectures.

            # Also remove if the number of edges is zero. This is is possible, one example in NAS-Bench-201:
            # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|none~0|avg_pool_3x3~1|none~2|'
            if len(rand_arch) == 0 or rand_arch.number_of_edges() == 0:
                continue

            unpruned_rand_arch = rand_arch
        evaluation_pool.append(rand_arch)
        if return_unpruned_archs:
            unpruned_evaluation_pool.append(unpruned_rand_arch)

    if save_config:
        if return_unpruned_archs:
            return evaluation_pool, nasbench_config_list, unpruned_evaluation_pool
        else:
            return evaluation_pool, nasbench_config_list, None
    else:
        if return_unpruned_archs:
            return evaluation_pool, None, unpruned_evaluation_pool
        else:
            return evaluation_pool, None, None
