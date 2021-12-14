# Utils.py for the conversion of DARTS Genotype and DAGs used in our interface

from copy import deepcopy

import networkx as nx
import numpy as np

from darts.cnn.genotypes import *


def darts2graph(genotype: Genotype,
                return_reduction=True,
                remove_skip=True) -> (nx.DiGraph, nx.DiGraph):
    """
    Convert a DARTS-style genotype representation of an edge-attributed DAG to a canonical form DAG in our interface.
    Returns: a tuple of two canonical form DAGs (normal and reduction cells)
    """

    def _cell2graph(cell, concat):
        G = nx.DiGraph()
        n_nodes = (len(cell) // 2) * 3 + 3
        G.add_nodes_from(range(n_nodes), op_name=None)
        n_ops = len(cell) // 2
        G.nodes[0]['op_name'] = 'input1'
        G.nodes[1]['op_name'] = 'input2'
        G.nodes[n_nodes - 1]['op_name'] = 'output'
        for i in range(n_ops):
            G.nodes[i * 3 + 2]['op_name'] = cell[i * 2][0]
            G.nodes[i * 3 + 3]['op_name'] = cell[i * 2 + 1][0]
            G.nodes[i * 3 + 4]['op_name'] = 'add'
            G.add_edge(i * 3 + 2, i * 3 + 4)
            G.add_edge(i * 3 + 3, i * 3 + 4)

        for i in range(n_ops):
            # Add the connections to the input
            for offset in range(2):
                if cell[i * 2 + offset][1] == 0:
                    G.add_edge(0, i * 3 + 2 + offset)
                elif cell[i * 2 + offset][1] == 1:
                    G.add_edge(1, i * 3 + 2 + offset)
                else:
                    k = cell[i * 2 + offset][1] - 2
                    # Add a connection from the output of another block
                    G.add_edge(int(k) * 3 + 4, i * 3 + 2 + offset)
        # Add connections to the output
        for i in concat:
            if i <= 1:
                G.add_edge(i, n_nodes - 1)  # Directly from either input to the output
            else:
                op_number = i - 2
                G.add_edge(op_number * 3 + 4, n_nodes - 1)
        # If remove the skip link nodes, do another sweep of the graph
        if remove_skip:
            for j in range(n_nodes):
                try:
                    G.nodes[j]
                except KeyError:
                    continue
                if G.nodes[j]['op_name'] == 'skip_connect':
                    in_edges = list(G.in_edges(j))
                    out_edge = list(G.out_edges(j))[0][1]  # There should be only one out edge really...
                    for in_edge in in_edges:
                        G.add_edge(in_edge[0], out_edge)
                    G.remove_node(j)
                elif G.nodes[j]['op_name'] == 'none':
                    G.remove_node(j)
            for j in range(n_nodes):
                try:
                    G.nodes[j]
                except KeyError:
                    continue

                if G.nodes[j]['op_name'] not in ['input1', 'input2']:
                    # excepting the input nodes, if the node has no incoming edge, remove it
                    if len(list(G.in_edges(j))) == 0:
                        G.remove_node(j)
                elif G.nodes[j]['op_name'] != 'output':
                    # excepting the output nodes, if the node has no outgoing edge, remove it
                    if len(list(G.out_edges(j))) == 0:
                        G.remove_node(j)
                elif G.nodes[j]['op_name'] == 'add':  # If add has one incoming edge only, remove the node
                    in_edges = list(G.in_edges(j))
                    out_edges = list(G.out_edges(j))
                    if len(in_edges) == 1 and len(out_edges) == 1:
                        G.add_edge(in_edges[0][0], out_edges[0][1])
                        G.remove_node(j)

        return G

    G_normal = _cell2graph(genotype.normal, genotype.normal_concat)
    try:
        G_reduce = _cell2graph(genotype.reduce, genotype.reduce_concat)
    except AttributeError:
        G_reduce = None
    if return_reduction and G_reduce is not None:
        return G_normal, G_reduce
    else:
        return G_normal, None


def graph2darts(G_normal: nx.DiGraph, G_reduce: nx.DiGraph = None) -> Genotype:
    """
    Convert a canonical form DAG of our interface to a corresponding DARTS Genotype.
    If the reduction cell is not supplied, then the normal cell DAG will be taken as the reduction cell too.
    """

    def _graph2cell(G):
        from math import floor
        normal = []
        n_nodes = np.max(G.nodes)
        for i in range(2, n_nodes - 1):
            try:
                op_name = G.nodes[i]['op_name']
                if op_name == 'add':
                    continue
                in_edge = list(G.in_edges(i))[0][0]
            except KeyError:
                adder = 1 if i % 3 == 0 else 2
                adder_in_edges = [i[0] for i in list(G.in_edges(i + adder))]
                if len(adder_in_edges) < 2:  # There is missing link, this suggest 'none' op
                    op_name = 'none'
                    # For a none edge, how the link is connected doesnt matter, so we arbitrarily connect it to the
                    # first input
                    in_edge = 0
                else:  # Otherwise a skip connection
                    op_name = 'skip_connect'
                    in_edge = [i for i in adder_in_edges if
                               i in [0, 1] or i % 3 == 1]  # The skip link is detected if its not from one of its leaf!
                    in_edge = in_edge[0]

            if in_edge <= 1:
                normal.append((op_name, in_edge))
            else:
                normal.append((op_name, in_edge // 3 + 1))
        output_edges = [i[0] for i in list(G.in_edges(n_nodes))]
        normal_concat = [2 + floor((i - 4) / 3) for i in output_edges]
        return normal, normal_concat

    normal, normal_concat = _graph2cell(G_normal)
    if G_reduce is not None:
        reduce, reduce_concat = _graph2cell(G_reduce)
    else:
        reduce = deepcopy(normal)
        reduce_concat = deepcopy(normal_concat)
    return Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)


def is_valid_darts(genotype: Genotype):
    for r in [genotype.normal, genotype.reduce]:
        connections = [i[1] for i in r]
        if 0 not in connections or 1 not in connections:
            return False
    return True


if __name__ == '__main__':
    # Try converting back and forth between the Genotypes and the networkx graphs
    # Note that a failed assertion does not necessarily mean a bug, because when 'None' connection is involved, we
    # simply wire the in-edge to the first input, as there is no information flow anyway.

    assert DARTS_V1 == graph2darts(*darts2graph(DARTS_V1, True))
    assert DARTS_V2 == graph2darts(*darts2graph(DARTS_V2, True))
    assert AmoebaNet == graph2darts(*darts2graph(AmoebaNet, True))
    assert NASNet == graph2darts(*darts2graph(NASNet, True))
    print('OK')
    # for i in range(100):
    #     original, digraph = random_sample_darts()
    #     assert graph2darts(digraph.normal) == original.normal
