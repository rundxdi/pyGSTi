import stim
import networkx as nx
import pyomo.environ as pe
from itertools import product
import matplotlib.pyplot as plt

# non-relative imports for now
from pygsti.extras.idletomography import idtcore
import collections as _collections
import itertools as _itertools
import time as _time
import warnings as _warnings
from itertools import product, permutations
from pygsti.baseobjs import basisconstructors
import numpy as _np
from pygsti.baseobjs import Basis
import re

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="two_qubit_log.log", encoding="utf-8", level=logging.INFO, filemode="w")


# Commutator Helper Functions
def commute(mat1, mat2):
    return mat1 @ mat2 + mat2 @ mat1


def anti_commute(mat1, mat2):
    return mat1 @ mat2 - mat2 @ mat1


# Hamiltonian Error Generator
# returns output of applying error gen in choi unit form
# input is state, output is state
def hamiltonian_error_generator(initial_state, indexed_pauli, identity):
    return 1 * (
        -1j * indexed_pauli @ initial_state @ identity
        + 1j * identity @ initial_state @ indexed_pauli
    )


# Stochastic Error Generator
def stochastic_error_generator(initial_state, indexed_pauli, identity):
    return 1 * (
        indexed_pauli @ initial_state @ indexed_pauli
        - identity @ initial_state @ identity
    )


# Pauli-correlation Error Generator
def pauli_correlation_error_generator(
    initial_state,
    pauli_index_1,
    pauli_index_2,
):
    return 2 * (
        pauli_index_1 @ initial_state @ pauli_index_2
        + pauli_index_2 @ initial_state @ pauli_index_1
        - 0.5 * commute(commute(pauli_index_1, pauli_index_2), initial_state)
    )


# Anti-symmetric Error Generator
def anti_symmetric_error_generator(initial_state, pauli_index_1, pauli_index_2):
    return 2j * (
        pauli_index_1 @ initial_state @ pauli_index_2
        - pauli_index_2 @ initial_state @ pauli_index_1
        + 0.5
        * commute(
            anti_commute(pauli_index_1, pauli_index_2),
            initial_state,
        )
    )



def coverage_edge_exists(error_gen_type, pauli_index, prep_string, meas_string):
    # print(f"{error_gen_type}")
    # print(f"{pauli_index}")
    # print(f"{prep_string}")
    # print(f"{meas_string}")
    if error_gen_type == "h":
        prep_ham_idx_comm = idtcore.half_pauli_comm(pauli_index, prep_string)
        # The paulis are trace orthonormal, so we only get a non-zero value
        # if meas_overall ~= prep_ham_idx_comm (i.e. up to the overall sign/phase).
        if prep_ham_idx_comm == 0:
            return False
        if idtcore.is_unsigned_pauli_equiv(meas_string, prep_ham_idx_comm):
            return True
    elif error_gen_type == "s":
        if not pauli_index.commutes(prep_string):
            if idtcore.is_unsigned_pauli_equiv(meas_string, prep_string):
                return True
    return False


def alt_coverage_edge_exists(error_gen_type, pauli_index, prep_string, meas_string):
    if error_gen_type == "h":
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")
        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )
        prep_string_iterator = [pstring for pstring in prep_string_iterator_extended if (set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X")) and set(pstring.pauli_indices("Y")).issubset(prep_string.pauli_indices("Y")) and set(pstring.pauli_indices("Z")).issubset(prep_string.pauli_indices("Z")))]
        meas_string_iterator_extended =stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_meas
        )
        meas_string_iterator = [pstring for pstring in meas_string_iterator_extended if (set(pstring.pauli_indices("X")).issubset(meas_string.pauli_indices("X")) and set(pstring.pauli_indices("Y")).issubset(meas_string.pauli_indices("Y")) and set(pstring.pauli_indices("Z")).issubset(meas_string.pauli_indices("Z")))]
        
        # logger.info(f"Testing for: H[{pauli_index}]; Experiment ({prep_string}/{meas_string})")
        for mstring in meas_string_iterator:
            # logger.info(f"Evaluating for observable {mstring}")
            used_indices = [i for i,ltr in enumerate(str(mstring)[1:]) if ltr != "_"]
            if len(used_indices) == 0:
                continue
            # logger.info(f"Evaluating for observable {mstring}")
            t = 0
            for string in prep_string_iterator:
                # logger.info(f"Substring = {string}")
                used_indices = [i for i,ltr in enumerate(str(string)[1:]) if ltr != "_"]
                if len(used_indices) == 0:
                    # logger.info("Passing due to failed overlap test")
                    continue
                # logger.info("Continuing due to successful overlap test")
                ident = stim.PauliString(len(string))
                error_gen = hamiltonian_error_generator(
                    string.to_unitary_matrix(endian="little"),
                    pauli_index.to_unitary_matrix(endian="little"),
                    ident.to_unitary_matrix(endian="little"),
                )
                # want approx non-zero rather than strict
                if _np.any(error_gen):
                    # print(error_gen)
                    # print(error_gen/(_np.linalg.norm(error_gen)/2))
                    error_gen_string = stim.PauliString.from_unitary_matrix(error_gen / 2)
                    second_matrix = mstring * error_gen_string
                    # what is the correct coefficient here?
                    # t += (1 / 2**num_qubits) * _np.trace(
                    # logger.info(f"Current Trace: {t}")
                    t += (1 / 2**(len(string)-1))*_np.trace(second_matrix.to_unitary_matrix(endian="little"))
                    # logger.info(f"Updated Trace: {t}")
                    # print(t)
            if _np.absolute(t) > 0.0001:
                logger.info(f"Positive Match \n\nH[{pauli_index}]; Experiment ({prep_string}/{meas_string}); Measureable {mstring}; Coef {t}\n")
                # return _np.real_if_close(t)
    elif error_gen_type == "s":
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )

        prep_string_iterator = [pstring for pstring in prep_string_iterator_extended if (set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X")) and set(pstring.pauli_indices("Y")).issubset(prep_string.pauli_indices("Y")) and set(pstring.pauli_indices("Z")).issubset(prep_string.pauli_indices("Z")))]

        t = 0
        logger.info(f"Testing for: S[{pauli_index}]; Experiment ({prep_string}/{meas_string})")
        for string in prep_string_iterator:
            logger.info(f"Substring = {string}")
            used_indices = [i for i,ltr in enumerate(str(string)[1:]) if ltr != "_"]
            pauli_used_indices = [i for i,ltr in enumerate(str(pauli_index)[1:]) if ltr != "_"]
            if len(used_indices) == 0 or used_indices != pauli_used_indices:
                continue
            string = stim.PauliString(''.join(str(string)[1:][i] for i in used_indices))

            new_pauli_index = stim.PauliString(''.join(str(pauli_index)[1:][i] for i in used_indices))
            ident = stim.PauliString(len(string))
            new_meas_string = stim.PauliString(''.join(str(meas_string)[1:][i] for i in used_indices))
            error_gen = stochastic_error_generator(
                string.to_unitary_matrix(endian="little"),
                new_pauli_index.to_unitary_matrix(endian="little"),
                ident.to_unitary_matrix(endian="little"),
            )
            # want approx non-zero rather than strict
            if _np.any(error_gen):
                # logger.info(f"Error Gen:\n{error_gen}")
                error_gen_string = stim.PauliString.from_unitary_matrix(error_gen / 2)
                second_matrix = new_meas_string * error_gen_string
                # what is the correct coefficient here?
                logger.info(f"Current Trace: {t}")
                # logger.info(f"Error Gen: \n{error_gen}")
                # logger.info(f"Error Gen String: {error_gen_string}")
                # logger.info(f"Second Matrix: {second_matrix}")
                # logger.info(f"Second Matrix as unitary: \n{second_matrix.to_unitary_matrix(endian='little')}")
                t += (1 / 2**(len(string)-1))*_np.trace(second_matrix.to_unitary_matrix(endian="little"))
                logger.info(f"Updated Trace: {t}")
                # print(t)
        if _np.absolute(t) > 0.0001:
            logger.info(f"Positive Match \n\nS[{pauli_index}]; Experiment ({prep_string}/{meas_string}); Coef {t}\n")
            return _np.real_if_close(t)
    elif error_gen_type == "c":
        pauli_index_1 = pauli_index[0]
        pauli_index_2 = pauli_index[1]
        logger.info(f"Testing for: C[{pauli_index_1,pauli_index_2}]; Experiment ({prep_string}/{meas_string})")
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )

        prep_string_iterator = [pstring for pstring in prep_string_iterator_extended if (set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X")) and set(pstring.pauli_indices("Y")).issubset(prep_string.pauli_indices("Y")) and set(pstring.pauli_indices("Z")).issubset(prep_string.pauli_indices("Z")))]
        
        t = 0
        logger.info(f"Testing for: C[{pauli_index}]; Experiment ({prep_string}/{meas_string})")
        for string in prep_string_iterator:
            logger.info(f"Substring = {string}")
            used_indices = [i for i,ltr in enumerate(str(string)[1:]) if ltr != "_"]
            pauli_used_indices_1 = [i for i,ltr in enumerate(str(pauli_index_1)[1:]) if ltr != "_"]
            pauli_used_indices_2 = [i for i,ltr in enumerate(str(pauli_index_2)[1:]) if ltr != "_"]
            if len(used_indices) == 0 or (used_indices != pauli_used_indices_1 and used_indices != pauli_used_indices_2):
                continue
            string = stim.PauliString(''.join(str(string)[1:][i] for i in used_indices))

            new_pauli_index_1 = stim.PauliString(''.join(str(pauli_index_1)[1:][i] for i in used_indices))
            new_pauli_index_2 = stim.PauliString(''.join(str(pauli_index_2)[1:][i] for i in used_indices))
            new_meas_string = stim.PauliString(''.join(str(meas_string)[1:][i] for i in used_indices))
            error_gen = pauli_correlation_error_generator(
                string.to_unitary_matrix(endian="little"),
                new_pauli_index_1.to_unitary_matrix(endian="little"),
                new_pauli_index_2.to_unitary_matrix(endian="little"),
            )
            # want approx non-zero rather than strict
            if _np.any(error_gen):
                # logger.info(f"Error Gen:\n{error_gen}")
                error_gen_string = stim.PauliString.from_unitary_matrix(error_gen / _np.linalg.norm(error_gen,ord=_np.inf))
                second_matrix = new_meas_string * error_gen_string
                # what is the correct coefficient here?
                # t += (1 / 2**num_qubits) * _np.trace(
                t += (1 / 2**(len(string)-1))*_np.trace(second_matrix.to_unitary_matrix(endian="little"))
                # print(t)
        if _np.absolute(t) > 0.0001:
            logger.info(f"Positive match \n\nC[{pauli_index_1,pauli_index_2}]; Experiment ({prep_string}/{meas_string}); Coef {t}\n")
            return _np.real_if_close(t)
                
    elif error_gen_type == "a":
        pauli_index_1 = pauli_index[0]
        pauli_index_2 = pauli_index[1]
        logger.info(f"Testing for: A[{pauli_index_1,pauli_index_2}]; Experiment ({prep_string}/{meas_string})")
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )

        prep_string_iterator = [pstring for pstring in prep_string_iterator_extended if (set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X")) and set(pstring.pauli_indices("Y")).issubset(prep_string.pauli_indices("Y")) and set(pstring.pauli_indices("Z")).issubset(prep_string.pauli_indices("Z")))]
        
        t = 0
        logger.info(f"Testing for: A[{pauli_index}]; Experiment ({prep_string}/{meas_string})")
        for string in prep_string_iterator:
            logger.info(f"Substring = {string}")
            used_indices = [i for i,ltr in enumerate(str(string)[1:]) if ltr != "_"]
            pauli_used_indices_1 = [i for i,ltr in enumerate(str(pauli_index_1)[1:]) if ltr != "_"]
            pauli_used_indices_2 = [i for i,ltr in enumerate(str(pauli_index_2)[1:]) if ltr != "_"]
            if len(used_indices) == 0 or (used_indices != pauli_used_indices_1 and used_indices != pauli_used_indices_2):
                continue
            string = stim.PauliString(''.join(str(string)[1:][i] for i in used_indices))

            new_pauli_index_1 = stim.PauliString(''.join(str(pauli_index_1)[1:][i] for i in used_indices))
            new_pauli_index_2 = stim.PauliString(''.join(str(pauli_index_2)[1:][i] for i in used_indices))
            new_meas_string = stim.PauliString(''.join(str(meas_string)[1:][i] for i in used_indices))
            error_gen = anti_symmetric_error_generator(
                string.to_unitary_matrix(endian="little"),
                new_pauli_index_1.to_unitary_matrix(endian="little"),
                new_pauli_index_2.to_unitary_matrix(endian="little"),
            )
            # want approx non-zero rather than strict
            if _np.any(error_gen):
                logger.info(f"Error Gen:\n{error_gen}")
                error_gen_string = stim.PauliString.from_unitary_matrix(error_gen / _np.linalg.norm(error_gen,ord=_np.inf))
                second_matrix = new_meas_string * error_gen_string
                # what is the correct coefficient here?
                # t += (1 / 2**num_qubits) * _np.trace(
                t += (1 / 2**(len(string)-1))*_np.trace(second_matrix.to_unitary_matrix(endian="little"))
                # print(t)
        if _np.absolute(t) > 0.0001:
            logger.info(f"Positive Match\n\nA[{pauli_index_1,pauli_index_2}]; Experiment ({prep_string}/{meas_string}); Coef {t}\n")
            return _np.real_if_close(t)
    return False




num_qubits = 2
max_weight = 2

HS_index_iterator = stim.PauliString.iter_all(
    num_qubits, min_weight=1, max_weight=max_weight
)

pauli_node_attributes = list([p for p in HS_index_iterator])
# this one is currently generating too many combinations
# unrelated to this, corre
ca_pauli_node_attributes = list(_itertools.combinations(pauli_node_attributes,2))

measure_string_iterator = stim.PauliString.iter_all(num_qubits, min_weight=num_qubits)
measure_string_attributes = list([p for p in measure_string_iterator])
prep_string_attributes = measure_string_attributes
prep_meas_pairs = list(product(prep_string_attributes, measure_string_attributes))


# print(prep_meas_pairs)

ident = stim.PauliString(num_qubits)


test_graph = nx.Graph()
# test_graph.add_nodes_from(enumerate(pauli_node_attributes), pauli_string = pauli_node_attributes, bipartite=1)
for i, j in prep_meas_pairs:
    test_graph.add_node(
        len(test_graph.nodes) + 1, prep_string=i, meas_string=j, bipartite=0
    )

#error_gen_classes = "h"
hs_error_gen_classes = "hs"
hs_error_gen_classes="h"
ca_error_gen_classes = "ca"
ca_error_gen_classes = ""

for j in hs_error_gen_classes:
    for i in range(len(pauli_node_attributes)):
        test_graph.add_node(
            len(test_graph.nodes) + 1,
            error_gen_class=j,
            pauli_index=pauli_node_attributes[i],
            bipartite=1,
        )

for j in ca_error_gen_classes:
    for i in range(len(ca_pauli_node_attributes)):
        test_graph.add_node(
            len(test_graph.nodes) + 1,
            error_gen_class=j,
            pauli_index=ca_pauli_node_attributes[i],
            bipartite=1,
        )

# print(test_graph.nodes[88])
# print([test_graph.nodes[node] for node in test_graph.nodes])
# quit()
bipartite_identifier = nx.get_node_attributes(test_graph, "bipartite")
# hey rewrite this to not be stupid.  or at least less stupid.
bipartite_pairs = [
    (k1, k2)
    for k1 in bipartite_identifier.keys()
    if bipartite_identifier[k1] == 0
    for k2 in bipartite_identifier.keys()
    if bipartite_identifier[k2] == 1
]

for pair in bipartite_pairs:
    n = alt_coverage_edge_exists(
        test_graph.nodes[pair[1]]["error_gen_class"],
        test_graph.nodes[pair[1]]["pauli_index"],
        test_graph.nodes[pair[0]]["prep_string"],
        test_graph.nodes[pair[0]]["meas_string"],
    )
    if n:
        test_graph.add_edge(pair[0], pair[1], coef = n)


# print(list(test_graph.nodes[n] for n in test_graph.nodes))
# print(list(edge for edge in test_graph.edges if edge[1]==87))
# quit()
# print(test_graph.nodes[88])
labels = {n: "" for n in test_graph.nodes}
pos = {n: (0, 0) for n in test_graph.nodes}
x_pos_err = 0
x_pos_exp = 0
# save_me = [2]
for n in test_graph.nodes:
    if test_graph.nodes[n].get("pauli_index"):
        labels[n] = (
            str(test_graph.nodes[n].get("error_gen_class"))
            + "_"
            + str(test_graph.nodes[n].get("pauli_index"))
        )
        pos[n] = (x_pos_err, 1)
        x_pos_err += 7000
    #         save_me.append(n)
    else:
        labels[n] = (
            str(test_graph.nodes[n]["prep_string"])
            + " / "
            + str(test_graph.nodes[n]["meas_string"])
        )
        pos[n] = (x_pos_exp, 0)
        x_pos_exp += 1000

# hxy_subgraph = nx.subgraph(test_graph, save_me)


# x_pos_err = 0
# x_pos_exp = 0
# for n in hxy_subgraph.nodes:
#     if test_graph.nodes[n].get("pauli_index"):
#         labels[n] = str(test_graph.nodes[n].get("error_gen_class")) + "_" + str(test_graph.nodes[n].get("pauli_index"))
#         pos[n] = (x_pos_err, 3)
#         x_pos_err += 2
#     else:
#         labels[n] = str(test_graph.nodes[n]["prep_string"]) + " / " + str(test_graph.nodes[n]["meas_string"])
#         pos[n] = (x_pos_exp, 0)
#         x_pos_exp += 2


# nx.draw(test_graph, nx.kamada_kawai_layout(test_graph))
# plt.figure(figsize=(11, 8.5))
# nx.draw_networkx_nodes(test_graph, pos, node_size=15)
# nx.draw_networkx_edges(test_graph, pos)
# nx.draw_networkx_labels(test_graph, pos, labels=labels, font_size=2)
# plt.savefig("dum_graf.pdf")
# quit()

with open("two_qubit_weight_one_test.txt", "w") as f:
    for k,v,d in test_graph.edges.data():
        f.write("\n")
        f.write("Experiment " + str(test_graph.nodes[k]["prep_string"]) + " / " + str(test_graph.nodes[k]["meas_string"]) + " is sensitive to the error generator " + str(test_graph.nodes[v].get("error_gen_class")).capitalize() + "[" + str(test_graph.nodes[v].get("pauli_index"))[1:] + "] with coefficient " + str(d["coef"]))
        f.write("\n")

quit()





m = pe.ConcreteModel()
m.covering_nodes = [
    n for n in test_graph.nodes if test_graph.nodes[n]["bipartite"] == 0
]
m.error_generator_nodes = [
    n for n in test_graph.nodes if test_graph.nodes[n]["bipartite"] == 1
]
m.edges = test_graph.edges
m.num_qubits = num_qubits
# print(m.edges)
m.experiment_choice = pe.Var(m.covering_nodes, domain=pe.Binary, initialize=0)
m.known_error_generators = pe.Var(
    m.error_generator_nodes, domain=pe.Binary, initialize=0
)
m.information_streams = pe.Var(m.edges, domain=pe.Binary, initialize=0)

# @m.Constraint(m.error_generator_nodes)
# def covering_logic_rule(m,covered_node):
#     return sum(m.experiment_choice[covering_node] for (covering_node,cov_node) in m.edges if cov_node==covered_node) >= m.known_error_generators[covered_node]

# @m.Constraint(m.error_generator_nodes)
# def full_knowledge_rule(m, covered_node):
#     return sum(m.experiment_choice[covering_node] for (covering_node,cov_node) in m.edges if cov_node == covered_node) >= nx.degree(test_graph, covered_node)


@m.Constraint(m.edges)
def error_gen_covering_rule(m, *edge):
    return m.known_error_generators[edge[1]] >= m.information_streams[edge]


@m.Constraint(m.edges)
def experiment_covering_rule(m, *edge):
    return m.experiment_choice[edge[0]] >= m.information_streams[edge]


@m.Constraint(m.error_generator_nodes)
def error_gen_selection_rule(m, node):
    return m.known_error_generators[node] <= sum(
        m.information_streams[edge] for edge in m.edges if edge[1] == node
    )


@m.Constraint(m.covering_nodes)
def experiment_selection_rule(m, node):
    return m.experiment_choice[node] <= sum(
        m.information_streams[edge] for edge in m.edges if edge[0] == node
    )


@m.Constraint(m.covering_nodes)
def saturation_rule(m, covering_node):
    if nx.degree(test_graph, covering_node) == 0:
        return pe.Constraint.Skip
    return (
        sum(m.information_streams[edge] for edge in m.edges if edge[0] == covering_node)
        <= 2**m.num_qubits - 1
    )


@m.Constraint(m.error_generator_nodes)
def full_coverage(m, covered_node):
    return m.known_error_generators[covered_node] >= 1


m.obj = pe.Objective(expr=sum(m.experiment_choice[n] for n in m.covering_nodes))
with open("hahahahano.txt", "w") as f:
    m.pprint(f)
opt = pe.SolverFactory("gurobi")
opt.solve(m, tee=True)
# m.pprint()
print(f"{pe.value(m.obj)}")
import re

info_streams = []
exp_egen_pairs = []
for v in m.component_data_objects(ctype=pe.Var):
    if "exp" in v.name and pe.value(v) >= 0.001:
        print(v.name, pe.value(v))
    if "inf" in v.name and pe.value(v) >= 0.001:
        print(v.name)
        nums = re.findall(r"\d+", v.name)
        info_streams.append((int(nums[0]), int(nums[1])))

for info_stream in info_streams:
    egen_class = (
        test_graph.nodes[info_stream[1]]["error_gen_class"]
        + str(test_graph.nodes[info_stream[1]]["pauli_index"])[1:]
    )
    prep_string = test_graph.nodes[info_stream[0]]["prep_string"]
    meas_string = test_graph.nodes[info_stream[0]]["meas_string"]
    exp_egen_pairs.append(((prep_string, meas_string), egen_class))

for pair in exp_egen_pairs:
    print("(" + str(pair[0][0])[1:] + "," + str(pair[0][1])[1:] + ") -----> " + pair[1])
