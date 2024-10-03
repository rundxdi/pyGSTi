import stim
import networkx as nx
import pyomo.environ as pe
from itertools import product
import matplotlib.pyplot as plt

# non-relative imports for now
import collections as _collections
import itertools as _itertools
import time as _time
import warnings as _warnings
from itertools import product, permutations
from pygsti.baseobjs import basisconstructors
import numpy as _np
from pygsti.baseobjs import Basis
import re
from icecream import ic
from ordered_set import OrderedSet

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="two_qubit_log.log", encoding="utf-8", level=logging.INFO, filemode="w"
)


# Commutator Helper Functions
def commute(mat1, mat2):
    return mat1 @ mat2 - mat2 @ mat1


def anti_commute(mat1, mat2):
    return mat1 @ mat2 + mat2 @ mat1


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
    return 1 * (
        pauli_index_1 @ initial_state @ pauli_index_2
        + pauli_index_2 @ initial_state @ pauli_index_1
        - 0.5 * anti_commute(anti_commute(pauli_index_1, pauli_index_2), initial_state)
    )


# Anti-symmetric Error Generator
def anti_symmetric_error_generator(initial_state, pauli_index_1, pauli_index_2):
    return 1j * (
        pauli_index_1 @ initial_state @ pauli_index_2
        - pauli_index_2 @ initial_state @ pauli_index_1
        + 0.5
        * anti_commute(
            commute(pauli_index_1, pauli_index_2),
            initial_state,
        )
    )


# Signing helper function for prep eigenstates
def pauli_prep_term_sign(sign_string, prep_string):
    sign = 1
    # ic(sign_string)
    # ic(prep_string)
    # ic(str(prep_string))
    # ic(list(prep_string))
    for psign,prep in zip(sign_string, list(prep_string)):
        if psign < 0 and prep:
            sign *= -1
    # ic(sign)
    return sign

def jacobian_coefficient_calc(error_gen_type, pauli_index, prep_string, meas_string):
    coef_list = []
    sign_string = prep_string[0]
    prep_string = prep_string[1]
    if error_gen_type == "h":
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")
        prep_string_iterator_extended = list(
            stim.PauliString.iter_all(num_qubits=num_qubits, allowed_paulis=stim_prep)
        )
        prep_string_iterator = [prep_string_iterator_extended[0]]
        prep_string_iterator += [
            pauli_prep_term_sign(sign_string, pstring) * pstring
            for pstring in prep_string_iterator_extended[1:]
            if (
                set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    prep_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    prep_string.pauli_indices("Z")
                )
            )
        ]
        # ic(prep_string_iterator)
        meas_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_meas
        )
        meas_string_iterator = [
            pstring
            for pstring in meas_string_iterator_extended
            if (
                set(pstring.pauli_indices("X")).issubset(meas_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    meas_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    meas_string.pauli_indices("Z")
                )
                and pstring.weight != 0
            )
        ]
        for mstring in meas_string_iterator:
            t = 0
            for string in prep_string_iterator:
                ident = stim.PauliString(len(string))
                error_gen = hamiltonian_error_generator(
                    string.to_unitary_matrix(endian="little"),
                    pauli_index.to_unitary_matrix(endian="little"),
                    ident.to_unitary_matrix(endian="little"),
                )
                if _np.any(error_gen):
                    norm = _np.linalg.norm(error_gen, ord=_np.inf)
                    error_gen_string = stim.PauliString.from_unitary_matrix(
                        error_gen / norm
                    )
                    second_matrix = mstring * error_gen_string
                    t += (
                        norm
                        * (1 / 2 ** (len(string)))
                        * _np.trace(second_matrix.to_unitary_matrix(endian="little"))
                    )
            if _np.absolute(t) > 0.0001:
                logger.info(
                    f"Positive Match \n\nH[{pauli_index}]; Experiment ({prep_string}/{meas_string}); Measureable {mstring}; Coef {t}\n"
                )
            coef_list.append(
                [
                    "H",
                    pauli_index,
                    prep_string,
                    sign_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]
            )

    elif error_gen_type == "s":
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")
        prep_string_iterator_extended = list(
            stim.PauliString.iter_all(num_qubits=num_qubits, allowed_paulis=stim_prep)
        )
        prep_string_iterator = [prep_string_iterator_extended[0]]
        prep_string_iterator += [
            pauli_prep_term_sign(sign_string, pstring) * pstring
            for pstring in prep_string_iterator_extended[1:]
            if (
                set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    prep_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    prep_string.pauli_indices("Z")
                )
            )
        ]
        meas_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_meas
        )
        meas_string_iterator = [
            pstring
            for pstring in meas_string_iterator_extended
            if (
                set(pstring.pauli_indices("X")).issubset(meas_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    meas_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    meas_string.pauli_indices("Z")
                )
                and pstring.weight != 0
            )
        ]
        for mstring in meas_string_iterator:
            t = 0
            for string in prep_string_iterator:
                ident = stim.PauliString(len(string))

                error_gen = stochastic_error_generator(
                    string.to_unitary_matrix(endian="little"),
                    pauli_index.to_unitary_matrix(endian="little"),
                    ident.to_unitary_matrix(endian="little"),
                )
                if _np.any(error_gen):
                    norm = _np.linalg.norm(error_gen, ord=_np.inf)
                    error_gen_string = stim.PauliString.from_unitary_matrix(
                        error_gen / norm
                    )
                    second_matrix = mstring * error_gen_string
                    t += (
                        norm
                        * (1 / 2 ** (len(string)))
                        * _np.trace(second_matrix.to_unitary_matrix(endian="little"))
                    )
            if _np.absolute(t) > 0.0001:
                logger.info(
                    f"Positive Match \n\nS[{pauli_index}]; Experiment ({prep_string}/{meas_string}); Observable {mstring} Coef {t}\n"
                )
            coef_list.append(
                [
                    "S",
                    pauli_index,
                    prep_string,
                    sign_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]
            )
    elif error_gen_type == "c":
        pauli_index_1 = pauli_index[0]
        pauli_index_2 = pauli_index[1]
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")

        prep_string_iterator_extended = list(
            stim.PauliString.iter_all(num_qubits=num_qubits, allowed_paulis=stim_prep)
        )
        prep_string_iterator = [prep_string_iterator_extended[0]]
        prep_string_iterator += [
            pauli_prep_term_sign(sign_string, pstring) * pstring
            for pstring in prep_string_iterator_extended[1:]
            if (
                set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    prep_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    prep_string.pauli_indices("Z")
                )
            )
        ]
        meas_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_meas
        )
        meas_string_iterator = [
            pstring
            for pstring in meas_string_iterator_extended
            if (
                set(pstring.pauli_indices("X")).issubset(meas_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    meas_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    meas_string.pauli_indices("Z")
                )
                and pstring.weight != 0
            )
        ]
        for mstring in meas_string_iterator:
            t = 0
            for string in prep_string_iterator:
                error_gen = pauli_correlation_error_generator(
                    string.to_unitary_matrix(endian="little"),
                    pauli_index_1.to_unitary_matrix(endian="little"),
                    pauli_index_2.to_unitary_matrix(endian="little"),
                )

                if _np.any(error_gen):
                    norm = _np.linalg.norm(error_gen, ord=_np.inf)
                    error_gen_string = stim.PauliString.from_unitary_matrix(
                        error_gen / _np.linalg.norm(error_gen, ord=_np.inf)
                    )
                    second_matrix = mstring * error_gen_string
                    t += (
                        norm
                        * (1 / 2 ** (len(string)))
                        * _np.trace(second_matrix.to_unitary_matrix(endian="little"))
                    )
            if _np.absolute(t) > 0.0001:
                logger.info(
                    f"Positive match \n\nC[{pauli_index_1},{pauli_index_2}]; Experiment ({prep_string}/{meas_string}); Observable {mstring}; Coef {t}\n"
                )
            coef_list.append(
                [
                    "C",
                    pauli_index_1,
                    pauli_index_2,
                    prep_string,
                    sign_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]
            )

    elif error_gen_type == "a":
        pauli_index_1 = pauli_index[0]
        pauli_index_2 = pauli_index[1]
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")

        prep_string_iterator_extended = list(
            stim.PauliString.iter_all(num_qubits=num_qubits, allowed_paulis=stim_prep)
        )
        prep_string_iterator = [prep_string_iterator_extended[0]]
        prep_string_iterator += [
            pauli_prep_term_sign(sign_string, pstring) * pstring
            for pstring in prep_string_iterator_extended[1:]
            if (
                set(pstring.pauli_indices("X")).issubset(prep_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    prep_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    prep_string.pauli_indices("Z")
                )
            )
        ]

        # ic(prep_string_iterator)
        meas_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_meas
        )
        meas_string_iterator = [
            pstring
            for pstring in meas_string_iterator_extended
            if (
                set(pstring.pauli_indices("X")).issubset(meas_string.pauli_indices("X"))
                and set(pstring.pauli_indices("Y")).issubset(
                    meas_string.pauli_indices("Y")
                )
                and set(pstring.pauli_indices("Z")).issubset(
                    meas_string.pauli_indices("Z")
                )
                and pstring.weight != 0
            )
        ]
        for mstring in meas_string_iterator:
            t = 0
            for string in prep_string_iterator:
                # ic(string.to_unitary_matrix(endian="little"))
                error_gen = anti_symmetric_error_generator(
                    string.to_unitary_matrix(endian="little"),
                    pauli_index_1.to_unitary_matrix(endian="little"),
                    pauli_index_2.to_unitary_matrix(endian="little"),
                )
                if _np.any(error_gen):
                    norm = _np.linalg.norm(error_gen, ord=_np.inf)
                    error_gen_string = stim.PauliString.from_unitary_matrix(
                        error_gen / _np.linalg.norm(error_gen, ord=_np.inf)
                    )
                    second_matrix = mstring * error_gen_string
                    t += (
                        norm
                        * (1 / 2 ** (len(string)))
                        * _np.trace(second_matrix.to_unitary_matrix(endian="little"))
                    )
            if _np.absolute(t) > 0.0001:
                logger.info(
                    f"Positive Match\n\nA[{pauli_index_1},{pauli_index_2}]; Experiment ({prep_string}/{meas_string}); Observable {mstring}; Coef {t}\n"
                )
            coef_list.append(
                [
                    "A",
                    pauli_index_1,
                    pauli_index_2,
                    prep_string,
                    sign_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]
            )
    return coef_list


if __name__ == "__main__":
    num_qubits = 3
    max_weight = 2

    HS_index_iterator = stim.PauliString.iter_all(
        num_qubits, min_weight=1, max_weight=max_weight
    )

    pauli_node_attributes = list([p for p in HS_index_iterator])
    ca_pauli_node_attributes = list(_itertools.combinations(pauli_node_attributes, 2))

    def ca_pauli_weight_filter(pauli_pair, max_weight):
        used_indices_1 = set(
            i for i, ltr in enumerate(str(pauli_pair[0])[1:]) if ltr != "_"
        )
        used_indices_2 = set(
            i for i, ltr in enumerate(str(pauli_pair[1])[1:]) if ltr != "_"
        )
        intersect = used_indices_1.intersection(used_indices_2)
        if len(intersect) > 0 and len(intersect) <= max_weight:
            return True

    ca_pauli_node_attributes = [
        ppair
        for ppair in ca_pauli_node_attributes
        if ca_pauli_weight_filter(ppair, max_weight)
    ]

    measure_string_iterator = stim.PauliString.iter_all(
        num_qubits, min_weight=num_qubits
    )
    measure_string_attributes = list([p for p in measure_string_iterator])

    hs_error_gen_classes = "hs"
    ca_error_gen_classes = "ca"

    hs_experiment = list(
        product(
            hs_error_gen_classes,
            pauli_node_attributes,
            measure_string_attributes,
            measure_string_attributes,
        )
    )
    ca_experiment = list(
        product(
            ca_error_gen_classes,
            ca_pauli_node_attributes,
            measure_string_attributes,
            measure_string_attributes,
        )
    )

    import pandas as pd

    # df = pd.DataFrame()
    jacobian_coef_dict = {"index": OrderedSet(), "columns": OrderedSet()}
    data = {}

    # These come back as class, index, prep_str, meas_str, observ_str: coef
    # I THINK this is right, should double check and write unit tests
    for key in hs_experiment + ca_experiment:
        elt = jacobian_coefficient_calc(*key)
        for el in elt:
            if el:
                observable = ",".join(str(s)[1:] for s in el[-4:-1])
                egen = ",".join(str(s) for s in el[:-4])
                coef = int(el[-1])
                jacobian_coef_dict["index"].add(observable)
                jacobian_coef_dict["columns"].add(egen)
                if data.get(egen):
                    data[egen].append(coef)
                else:
                    data[egen] = [coef]

    df = pd.DataFrame(
        data, index=jacobian_coef_dict["index"], columns=jacobian_coef_dict["columns"]
    )
    ic(df)

    # whatever = df.to_numpy()
    # inv = _np.linalg.pinv(whatever)
    # ic(whatever)
    # ic(inv)

    # quit()

    # hs_experiment = list(
    #     product(
    #         hs_error_gen_classes,
    #         pauli_node_attributes,
    #         measure_string_attributes,
    #         measure_string_attributes,
    #     )
    # )
    # ca_experiment = list(
    #     product(
    #         ca_error_gen_classes,
    #         ca_pauli_node_attributes,
    #         measure_string_attributes,
    #         measure_string_attributes,
    #     )
    # )

    # jacobian_coefficient_dict = {"H": [], "S": [], "C": [], "A": []}
    # experiment_set = OrderedSet()

    # egen_dict = {"H": OrderedSet(), "S": OrderedSet(), "C": OrderedSet(), "A": OrderedSet()}

    # # These come back as class, index, prep_str, meas_str, observ_str: coef
    # for key in hs_experiment + ca_experiment:
    #     elt = jacobian_coefficient_calc(*key)
    #     # ic(elt)
    #     for el in elt:
    #         # ic(el)
    #         if el:
    #             experiment = ",".join(str(s)[1:] for s in el[-4:-1])
    #             observable = str(el[-2])[1:]
    #             egen = ",".join(str(s) for s in el[:-4])
    #             egen_class = el[0]
    #             coef = int(el[-1])
    #             jacobian_coefficient_dict[egen_class].append(coef)
    #             experiment_set.add(experiment)

    #             egen_dict[egen_class].add(egen)
    #             # ic(jacobian_coefficient_dict)

    # egen_list = _np.array([v for v in egen_dict.values()]).flatten()
    # # for k,v in jacobian_coefficient_dict.items():
    # #     ic(k,v)
    # h_matrix = _np.array(jacobian_coefficient_dict["H"]).reshape(len(experiment_set), len(egen_dict["H"]))
    # s_matrix = _np.array(jacobian_coefficient_dict["S"]).reshape(len(experiment_set), len(egen_dict["S"]))
    # c_matrix = _np.array(jacobian_coefficient_dict["C"]).reshape(len(experiment_set), len(egen_dict["C"]))
    # a_matrix = _np.array(jacobian_coefficient_dict["A"]).reshape(len(experiment_set), len(egen_dict["A"]))
    # jacobian = _np.concatenate([h_matrix,s_matrix, c_matrix,a_matrix], axis=1)
    # ic(jacobian)
    # df = pd.DataFrame(jacobian, index=experiment_set, columns=egen_list)
    # ic(df)
