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
from icecream import ic

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


def jacobian_coefficient_calc(error_gen_type, pauli_index, prep_string, meas_string):
    if error_gen_type == "h":
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")
        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )
        prep_string_iterator = [
            pstring
            for pstring in prep_string_iterator_extended
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
        h_coef_list = []
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
            h_coef_list.append(
                [
                    "H",
                    pauli_index,
                    prep_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]
            )
        return h_coef_list
    elif error_gen_type == "s":
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")
        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )

        prep_string_iterator = [
            pstring
            for pstring in prep_string_iterator_extended
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
                return [
                    "S",
                    pauli_index,
                    prep_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]
    elif error_gen_type == "c":
        pauli_index_1 = pauli_index[0]
        pauli_index_2 = pauli_index[1]
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")

        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )

        prep_string_iterator = [
            pstring
            for pstring in prep_string_iterator_extended
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
                return [
                    "C",
                    pauli_index_1,
                    pauli_index_2,
                    prep_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]

    elif error_gen_type == "a":
        pauli_index_1 = pauli_index[0]
        pauli_index_2 = pauli_index[1]
        num_qubits = len(prep_string)
        stim_prep = str(prep_string).strip("+-")
        stim_meas = str(meas_string).strip("+-")

        prep_string_iterator_extended = stim.PauliString.iter_all(
            num_qubits=num_qubits, allowed_paulis=stim_prep
        )

        prep_string_iterator = [
            pstring
            for pstring in prep_string_iterator_extended
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
            )
        ]
        for mstring in meas_string_iterator:
            t = 0
            for string in prep_string_iterator:
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
                return [
                    "A",
                    pauli_index_1,
                    pauli_index_2,
                    prep_string,
                    meas_string,
                    mstring,
                    _np.real_if_close(t),
                ]


if __name__ == "__main__":
    num_qubits = 1
    max_weight = 1

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

    hs_error_gen_classes = "h"
    ca_error_gen_classes = ""

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

    jacobian_coefficient_dict = {}

    # These come back as class, index, prep_str, meas_str, observ_str: coef
    for key in hs_experiment + ca_experiment:
        elt = jacobian_coefficient_calc(*key)
        ic(elt)
        for el in elt:
            if el:
                jacobian_coefficient_dict[tuple(str(e) for e in el[:-1])] = int(el[-1])

ic(jacobian_coefficient_dict)
