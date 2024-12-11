from ordered_set import OrderedSet
from itertools import product
import itertools as _itertools
import stim
from idtcorev2 import jacobian_coefficient_calc
import pygsti
from pygsti.circuits.circuit import Circuit
from pygsti.baseobjs import CompleteElementaryErrorgenBasis, QubitSpace, Label, Basis
import pandas as pd
import math
import numpy as np
from pygsti.extras import idletomography as idt
from idttools import allerrors, all_full_length_observables, alloutcomes
import collections as _collections
import json
import more_itertools
import re


def generate_experiment_design_stuff(num_qubits, max_weight):
    HS_index_iterator = stim.PauliString.iter_all(
        num_qubits, min_weight=1, max_weight=max_weight
    )

    pauli_node_attributes = list([p for p in HS_index_iterator])
    ca_pauli_node_attributes = list(_itertools.permutations(pauli_node_attributes, 2))

    def ca_pauli_weight_filter(pauli_pair, max_weight):
        used_indices_1 = set(
            i for i, ltr in enumerate(str(pauli_pair[0])[1:]) if ltr != "_"
        )
        used_indices_2 = set(
            i for i, ltr in enumerate(str(pauli_pair[1])[1:]) if ltr != "_"
        )
        union = used_indices_1.union(used_indices_2)
        if len(union) > 0 and len(union) <= max_weight:
            return True

    ca_pauli_node_attributes = [
        ppair
        for ppair in ca_pauli_node_attributes
        if ca_pauli_weight_filter(ppair, max_weight)
    ]
    measure_string_iterator = stim.PauliString.iter_all(
        num_qubits, min_weight=num_qubits
    )
    sign_iterator = list(product([1, -1], repeat=num_qubits))
    awk_iterator = list(
        i[0] for i in product(sign_iterator, [p for p in measure_string_iterator])
    )
    prep_string_iterator = product(
        [math.prod([item for item in sign_tuple]) for sign_tuple in sign_iterator],
        [p for p in measure_string_iterator],
    )
    measure_string_attributes = list([p for p in measure_string_iterator])
    prep_string_attributes = list(a * b for a, b in prep_string_iterator)
    return (
        awk_iterator,
        pauli_node_attributes,
        prep_string_attributes,
        measure_string_attributes,
        ca_pauli_node_attributes,
    )


def jacobian_df(
    hs_error_gen_classes,
    ca_error_gen_classes,
    pauli_node_attributes,
    prep_string_attributes,
    measure_string_attributes,
    ca_pauli_node_attributes,
    awk_iterator,
):

    hs_experiment = list(
        product(
            hs_error_gen_classes,
            pauli_node_attributes,
            zip(awk_iterator, prep_string_attributes),
            measure_string_attributes,
        )
    )
    ca_experiment = list(
        product(
            ca_error_gen_classes,
            ca_pauli_node_attributes,
            zip(awk_iterator, prep_string_attributes),
            measure_string_attributes,
        )
    )

    # df = pd.DataFrame()
    jacobian_coef_dict = {"index": OrderedSet(), "columns": OrderedSet()}
    data = {}

    # These come back as class, index, prep_str, meas_str, observ_str: coef
    # I THINK this is right, should double check and write unit tests
    for key in hs_experiment + ca_experiment:
        elt = jacobian_coefficient_calc(*key)
        for el in elt:
            if el:
                observable = ",".join(str(s)[:] for s in el[-5:-1])
                sign_string = el[-4]
                egen = ",".join(str(s) for s in el[:-5])
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
    return df


def pauli_pairs_to_circuits(pauli_pair, pauli_basis_dict):
    sgn = pauli_pair[0].sign
    circuits = []
    for pauli_str in pauli_pair:
        opstr = []
        for i, term in enumerate(str(pauli_str)[1:]):
            if i == 0:
                key = sgn + term
            else:
                key = "+" + term
            opstr.extend([Label(opname, i) for opname in pauli_basis_dict[key]])
        circuits.append(Circuit(opstr, num_lines=len(pauli_str)))
    return circuits


def make_idt_circuits(num_qubits):
    gates = ["Gi", "Gx", "Gy", "Gcnot"]
    max_lengths = [1, 2]
    pspec = pygsti.processors.QubitProcessorSpec(
        num_qubits,
        gates,
        geometry="line",
        nonstd_gate_unitaries={(): num_qubits, "Gi": np.eye((2**num_qubits))},
        availability={"Gi": [tuple(i for i in range(num_qubits))]},
    )
    mdl_target = pygsti.models.create_crosstalk_free_model(pspec)
    paulidicts = idt.determine_paulidicts(mdl_target)
    global_idle_string = [Label("Gi", tuple(i for i in range(num_qubits)))]
    idle_experiments = idt.make_idle_tomography_list(
        num_qubits, max_lengths, paulidicts, idle_string=global_idle_string
    )
    return idle_experiments, pspec, paulidicts


def create_noise_model(term_dict, mdl_pspec):
    mdl_datagen = pygsti.models.create_crosstalk_free_model(
        mdl_pspec,
        lindblad_error_coeffs={"Gi": term_dict},
        lindblad_parameterization="GLND",
    )
    return mdl_datagen


def simulate_noisy_idt(mdl_datagen, idle_experiments, seed=None):
    # Error models! Random with right CP constraints from Taxonomy paper
    ds = pygsti.data.simulate_data(
        mdl_datagen, idle_experiments, 10000000, seed=seed, sample_error="none"
    )
    return ds


def report_observed_rates(
    nqubits, dataset, max_lengths, pauli_basis_dicts, maxweight=2, idle_string=None
):

    idle_string = [Label("Gi", tuple(i for i in range(nqubits)))]

    all_fidpairs = dict(enumerate(idt.idle_tomography_fidpairs(nqubits)))
    if nqubits == 1:  # special case where line-labels may be ('*',)
        if len(dataset) > 0:
            first_circuit = list(dataset.keys())[0]
            line_labels = first_circuit.line_labels
        else:
            line_labels = (0,)
        GiStr = Circuit(idle_string, line_labels=line_labels)
    else:
        GiStr = Circuit(idle_string, num_lines=nqubits)
    obs_infos = dict()
    errors = allerrors(nqubits, maxweight)
    fit_order = 1
    observed_error_rates = {}
    obs_error_rates_by_exp = {}
    whatever = {}
    for ifp, pauli_fidpair in all_fidpairs.items():
        all_outcomes = idt.idttools.allobservables(pauli_fidpair[1], maxweight)
        infos_for_this_fidpair = _collections.OrderedDict()

        for j, out in enumerate(all_outcomes):
            info = idt.compute_observed_err_rate(
                dataset,
                pauli_fidpair,
                pauli_basis_dicts,
                GiStr,
                out,
                max_lengths,
                fit_order,
            )

            infos_for_this_fidpair[out] = info

            obs_infos[ifp] = infos_for_this_fidpair
            observed_error_rates[ifp] = [
                info["rate"] for info in infos_for_this_fidpair.values()
            ]
            obs_error_rates_by_exp[
                str(pauli_fidpair[0]).replace("+", ""),
                str(pauli_fidpair[1]).replace("+", ""),
                str(out).replace("I", "_"),
            ] = [info["rate"] for info in infos_for_this_fidpair.values()][-1]
        whatever[pauli_fidpair] = 1
    return observed_error_rates, obs_error_rates_by_exp


def observed_rates_to_intrinsic(j_df, observed_rates):
    j = j_df.to_numpy()
    jinv = np.linalg.pinv(j)
    intrins_errs = jinv @ observed_rates
    return intrins_errs


def parse_key(key):
    regexed_key = re.findall(r"\w+", key)
    parsed_key = {}
    parsed_key["error_gen_class"] = regexed_key[0]
    if parsed_key["error_gen_class"] in "HS":
        parsed_key["error_gen_index"] = regexed_key[1:2]
        parsed_key["qubit_index"] = regexed_key[2:-2]
    else:
        parsed_key["error_gen_index"] = regexed_key[1:3]
        parsed_key["qubit_index"] = regexed_key[3:-2]
    parsed_key["error_rate"] = str.join(".", regexed_key[-2:])
    return parsed_key


def parse_json(filename, num_qubits=2):
    with open(filename, "r") as fr:
        results = json.load(fr)
    for k, v in results.items():

        parsed_key = parse_key(k)
        if parsed_key["error_gen_class"] in "HS":
            placeholder_str = ["_"] * num_qubits
            for i, j in zip(
                parsed_key["qubit_index"],
                parsed_key["error_gen_index"][0],
            ):
                placeholder_str[int(i)] = j
            placeholder_str = str.join("", ["+"] + placeholder_str)
            placeholder_str = str.join(
                ",", [parsed_key["error_gen_class"]] + [placeholder_str]
            )
        else:
            placeholder_str = [["_"] * num_qubits, ["_"] * num_qubits]
            for j in range(2):
                for i, k in zip(
                    parsed_key["qubit_index"], parsed_key["error_gen_index"][j]
                ):
                    placeholder_str[j][int(i)] = k
                placeholder_str[j] = str.join("", ["+"] + placeholder_str[j])
            placeholder_str = str.join(
                ",", [parsed_key["error_gen_class"]] + placeholder_str
            )
        placeholder_str = placeholder_str.replace("I", "_")
        print(placeholder_str, parsed_key["error_rate"], v.get(placeholder_str))
        # exit()
    return results


## C _X, * is creating too few things....

_predefined_rules = ["log", "quad", "lin"]


def length_rule(min_length=1, max_length=64, iter_rule="log"):
    max_lengths = []
    if iter_rule not in _predefined_rules:
        raise ValueError(
            "Iteration rule not in predefined set. Please use 'log', 'quad', or 'lin' instead."
        )
    else:
        if iter_rule == "log":
            return list(
                more_itertools.filter_map(
                    lambda a: a if math.log2(a) % 1 == 0 else None,
                    range(min_length, max_length + 1),
                )
            )
        if iter_rule == "quad":
            return list(
                more_itertools.filter_map(
                    lambda a: a if math.sqrt(a) % 1 == 0.0 else None,
                    range(min_length, max_length + 1),
                )
            )
        return list(range(min_length, max_length + 1))


if __name__ == "__main__":

    x = parse_json("why_would_i_do_this.json")
    # print(x)
    for k, v in x.items():
        print(k)
        print(type(k))
        # exit()
    exit()

if __name__ == "__main__":
    num_qubits = [1, 2]
    max_lengths = [1, 2]
    rate = 1e-4
    hs_error_gen_classes = "hs"
    ca_error_gen_classes = "ca"
    summary_output_of_horror = {}

    single_rate_term_dicts = []
    for i, nq in enumerate(num_qubits):
        single_rate_term_dicts.append([])
        for j, wt in enumerate(range(1, nq + 1)):
            single_rate_term_dicts[i].append([])
            elemgen_basis = CompleteElementaryErrorgenBasis(
                Basis.cast("pp", 4),
                QubitSpace(nq),
                max_ham_weight=wt,
                max_other_weight=wt,
            )
            elemgen_labels = elemgen_basis.labels
            for lbl in elemgen_labels:
                term_dict = {lbl: rate}
                single_rate_term_dicts[i][j].append(term_dict)

    for i, nq in enumerate(num_qubits):
        idt_experiment, target_pspec, paulidicts = make_idt_circuits(nq)
        for j, wt in enumerate(range(1, nq + 1)):
            (
                awk_iterator,
                pauli_node_attributes,
                prep_string_attributes,
                measure_string_attributes,
                ca_pauli_node_attributes,
            ) = generate_experiment_design_stuff(nq, max_weight=wt)
            jac_df = jacobian_df(
                hs_error_gen_classes,
                ca_error_gen_classes,
                pauli_node_attributes,
                prep_string_attributes,
                measure_string_attributes,
                ca_pauli_node_attributes,
                awk_iterator,
            )
            for term_dict in single_rate_term_dicts[i][j]:
                noise_model = create_noise_model(term_dict, target_pspec)
                noisy_ds = simulate_noisy_idt(noise_model, idt_experiment, seed=8082024)

                observed_error_rates, obs_error_rates_by_exp = report_observed_rates(
                    nq, noisy_ds, max_lengths, paulidicts
                )
                obs_rats = [v for v in obs_error_rates_by_exp.values()]
                intrinsic_rates = observed_rates_to_intrinsic(jac_df, obs_rats)
                # convert the keys in df.columns to error generator labels.

                intrinsic_rates_dict = dict(zip(jac_df.columns, intrinsic_rates))
                # estimated_rate_diff = [intrinsic_rates_dict for ]
                summary_output_of_horror[str(term_dict)] = intrinsic_rates_dict

    with open("why_would_i_do_this.json", "w") as fo:
        json.dump(summary_output_of_horror, fo)
