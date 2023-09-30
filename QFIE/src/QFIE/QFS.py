from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    transpile,
    execute
)

import math
from . import fuzzy_partitions as fp

Qregisters = []


def generate_circuit(fuzzy_partitions):
    """Function generating a quantum circuit with width required by QFS"""
    qc = QuantumCircuit()
    for partition in fuzzy_partitions:
        qc.add_register(
            QuantumRegister(
                math.ceil(math.log(partition.len_partition() + 1, 2)),
                name=partition.name,
            )
        )
        Qregisters.append(
            QuantumRegister(
                math.ceil(math.log(partition.len_partition() + 1, 2)),
                name=partition.name,
            )
        )

    return qc


def output_register(qc, output_partition):
    qc.add_register(
        QuantumRegister(output_partition.len_partition(), name=output_partition.name)
    )
    Qregisters.append(
        QuantumRegister(output_partition.len_partition(), name=output_partition.name)
    )
    return qc


def output_single_qubit_register(qc, name):
    qc.add_register(QuantumRegister(1, name=name))
    return qc


def select_qreg_by_name(qc, name):
    """Function returning the quantum register in QC selected by name"""
    for qr in qc.qregs:
        if name == qr.name:
            break
    return qr


def negation_0(qc, qr, bit_string):
    """Function which insert a NOT gate if the bit in the rule is 0"""
    for index in range(len(bit_string)):
        if bit_string[index] == "0":
            qc.x(qr[index])


def merge_subcounts(subcounts, output_partition):
    merged_counts = {}
    full_out_states = []
    state = ["0" for _ in range(len(output_partition.sets))]
    for i in range(len(output_partition.sets)):
        state[-i - 1] = "1"
        key = "".join(bit for bit in state)
        # TODO: reverse key string here in case of Qiskit ordering issues
        full_out_states.append(key)
        merged_counts[key] = 0
        state[-i - 1] = "0"
    for set in output_partition.sets:
        try:
            merged_counts[
                full_out_states[output_partition.sets.index(set)]
            ] = subcounts[set]["1"]
        except KeyError:
            pass
    return merged_counts


def convert_rule(qc, fuzzy_rule, partitions, output_partition):
    """Function which convert a fuzzy rule in the equivalent quantum circuit.
    You can use multiple times convert_rule to concatenate the quantum circuits related to different
    rules."""
    all_partition = partitions.copy()
    all_partition.append(output_partition)
    rule = fp.fuzzy_rules().add_rules(fuzzy_rule, all_partition)
    controls = []
    targs = []
    for index in range(len(rule)):
        if rule[index] == "and" or rule[index] == "then":
            qr = select_qreg_by_name(qc, rule[index - 2])
            negation_0(qc, qr, rule[index - 1])
            for i in range(select_qreg_by_name(qc, rule[index - 2]).size):
                if len(rule[index - 1]) > i:
                    controls.append(select_qreg_by_name(qc, rule[index - 2])[i])
                else:
                    break
        if rule[index] == "then":
            targs.append(
                select_qreg_by_name(qc, output_partition)[int(rule[index + 2][::-1], 2)]
            )

    qc.mcx(controls, targs[0])
    for index in range(len(rule)):
        if rule[index] == "and" or rule[index] == "then":
            qr = select_qreg_by_name(qc, rule[index - 2])
            negation_0(qc, qr, rule[index - 1])



def compute_qc(backend, qc,  qc_label, n_shots, verbose=True, transpilation_info=False):
    f""" Function computing the quantum circuit qc named qc_label on a backend
     
     Args:
          backend: quantum backend to run the quantum circuit.
          qc (QuantumCircuit): quantum circuit to execute;
          qc_label (str): quantum circuit label;
          n_shots (int): Number of shots;
          verbose (Bool): True to see detail of execution;
          transpilation_info (Bool): True to get information about transpiled qc. If true, the transpiled qc will be 
            used for the execution. 
            
     Return:
         A dictionary with qc_label as key and counts as value.
            
          """
    if verbose:
        try:
            backend_name = backend.backend_name
        except:
            backend_name = backend.DEFAULT_CONFIGURATION['backend_name']
        print('Running qc ' + qc_label + ' on ' + backend_name)
    if transpilation_info:
        transpiled_qc = transpile(
            qc, backend, optimization_level=3
        )
        print(
            "transpiled depth qc " + str(qc_label), transpiled_qc.depth()
        )
        print(
            "CNOTs number qc " + str(qc_label),
            transpiled_qc.count_ops()["cx"],
        )
        job = execute(transpiled_qc, backend, shots=n_shots)
    else:
        job = execute(qc, backend, shots=n_shots)
    result = job.result()

    return {qc_label:result.get_counts()}