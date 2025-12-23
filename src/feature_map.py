# src/feature_map.py
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit import QuantumCircuit

def get_zz_feature_map(num_features, reps=2, entanglement="full"):
    """
    ZZFeatureMap (Qiskit) for quantum kernel / QSVM.
    Maps classical features into a quantum state.
    """
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement)

def get_pauli_feature_map(num_features, paulis=["Z"], reps=1):
    return PauliFeatureMap(feature_dimension=num_features, paulis=paulis, reps=reps)
