import sys
sys.path.append("../../qiskit") 

from qiskit import QuantumProgram
import Qconfig



qProgram = QuantumProgram()
n = 3
qRegister = qProgram.create_quantum_registers("qRegister", n)
cRegister = qProgram.create_classical_registers("cRegister", n)
qCircuit = qProgram.create_circuit("qCircuit", ["qRegister"], ["cRegister"])


for i in range(n):
    qCircuit.h(qRegister[i])


qCircuit.z(qRegister[0])
qCircuit.cz(qRegister[1], qRegister[2])


for i in range(n):
    qCircuit.h(qRegister[i])



qCircuit.measure(qRegister[0], cRegister[0])
qCircuit.measure(qRegister[1], cRegister[1])
qCircuit.measure(qRegister[2], cRegister[2])

device = 'ibmqx_qasm_simulator' # Backend to execute your program, in this case it is the online simulator
circuits = ["qCircuit"]  # Group of circuits to execute
qProgram.compile(circuits, "local_qasm_simulator") # Compile your program

result = qProgram.run(wait=2, timeout=240)

print(qProgram.get_counts("qCircuit"))
