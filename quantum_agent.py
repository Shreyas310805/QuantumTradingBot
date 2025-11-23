import pennylane as qml
import torch
import torch.nn as nn

# --- Quantum Circuit Setup ---
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# --- Hybrid Neural Network ---
class QuantumDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QuantumDQN, self).__init__()
        
        self.pre_net = nn.Linear(input_shape, n_qubits)
        
        weight_shapes = {"weights": (3, n_qubits)} 
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.post_net = nn.Linear(n_qubits, n_actions)

    def forward(self, x):
        # 1. Input ko Float banao
        x = x.to(torch.float32)
        
        x = torch.tanh(self.pre_net(x))
        
        # 2. Quantum output ko bhi Float banao
        x = self.q_layer(x)
        x = x.to(torch.float32)
        
        x = self.post_net(x)
        return x