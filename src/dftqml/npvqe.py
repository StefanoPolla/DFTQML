import numpy as np

import openfermion as of
import pennylane as qml

TUNNELING = 1  # constant tunneling factor (sets the energy units)


def spaced_particles_state(nsites: int, nelec: int) -> np.ndarray:
    """Returns a CB state populated with evenly spaced electron pairs.
    The first electron pair is in the first site and the last is in the last
    site, all others are spaced as evenly as possible within the remaining
    sites.

    Args:
        nsites (int): number of sites
        nelec (int): total number of electrons (alpha and beta). must be even.

    Returns:
        np.ndarray: array representing the state in the *interleaved* spins
            convention
    """
    if nelec % 2 != 0:
        raise (ValueError('particle number nelec should be even'))
    state = np.zeros(nsites)
    state[0] = 1
    for i in range(1, nelec // 2):
        state[(i * (nsites - 1) // (nelec // 2 - 1))] = 1.
    return np.repeat(state, 2)


# *** Ansatz class ***


class NPFabricAnsatz():
    '''
    Contains all the information necessary to efficiently construct and query
    the NP fabric ansatz for the Fermi-Hubbard chain.
    '''


    def __init__(self, system, potential, interface='auto', diff_method='best'):
        self.system = system
        self.potential = potential
        self.block_hamiltonian = system.block_hamiltonian(potential)
        self.nqubits = 2 * self.system.nsites
        self.dev = qml.device('lightning.qubit', wires=self.nqubits)
        self.init_state = spaced_particles_state(self.system.nsites,
                                                 self.system.nelec)
        self.up_then_down = [of.up_then_down(i, self.nqubits)
                             for i in range(self.nqubits)]
        self.interface = interface
        self.diff_method = diff_method


    def sqrtswap_init_params(self, depth: int) -> np.ndarray:
        '''
        Args:
            depth(int): number of NP fabric layers

        Returns:
            np.ndarray: flat parameter array
        '''
        params = np.zeros(qml.GateFabric.shape(depth, self.nqubits))
        params[:, :, 1] = np.pi / 2
        return np.ravel(params)

    def zero_init_params(self, depth: int) -> np.ndarray:
        '''
        Args:
            depth(int): number of NP fabric layers

        Returns:
            np.ndarray: flat parameter array
        '''
        params = np.zeros(qml.GateFabric.shape(depth, self.nqubits))
        return np.ravel(params)

    def reshape_parameter_vector(self, parameters):
        '''Reshape into a tensor witht the correct structure for the gate
        fabric'''
        return np.reshape(parameters, (-1, (self.nqubits // 2 - 1), 2))

    def circuit(self, parameters):
        qml.GateFabric(self.reshape_parameter_vector(parameters),
                       wires=np.arange(self.nqubits),
                       init_state=self.init_state)
        if self.system.up_then_down:
            qml.Permute(np.arange(self.nqubits), self.up_then_down)

    def state(self, parameters):
        @qml.qnode(self.dev, interface=self.interface, diff_method=self.diff_method)
        def _state(parameters):
            self.circuit(parameters)
            return qml.state()
        return _state(parameters)

    def energy(self, parameters):
        """Return ansatz energy, does not support automatic differentiation.
        The state and Hamiltonian are projected on the conserved symmetry
        subspace ad defined by self.system, making evaluation more efficient
        but removing support for automatic differentiation.
        """
        block_state = self.system.block_projector @ self.state(parameters)
        energy = np.real(
            block_state.conj() @ self.block_hamiltonian @ block_state)
        return energy
