import numpy as np

import openfermion as of
import cirq


TUNNELING = 1  # constant tunneling factor (sets the energy units)


# *** Ansatz parametrized layers ***

def coulomb_layer(par, nsites, coulomb, qubits):
    for i in range(nsites):
        yield cirq.CZPowGate(exponent=par * coulomb).on(
            qubits[i], qubits[nsites + i])


def potential_layer(par, nsites, potential, qubits):
    for i in range(nsites):
        yield cirq.ZPowGate(exponent=potential[i] * par).on(qubits[i])
        yield cirq.ZPowGate(exponent=potential[i] * par).on(qubits[nsites + i])


def tunneling_gates(site_idx, par, nsites, nelec, qubits):
    # Jordan-Wigner sign for PBC
    if site_idx == (nsites - 1):
        # -1 if there an odd number of electrons in the bulk (1 is on the edge)
        jw_sign = (-1)**(nelec // 2 - 1)
    else:
        jw_sign = 1

    yield cirq.ISwapPowGate(exponent=-2 * jw_sign * TUNNELING * par).on(
        qubits[site_idx], qubits[(site_idx + 1) % nsites])
    yield cirq.ISwapPowGate(exponent=-2 * jw_sign * TUNNELING * par).on(
        qubits[site_idx + nsites], qubits[(site_idx + 1) % nsites + nsites])


def tunneling_layer(par_even, par_odd, nsites, nelec, qubits):
    for i in range(0, nsites, 2):  # even links
        yield tunneling_gates(i, par_even, nsites, nelec, qubits)
    for i in range(1, nsites, 2):  # odd links
        yield tunneling_gates(i, par_odd, nsites, nelec, qubits)


def pqc_gen(parameters, nsites, nelec, coulomb, potential, qubits):
    for (par_coulomb,
         par_potential,
         par_t_even,
         par_t_odd) in np.reshape(parameters, (-1, 4)):
        yield coulomb_layer(par_coulomb, nsites, coulomb, qubits)
        yield potential_layer(par_potential, nsites, potential, qubits)
        yield tunneling_layer(par_t_even, par_t_odd, nsites, nelec, qubits)


# *** Gaussian state preparation ***

def gaussian_state_preparation(noninteracting_hamiltonian, nelec, qubits):
    qham = of.get_quadratic_hamiltonian(noninteracting_hamiltonian)
    yield of.circuits.prepare_gaussian_state(
        qubits, qham, occupied_orbitals=[np.arange(nelec // 2)] * 2)


# *** Ansatz class ***

class VariationalHamiltonianAnsatz():
    '''
    Contains all the information necessary to efficiently construct and query
    the VH ansatz for the Fermi-Hubbard chain.

    Computes and stores the gaussian state prep circuit, so it doesn't have to
    be calculated at each step.
    '''

    def __init__(self, system, potential):
        self.system = system
        self.potential = potential

        self.block_hamiltonian = system.block_hamiltonian(potential)

        self.qubits = cirq.LineQubit.range(2 * system.nsites)

        # gaussian state preparation circuit
        hamiltonian = system.hamiltonian(potential)
        qham = of.get_quadratic_hamiltonian(
            hamiltonian, n_qubits=len(self.qubits),
            ignore_incompatible_terms=True)

        self.gaussian_circuit = cirq.Circuit(
            of.circuits.prepare_gaussian_state(
                self.qubits, qham,
                occupied_orbitals=[np.arange(system.nelec // 2)] * 2
            ))

    def adiabatic_init_params(self, depth, time_unit):
        '''
        time_unit is the time each term gets evolved for in one step.
        e.g. the maximum rotation angle for the coulomb term is
        π * coulomb * time_unit

        TODO work on the pi-conventions.

        preliminary study suggestion: pick time_unit = 0.06 / sqrt(depth)
        TODO more research on this.
        '''
        schedule = [[(1 + j) / (1 + depth), 1, 1, 1] for j in np.arange(depth)]
        return np.ravel(schedule) * time_unit

    def circuit(self, parameters):
        return self.gaussian_circuit + cirq.Circuit([
            pqc_gen(parameters, self.system.nsites, self.system.nelec,
                    self.system.coulomb, self.potential, self.qubits)])

    def state(self, parameters):
        return self.circuit(parameters).final_state_vector()

    def energy(self, parameters):
        block_state = self.system.block_projector @ self.state(parameters)
        energy = np.real(
            block_state.conj() @ self.block_hamiltonian @ block_state)
        return energy
