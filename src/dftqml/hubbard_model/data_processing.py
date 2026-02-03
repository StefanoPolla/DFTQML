"""
Data processing functions for DFT-QML
This module contains functions for data augmentation and transformations of one-RDMs and two-RDMs."""
from math import factorial
from typing import Tuple
import itertools
import numpy as np
from numpy.typing import ArrayLike
import torch

DTYPE = np.float32

# ------------- Data augmentation -------------


def augment_by_shift_and_mirror(local_inputs: np.ndarray, invariant_outputs: np.ndarray) -> Tuple:
    """
    Augment the dataset according to translational and mirror symmetries.

    For each datapoint, 2 * L datapoints are created by taking shifted and
    mirrored versions of each density, each with the same energy.

    Args:
        local_inputs (np.ndarray): list of densities or one-RDMs of shape (n_idcs, ..., L),
            where n_idcs is the number of datapoints, and the last index represents locality,
            i.e. a shift in the last index corresponds to a real-space shift.
        invariant_outputs (np.ndarray): corresponding list of energies, invariant under shift.

    Returns:
        Tuple: (augmented_inputs, augmented_outputs), where:
            - augmented_inputs has shape (n_idcs * 2 * L, ..., L)
            - augmented_outputs has shape (n_idcs * 2 * L,)
    """
    shape = local_inputs.shape
    L = shape[-1]

    # Generate index matrix for translational and mirror symmetries
    symmetry_index_matrix = np.array(
        [[(s * (i - j)) % L for i in range(L)]
         for s in [+1, -1] for j in range(L)]
    )

    # Apply symmetry transformations
    augmented_inputs = np.take(local_inputs, symmetry_index_matrix, axis=-1)

    # flatten the selection dimension
    # Move the selection dim to the front
    augmented_inputs = np.moveaxis(augmented_inputs, -2, 1)
    new_shape = (-1,) + shape[1:]  # Combine first two axes
    augmented_inputs = augmented_inputs.reshape(new_shape)

    # Repeat invariant outputs for each transformation
    augmented_outputs = np.repeat(invariant_outputs, 2 * L)
    return augmented_inputs, augmented_outputs


def augment_by_permutations(one_rdms: np.ndarray, rmft_energies: np.ndarray) -> Tuple:
    """
    Augment RDMFT dataset according to permutation symmetry.

    For each datapoint, L! (factorial) datapoints are created by taking all possible permutations of
    local sites one-RDMs, each with the same interaction energy <U>.

    Args:
        one_rdms (np.ndarray): list of one-RDMs of shape (n_idcs, L, L),
            where n_idcs is the number of datapoints, and the last two indices represent locality.
        rmft_energies (np.ndarray): corresponding list of interaction energies, invariant under
            sites permutation.

    Returns:
        Tuple: (augmented_inputs, augmented_outputs), where:
            - augmented_inputs has shape (n_idcs * L!, L, L)
            - augmented_outputs has shape (n_idcs * L!,)
    """
    shape = one_rdms.shape
    L = shape[-1]

    # Expand one-RDMs to simplify the locality structure
    symmetric_rdms = one_rdm_compressed_to_symmetric_matrix(one_rdms)

    # Generate all permutations of the of indices 0, 1, ..., L-1
    perms = list(itertools.permutations(range(L)))
    perms = np.array(perms)  # Shape: (L!, L)

    # Advanced indexing for efficient broadcasting of permutations
    permuted_rdms = symmetric_rdms[..., perms[:,
                                              np.newaxis, :], perms[:, :, np.newaxis]]

    # Flatten the permutation dimension
    augmented_inputs = permuted_rdms.reshape((-1, L, L))

    # Return to compressed encoding
    augmented_inputs = one_rdm_symmetric_matrix_to_compressed(augmented_inputs)

    # Repeat outputs for each transformation (interaction energy is invariant under permutations)
    augmented_outputs = np.repeat(rmft_energies, factorial(L))

    return augmented_inputs, augmented_outputs


# ------------- data shuffling -------------

def shuffle_data(inputs: np.ndarray, outputs: np.ndarray) -> Tuple:
    """
    Shuffle the data along the first axis.

    Args:
        inputs (np.ndarray): input data of shape (n_idcs, ...)
        outputs (np.ndarray): output data of shape (n_idcs,)

    Returns:
        Tuple: shuffled_inputs, shuffled_outputs
    """
    assert inputs.shape[0] == outputs.shape[0], "Inputs and outputs must have the same number of datapoints."
    idxs = np.random.permutation(inputs.shape[0])
    return inputs[idxs], outputs[idxs]


# ------------- one-RDM transformations -------------


def one_rdm_compressed_to_all_correlators(compressed_rdm: np.ndarray) -> np.ndarray:
    """
    Expand one-RDM(s) such that each j-th column contains all correlators involving the j-th site.

    The input RDM has shape (L//2 + 1, L), where L is the number of sites. The elements
        `one_rdm[d, i] = <c^\dag_j c_(j+d)>` represent the distance-d correlators at each site j.

    The output has shape (L, L), with the same encoding `one_rdm[d, i] = <c^\dag_j c_(j+d)>`.

    NOTE: this construction (as well as the compressed encoding) only makes sense for real symmetric
        one-RDMs, such as the ones from eigenstates of time-reversal-symmetrtic Hamiltonians.

    Args:
        compressed_rdm (np.ndarray): array of shape (..., L//2 + 1, L), containing the one-RDMs in the
            compressed encoding, with the local index j on the last axis.

    Returns:
        np.ndarray: expanded one-RDM of shape (L, L)
    """
    shape = compressed_rdm.shape
    L = shape[-1]
    m = L // 2 + 1
    assert shape[-2] == m, "The second-to-last axis must have length L//2 + 1."

    expanded_rdm = np.zeros((*shape[:-2], L, L), dtype=DTYPE)
    expanded_rdm[..., :m, :] = compressed_rdm

    for d in range(m, L):
        rolled_row = np.roll(
            compressed_rdm[..., (L - d), :], shift=-d, axis=-1)
        expanded_rdm[..., d, :] = rolled_row

    return expanded_rdm


def one_rdm_compressed_to_symmetric_matrix(compressed_rdm: np.ndarray) -> np.ndarray:
    """
    Expand one-RDM(s) such that each j-th column contains all correlators involving the j-th site.

    The input RDM has shape (L//2 + 1, L), where L is the number of sites. The elements
        `one_rdm[d, i] = <c^\dag_j c_(j+d)>` represent the distance-d correlators at each site j.

    The output has shape (L, L), with encoding `output[i, j] = <c^\dag_i c_j>`.

    NOTE: this construction (as well as the compressed encoding) only makes sense for real symmetric
        one-RDMs, such as the ones from eigenstates of time-reversal-symmetrtic Hamiltonians.

    Args:
        compressed_rdm (np.ndarray): array of shape (..., L//2 + 1, L), containing the one-RDMs in the
            compressed encoding, with the local index j on the last axis.

    Returns:
        np.ndarray: expanded one-RDM of shape (L, L)
    """
    shape = compressed_rdm.shape
    L = shape[-1]
    expanded_rdm = one_rdm_compressed_to_all_correlators(compressed_rdm)
    symmetric_rdm = np.zeros_like(expanded_rdm)

    for j in range(L):
        rolled_column = np.roll(expanded_rdm[..., j], shift=j, axis=-1)
        symmetric_rdm[..., j] = rolled_column

    return symmetric_rdm


def one_rdm_symmetric_matrix_to_compressed(symmetric_rdm: np.ndarray) -> np.ndarray:
    """
    Compress one-RDM(s) such that the j-th column contains unique correlators involving site j.

    Inverse of `one_rdm_compressed_to_symmetric_matrix`.
    """
    shape = symmetric_rdm.shape
    L = shape[-1]

    expanded_rdm = np.zeros_like(symmetric_rdm)

    for j in range(L):
        rolled_column = np.roll(symmetric_rdm[..., j], shift=-j, axis=-1)
        expanded_rdm[..., j] = rolled_column

    # Keep only the first L//2 + 1 rows
    return expanded_rdm[..., : (L // 2 + 1), :]


# ******** two-RDM transformations********
def hamiltonian_terms_from_two_rdm(two_rdm, n_particles):
    r"""
    Compute the expectation value of the Hamiltonian terms using the two-RDM, with the following
    convention:
    first row: spin-summed density
        <c_{i, up}^{\dagger} c_i + c_{i, down}^{\dagger} c_i>

    second row: spin-summed nearest-neighbor pair correlators
        <c_{i, up}^{\dagger} c_{j, up} + c_{i, down}^{\dagger} c_{j, down}>

    third row - on-site interaction
        c_{i, up}^{\dagger} c_{i, up} c_{i, down}^{\dagger} c_{i, down}

    Args:
        two_rdm (ArrayLike): Two-body RDM, shape (n_sites, n_sites, n_sites, n_sites)
        n_particles (int): Number of particles

    Returns:
        ArrayLike: the Hamiltonian terms, shape (3, nsites)
    """
    # Check type
    is_torch = torch.is_tensor(two_rdm)
    xp = torch if is_torch else np

    # density = diagonal elements
    density = xp.einsum('ppqq->p', two_rdm) / (n_particles - 1)
    # hopping = off-diagonal elements (nearest neighbor)
    one_rdm = xp.einsum('pqrr->pq', two_rdm) / (n_particles - 1)
    if is_torch:
        diag_k1 = xp.diagonal(one_rdm, offset=1)
        last_elem = one_rdm[0, -1].unsqueeze(0)
        hopping = 2 * xp.cat([diag_k1, last_elem])
    else:
        diag_k1 = xp.diag(one_rdm, k=1)
        last_elem = xp.array([one_rdm[0, -1]])
        hopping = 2 * xp.concatenate([diag_k1, last_elem])
    # interaction = on-site terms
    interaction = .5 * xp.einsum('pppp->p', two_rdm)

    # Stack into shape (3, n_sites)
    terms = xp.stack([density, hopping, interaction], axis=0)
    return terms


def compress_two_rdm(two_rdm):
    """Compress a two-RDM by removing redundant elements."""
    L = two_rdm.shape[0]
    compressed = {}
    for p in range(L):
        for q in range(p, L):
            for r in range(L):
                for s in range(p, L):
                    if (p, q) <= (r, s):
                        compressed[(p, q, r, s)] = two_rdm[p, q, r, s]
    return compressed


def decompress_two_rdm(compressed):
    """Decompress a two-RDM by filling in the symmetry-related elements."""
    L = max(max(p, q, r, s) for p, q, r, s in compressed.keys()) + 1
    two_rdm = np.zeros((L, L, L, L))
    for (p, q, r, s), value in compressed.items():
        # Fill all symmetry-related positions
        for (a, b, c, d) in [
            (p, q, r, s),
            (r, s, p, q),
            (q, p, s, r),
            (s, r, q, p)
        ]:
            two_rdm[a, b, c, d] = value
    return two_rdm


def unpack_dict(compressed):
    """Unpack a dictionary of compressed two-RDMs into a full two-RDM."""
    return np.array(compressed.values()), list(compressed.keys())


def pack_dict(compressed_vector, compressed_indices):
    """Pack a vector of compressed two-RDM elements into a dictionary."""
    compressed = {}
    for i, index in enumerate(compressed_indices):
        compressed[index] = compressed_vector[i]
    return compressed
