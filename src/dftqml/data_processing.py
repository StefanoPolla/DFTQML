import numpy as np
from typing import Tuple

DTYPE = np.float32


# ------------- one-RDM transformations -------------


def expand_one_rdm(one_rdm: np.ndarray) -> np.ndarray:
    """
    Expand one-RDM(s) such that each j-th column contains all correlators involving the j-th site.
    
    The input RDM has shape (L//2 + 1, L), where L is the number of sites. The elements
        `one_rdm[d, i] = <c^\dag_j c_(j+d)>` represent the distance-d correlators at each site j.

    The output has shape (L, L), with the same encoding `one_rdm[d, i] = <c^\dag_j c_(j+d)>`.

    NOTE: this construction (as well as rhe compressed encoding) only makes sense for real symmetric
        one-RDMs, such as the ones from eigenstates of time-reversal-symmetrtic Hamiltonians.

    Args:
        one_rdm (np.ndarray): array of shape (..., L//2 + 1, L), containing the one-RDMs in the
            compressed encoding, with the local index j on the last axis.

    Returns:
        np.ndarray: expanded one-RDM of shape (L, L)
    """
    shape = one_rdm.shape
    L = shape[-1]
    m = L // 2 + 1
    assert shape[-2] == m, "The second-to-last axis must have length L//2 + 1."

    expanded_rdm = np.zeros((*shape[:-2], L, L), dtype=DTYPE)
    expanded_rdm[..., :m, :] = one_rdm

    for d in range(m, L):
        rolled_row = np.roll(one_rdm[..., (L - d), :], shift=-d, axis=-1)
        expanded_rdm[..., d, :] = rolled_row

    return expanded_rdm


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
        [[(s * (i - j)) % L for i in range(L)] for s in [+1, -1] for j in range(L)]
    )

    # Apply symmetry transformations
    augmented_inputs = np.take(local_inputs, symmetry_index_matrix, axis=-1)

    # flatten the selection dimension
    augmented_inputs = np.moveaxis(augmented_inputs, -2, 1)  # Move the selection dim to the front
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
    pass
