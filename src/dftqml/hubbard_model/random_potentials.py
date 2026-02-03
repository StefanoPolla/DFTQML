import numpy as np

N_ATTEMPTS = 100
MU_STD_THRESHOLD = 0.4
W_RANGE = [0.005, 2.5]


def random_potential_nelson(L, rng=np.random.default_rng()):
    """Generate a random potential with uniformly-distributed strength W within W_RANGE.

    Procedure:
      1. Sample W ~ Uniform(W_RANGE).
      2. Sample each site independently from Uniform(-W, +W).
      3. Reject if std(potential) > MU_STD_THRESHOLD.
      4. Shift to zero mean.
    """
    for _ in range(N_ATTEMPTS):
        W = rng.uniform(*W_RANGE)
        potential = rng.uniform(-W, +W, L)
        if np.std(potential) > MU_STD_THRESHOLD:
            continue
        return potential - np.mean(potential)
    else:
        raise RuntimeError(f"after {N_ATTEMPTS} tries no valid potential was extracted")


def random_potential_by_volume(L, rng=np.random.default_rng()):
    """Sample a potential uniformly by volume inside an L-dimensional hypersphere.

    Hypersphere defined so that std(mu) <= MU_STD_THRESHOLD. For a zero-mean vector x,
    std(x) = ||x||/sqrt(L). We therefore cap ||x|| <= MU_STD_THRESHOLD * sqrt(L).

    Algorithm:
      1. Sample direction d ~ Normal(0,1)^L, mean-center then normalize.
      2. Sample u ~ Uniform(0,1). Radius r = R_max * u**(1/L) for volume-uniform distribution.
      3. potential = d * r (already zero mean by construction).
    """
    R_max = MU_STD_THRESHOLD * np.sqrt(L)
    for _ in range(N_ATTEMPTS):
        direction = rng.normal(0, 1, L)
        direction -= np.mean(direction)
        direction /= np.linalg.norm(direction)
        u = rng.uniform(0, 1)
        radius = R_max * (u ** (1.0 / L))
        potential = direction * radius
        # Validate zero mean
        assert np.isclose(np.mean(potential), 0.0), "Mean is not zero after adjustment"
        return potential
    else:
        raise RuntimeError("volume sampling failed unexpectedly")
