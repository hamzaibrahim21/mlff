"""Per-atom long-range energies must not depend on the neighbor-list format or the atom order.

'ordered_sparse' stores every pair once (i < j); 'sparse' stores it twice. Both must hand each
atom half of each of its pair energies, so the per-atom values (and the total) are identical.
"""
import numpy as np
import numpy.testing as npt
import pytest
import jax.numpy as jnp

from mlff.nn.observable.observable_sparse import DispersionEnergySparse, ElectrostaticEnergySparse

N_ATOMS = 9
N_PAD = 3  # padded edges carry the out-of-range index N_ATOMS, as in jax_md neighbor lists


def _inputs(positions, charges, numbers, ratios, fmt, perm=None):
    """Long-range inputs for all pairs; `perm` relabels atoms (new atom k is old atom perm[k])."""
    if perm is not None:
        positions, charges, numbers, ratios = positions[perm], charges[perm], numbers[perm], ratios[perm]
    n = len(positions)
    i, j = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i] - positions[j], axis=-1)
    pad_idx = np.full(N_PAD, n)
    pad_d = np.ones(N_PAD)
    if fmt == 'ordered_sparse':
        idx_i, idx_j = np.concatenate([i, pad_idx]), np.concatenate([j, pad_idx])
        d_ij = np.concatenate([d, pad_d])
    else:
        idx_i, idx_j = np.concatenate([i, j, pad_idx]), np.concatenate([j, i, pad_idx])
        d_ij = np.concatenate([d, d, pad_d])
    return dict(
        node_mask=jnp.ones(n, dtype=bool),
        idx_i_lr=jnp.asarray(idx_i, dtype=jnp.int32),
        idx_j_lr=jnp.asarray(idx_j, dtype=jnp.int32),
        d_ij_lr=jnp.asarray(d_ij, dtype=jnp.float32),
        partial_charges=jnp.asarray(charges, dtype=jnp.float32),
        hirshfeld_ratios=jnp.asarray(ratios, dtype=jnp.float32),
        atomic_numbers=jnp.asarray(numbers, dtype=jnp.int32),
    )


def _module(kind, fmt):
    if kind == 'electrostatic':
        return ElectrostaticEnergySparse(
            prop_keys=None, partial_charges=None, cutoff_lr=10.0, neighborlist_format=fmt
        ), 'electrostatic_energy'
    return DispersionEnergySparse(
        prop_keys=None, cutoff_lr=1000.0, cutoff_lr_damping=2.0, hirshfeld_ratios=None,
        legacy=True, neighborlist_format=fmt
    ), 'dispersion_energy'


def _per_atom(kind, fmt, system, perm=None):
    module, key = _module(kind, fmt)
    return np.asarray(module.apply({}, _inputs(*system, fmt, perm))[key])


@pytest.fixture
def system():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, 7.0, size=(N_ATOMS, 3))
    charges = rng.normal(0.0, 0.4, size=N_ATOMS)
    numbers = rng.choice([1, 6, 7, 8], size=N_ATOMS)
    ratios = rng.uniform(0.8, 1.1, size=N_ATOMS)
    return positions, charges, numbers, ratios


@pytest.mark.parametrize('kind', ['electrostatic', 'dispersion'])
def test_ordered_sparse_matches_sparse(kind, system):
    ordered = _per_atom(kind, 'ordered_sparse', system)
    sparse = _per_atom(kind, 'sparse', system)

    npt.assert_allclose(ordered, sparse, rtol=1e-5, atol=1e-7)
    npt.assert_allclose(ordered.sum(), sparse.sum(), rtol=1e-5)
    # The highest-indexed atom has no partner with a larger index, but still owns half of its pairs.
    assert abs(ordered[-1]) > 1e-6


@pytest.mark.parametrize('kind', ['electrostatic', 'dispersion'])
def test_ordered_sparse_is_permutation_invariant(kind, system):
    perm = np.random.default_rng(1).permutation(N_ATOMS)

    reference = _per_atom(kind, 'ordered_sparse', system)
    permuted = _per_atom(kind, 'ordered_sparse', system, perm=perm)

    npt.assert_allclose(permuted, reference[perm], rtol=1e-5, atol=1e-7)
