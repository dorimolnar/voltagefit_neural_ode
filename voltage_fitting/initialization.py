# We initialize some eigenvalues, which are in the stability region of a certain solver (we use RK1 = Euler) when scaled with the step size
# In the commented part I started to make the code compatible with jax.vmap
import math
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial


def in_stability_region(lambda_val, solver_order, epsilon=1e-3):
    """
    Checks whether a given eigenvalue is inside the RK solver's stability region.
    """
    # Construct the stability polynomial: 1 + λ + λ²/2! + ... + λ^p/p!
    stability_poly = sum([(lambda_val)**k / jax.scipy.special.factorial(k) for k in range(solver_order + 1)])
    return jnp.abs(stability_poly) < (1 - epsilon)

# def in_stability_region(lambda_val, solver_order, epsilon):
#     """
#     Checks whether a given eigenvalue is inside the RK solver's stability region.
#     """
#     # Construct the stability polynomial: 1 + λ + λ²/2! + ... + λ^p/p!
#     ks = jnp.arange(solver_order + 1)
#     terms = (lambda_val ** ks) / jax.scipy.special.factorial(ks)
#     stability_poly = jnp.sum(terms)
#     return jnp.abs(stability_poly) < (1 - epsilon)

def rejection_sample_eigenvalues(key, solver_order, h, u_dim, use_complex=True, epsilon=1e-3):
    """
    Stability region rejection sampling for generating eigenvalues for a given solver.
    Returns u_dim eigenvalues inside the stability region scaled *without* scaling with h.
    """

    eigenvalues = []

    while len(eigenvalues) < u_dim:
        key, subkey1, subkey2 = jax.random.split(key, 3)
        # Sample real part
        real = jax.random.uniform(subkey1, (), minval=-3.0, maxval=-0.1)

        # Decide whether to add imaginary part
        if use_complex and len(eigenvalues) < u_dim - 1:
            imaginary = jax.random.uniform(subkey2, (), minval=-3.0, maxval=3.0)
        else:
            imaginary = 0.0

        lam = real + 1j * imaginary

        if in_stability_region(lam, solver_order, epsilon):
            eigenvalues.append(lam / h)  # Scale back so that h * λ gives Jacobian
            if imaginary != 0:
                eigenvalues.append(jnp.conj(lam / h))

    return jnp.array(eigenvalues)


# @partial(jax.jit, static_argnums=(1,2,3,4))
# def rejection_sample_eigenvalues(key, solver_order, h, u_dim, use_complex=True, epsilon=1e-3, n_candidates=5000):
#     """
#     Vectorized rejection sampling with conjugate pairing.
#     Ensures that if a candidate has nonzero imaginary part, 
#     both λ and conj(λ) are included.
#     """

#     # Sample reals & imaginaries
#     key_r, key_i = jax.random.split(key)
#     reals = jax.random.uniform(key_r, (n_candidates,), minval=-3.0, maxval=-0.1)

#     if use_complex:
#         imags = jax.random.uniform(key_i, (n_candidates,), minval=-3.0, maxval=3.0)
#     else:
#         imags = jnp.zeros(n_candidates)

#     lambdas = reals + 1j * imags

#     # Mask: valid if inside stability region
#     mask = jax.vmap(lambda lam: in_stability_region(lam, solver_order, epsilon))(lambdas)

#     # Scale eigenvalues
#     lambdas = lambdas / h

#     # For each λ, create [λ, conj(λ)] if imag != 0, else just [λ, nan]
#     def make_pair(lam):
#         conj_lam = jnp.conj(lam)
#         # If imaginary part is tiny, just keep lam
#         is_real = jnp.isclose(jnp.imag(lam), 0.0)
#         return jnp.where(is_real, jnp.array([lam, jnp.nan]), jnp.array([lam, conj_lam]))

#     paired = jax.vmap(make_pair)(lambdas)  # shape (n_candidates, 2)

#     # Flatten to 1D
#     paired = paired.reshape(-1)

#     # Keep only valid ones (mask must be doubled to match pairs)
#     mask2 = jnp.repeat(mask, 2)
#     valid = jnp.where(mask2, paired, jnp.nan)

#     # Take first u_dim non-nan values
#     def take_first_n(vals, n):
#         is_nan = jnp.isnan(vals)
#         sort_key = jnp.where(is_nan, jnp.inf, jnp.arange(vals.shape[0]))
#         idx = jnp.argsort(sort_key)[:n]
#         return vals[idx]

#     eigs = take_first_n(valid, u_dim)
#     return eigs

# What if we sample eigenvalues ON the circle? Answer: nothing special
def sample_eigenvalues_on_circle(h, u_dim, use_complex=True, seed=None):
    """
    Samples u_dim eigenvalues such that h * lambda lies exactly on the unit circle.
    Returns eigenvalues (i.e., lambda) such that |h * lambda + 1| = 1.
    """
    if seed is not None:
        np.random.seed(seed)

    eigenvalues = []
    while len(eigenvalues) < u_dim:
        angle = np.random.uniform(0, 2 * np.pi)
        lam_h = -1 + 2 * np.exp(1j * angle)  # Point on the unit circle
        lam = lam_h / h            # Rescale to get lambda

        if use_complex and len(eigenvalues) < u_dim - 1:
            eigenvalues.append(lam)
            eigenvalues.append(np.conj(lam))
        else:
            eigenvalues.append(np.real(lam))  # Only real part if no complex allowed

    return np.array(eigenvalues[:u_dim])

def make_lambda_matrix(layer_shape, eigvals, layer_index, total_layers, input_dim=None):
    """
    Construct Λᵢ matrix for a given layer based on desired eigenvalues.

    Args:
        layer_shape: tuple (out_dim, in_dim)
        eigvals: list of real or complex eigenvalues, complex conjugate pairs follow each other!
        layer_index: int, 0-indexed position in network
        total_layers: total number of layers
        input_dim: used only for Λ₁ when inputs include x state vector (du + dx)

    Returns:
        Λᵢ matrix of shape (out_dim, in_dim)
    """
    in_dim, out_dim = layer_shape
    n_root = 1.0 / total_layers
    blocks = []
    
    i = 0
    while i < len(eigvals):
        lam = eigvals[i]
        lam_root = lam ** n_root

        if jnp.abs(jnp.imag(lam)) > 1e-6:
            # Create one block for the conjugate pair
            real, imag = jnp.real(lam_root), jnp.imag(lam_root)
            block = jnp.array([[real, imag], [-imag, real]])
            blocks.append(block)
            i += 2  # Skip the conjugate pair

        #what happens here??
        else:
            block = jnp.array([[lam_root]])
            blocks.append(block)
            i += 1

    # Flatten blocks into one matrix
    block_matrix = jax.scipy.linalg.block_diag(*blocks)

    # Pad with zeros if needed to match out_dim × in_dim
    padded = jnp.zeros((in_dim, out_dim))
    h, w = block_matrix.shape
    padded = padded.at[:h, :w].set(block_matrix)

    # If this is Λ₁ and input_dim > len(eigvals), add small random input connection
    if layer_index == 0 and input_dim is not None and input_dim > h:
        # Use ν_k ∼ U(-ksi, ksi) where ksi = avg(|Re(λ**1/n)|)
        ksi = jnp.mean(jnp.abs(jnp.real(eigvals**n_root)))
        input_conn = jax.random.uniform(jax.random.PRNGKey(42), (input_dim - h, w), minval=-ksi, maxval=ksi)
        padded = padded.at[h:, :w].set(input_conn)

    return padded.T

def generate_all_lambda_matrices(layer_shapes, eigvals, total_layers):
    """
    Generates Lambdaᵢ matrices using the given eigenvalues.
    
    Args:
        layer_shapes: List of (in_dim, out_dim) tuples for each layer.
        eigvals: list of real or complex eigenvalues, complex conjugate pairs follow each other!
        total_layers: total number of layers
    
    Returns:
        A list of Lambdaᵢ matrices, each of shape (out_dim, in_dim).
    """
    lambda_matrices = []

    for i, shape in enumerate(layer_shapes):
        lam_i = make_lambda_matrix(
            layer_shape=shape,
            eigvals=eigvals,
            layer_index=i,
            total_layers=total_layers,
            input_dim=shape[0] if i == 0 else None  # for Λ₁ only
        )
        lambda_matrices.append(lam_i)
    return lambda_matrices

def sample_pi_matrix(key, dim):
    """Generates a random orthogonal matrix from the Haar distribution."""
    # Sample a random matrix with i.i.d. standard normal entries
    normal_matrix = jax.random.normal(key, shape=(dim, dim))
    
    # QR decomposition
    q, r = jnp.linalg.qr(normal_matrix)

    # Adjust sign to ensure uniformity (Mezzadri correction)
    d = jnp.sign(jnp.diag(r))
    q = q * d  # Broadcasted along rows

    return q

def generate_all_pi_matrices(key, layer_shapes):
    """
    Generates Πᵢ matrices using the output dimensions of each layer.
    
    Args:
        key: JAX random key.
        layer_shapes: List of (in_dim, out_dim) tuples for each layer.
    
    Returns:
        A list of Πᵢ matrices, each of shape (d_hi, d_hi).
    """
    hidden_dims = [out_dim for (_, out_dim) in layer_shapes[:-1]]
    keys = jax.random.split(key, num=len(hidden_dims))
    pi_matrices = [sample_pi_matrix(k, dim) for k, dim in zip(keys, hidden_dims)]
    return pi_matrices

def generate_W_matrices(lambda_matrices, pi_matrices):
    """
    Given a list of Lambda matrices and Pi matrices, generate the list of W_i matrices.
    Assumes len(lambda_matrices) = len(pi_matrices) + 1.
    """
    W_matrices = []

    n_layers = len(lambda_matrices)

    # First layer: W₁ = Π₁ Λ₁
    W_1 = pi_matrices[0] @ lambda_matrices[0]
    W_matrices.append(W_1)

    # Middle layers: Wᵢ = Πᵢ Λᵢ Πᵢ₋₁⁻¹
    for i in range(1, n_layers - 1):
        Pi = pi_matrices[i]
        Lambda = lambda_matrices[i]
        Pi_prev = pi_matrices[i - 1]
        W_i = Pi @ Lambda @ Pi_prev.T  # .T since Pi is orthogonal
        W_matrices.append(W_i)

    # Last layer: Wₙ = Λₙ Πₙ₋₁⁻¹
    Lambda_last = lambda_matrices[-1]
    Pi_prev = pi_matrices[-1]
    W_n = Lambda_last @ Pi_prev.T
    W_matrices.append(W_n)

    return W_matrices