import numpy as np


def apply_channel_noise(bits, noise_probability, rng):
    """
    Apply channel noise by flipping bits with the given probability.

    Parameters:
        bits (np.ndarray): Bob's measured bits.
        noise_probability (float): Probability that a bit flips.
        rng (np.random.Generator): NumPy random number generator.

    Returns:
        np.ndarray: Noisy bit array.
    """

    noisy_bits = bits.copy()

    valid_mask = noisy_bits != -1

    noise_mask = (
        rng.random(noisy_bits.size) < noise_probability
    ) & valid_mask

    noisy_bits[noise_mask] ^= 1

    return noisy_bits


def apply_photon_loss(bits, bases, loss_probability, rng):
    """
    Simulate photon loss.

    Lost photons are stored as -1 instead of None for speed.

    Parameters:
        bits (np.ndarray)
        bases (np.ndarray)
        loss_probability (float)
        rng (np.random.Generator)

    Returns:
        received_bits, received_bases
    """

    received_bits = bits.copy()
    received_bases = bases.copy()

    loss_mask = (
        rng.random(bits.size) < loss_probability
    )

    received_bits[loss_mask] = -1
    received_bases[loss_mask] = -1

    return received_bits, received_bases