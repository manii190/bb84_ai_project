import numpy as np


def calculate_qber(alice_key, bob_key):
    """
    Calculate the Quantum Bit Error Rate.

    QBER = number of different bits / total sifted-key bits
    """
    if alice_key.size == 0:
        return 0.0

    errors = np.count_nonzero(alice_key != bob_key)

    return errors / alice_key.size


def calculate_basis_qber(
    alice_bits,
    bob_bits,
    alice_bases,
    bob_bases,
    target_basis
):
    """
    Calculate QBER for one basis.

    Basis values:
        0 = plus basis
        1 = cross basis
        -1 = lost photon
    """
    valid_mask = bob_bits != -1

    basis_mask = (
        valid_mask
        & (alice_bases == bob_bases)
        & (alice_bases == target_basis)
    )

    total = np.count_nonzero(basis_mask)

    if total == 0:
        return 0.0

    errors = np.count_nonzero(
        alice_bits[basis_mask] != bob_bits[basis_mask]
    )

    return errors / total


def calculate_qber_variation(
    alice_key,
    bob_key,
    block_size=10
):
    """
    Calculate the population standard deviation of block QBER values.
    """
    key_length = alice_key.size

    if key_length == 0:
        return 0.0

    number_of_full_blocks = key_length // block_size
    remaining_bits = key_length % block_size

    block_qbers = []

    if number_of_full_blocks > 0:
        usable_length = number_of_full_blocks * block_size

        alice_blocks = alice_key[:usable_length].reshape(
            number_of_full_blocks,
            block_size
        )

        bob_blocks = bob_key[:usable_length].reshape(
            number_of_full_blocks,
            block_size
        )

        block_errors = np.count_nonzero(
            alice_blocks != bob_blocks,
            axis=1
        )

        full_block_qbers = block_errors / block_size

        block_qbers.extend(full_block_qbers.tolist())

    if remaining_bits > 0:
        alice_last_block = alice_key[
            number_of_full_blocks * block_size:
        ]

        bob_last_block = bob_key[
            number_of_full_blocks * block_size:
        ]

        last_block_qber = np.count_nonzero(
            alice_last_block != bob_last_block
        ) / remaining_bits

        block_qbers.append(last_block_qber)

    if len(block_qbers) < 2:
        return 0.0

    return float(np.std(block_qbers, ddof=0))


def extract_features(
    alice_bits,
    bob_bits,
    alice_bases,
    bob_bases,
    alice_key,
    bob_key,
    number_of_bits,
    detected_photons,
    noise_probability
):
    """
    Extract the seven machine-learning features from one BB84 session.
    """
    qber = calculate_qber(
        alice_key,
        bob_key
    )

    photon_loss_rate = (
        number_of_bits - detected_photons
    ) / number_of_bits

    detection_rate = (
        detected_photons / number_of_bits
    )

    plus_basis_qber = calculate_basis_qber(
        alice_bits,
        bob_bits,
        alice_bases,
        bob_bases,
        0
    )

    cross_basis_qber = calculate_basis_qber(
        alice_bits,
        bob_bits,
        alice_bases,
        bob_bases,
        1
    )

    qber_variation = calculate_qber_variation(
        alice_key,
        bob_key
    )

    return {
        "qber": float(qber),
        "noise_level": float(noise_probability),
        "photon_loss_rate": float(photon_loss_rate),
        "detection_rate": float(detection_rate),
        "plus_basis_qber": float(plus_basis_qber),
        "cross_basis_qber": float(cross_basis_qber),
        "qber_variation": float(qber_variation)
    }