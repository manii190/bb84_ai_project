"""
High-speed statistical BB84 simulator.

This simulator generates BB84 session features using probability
distributions instead of simulating every photon individually.

It is designed for generating millions of BB84 sessions quickly.
"""

import time
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "qber",
    "noise_level",
    "photon_loss_rate",
    "detection_rate",
    "plus_basis_qber",
    "cross_basis_qber",
    "qber_variation",
    "interception_probability",
    "label",
]


def safe_divide(numerator, denominator):
    """
    Divide two NumPy arrays safely.

    Returns zero where the denominator is zero.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    result = np.zeros_like(numerator, dtype=np.float64)

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )

    return result


def calculate_total_error_probability(
    noise_probability,
    interception_probability,
):
    """
    Calculate the expected BB84 error probability.

    In an intercept-resend attack, Eve introduces an average error
    probability of approximately 25% multiplied by the fraction of
    photons that she intercepts.

    Channel noise may also flip a bit. The combined probability accounts
    for the possibility that two flips restore the original bit.
    """
    eve_error_probability = 0.25 * interception_probability

    total_error_probability = (
        eve_error_probability
        + noise_probability
        - 2.0
        * eve_error_probability
        * noise_probability
    )

    return np.clip(
        total_error_probability,
        0.0,
        1.0,
    )


def estimate_qber_variation(
    error_probability,
    sifted_key_length,
    block_size=10,
):
    """
    Estimate the variation in QBER between blocks of the sifted key.

    This is a statistical approximation of the block-based calculation
    used in the original bit-level simulator.
    """
    effective_block_size = np.minimum(
        block_size,
        np.maximum(sifted_key_length, 1),
    )

    variation = np.sqrt(
        error_probability
        * (1.0 - error_probability)
        / effective_block_size
    )

    number_of_blocks = np.maximum(
        np.ceil(sifted_key_length / block_size),
        1,
    )

    correction = np.sqrt(
        np.maximum(number_of_blocks - 1, 0)
        / number_of_blocks
    )

    variation = variation * correction

    variation = np.where(
        sifted_key_length < 2,
        0.0,
        variation,
    )

    return variation


def generate_fast_bb84_sessions(
    number_of_sessions,
    number_of_bits=1000,
    attack_ratio=0.50,
    minimum_interception=0.10,
    maximum_interception=1.00,
    minimum_noise=0.00,
    maximum_noise=0.05,
    minimum_photon_loss=0.00,
    maximum_photon_loss=0.10,
    seed=None,
):
    """
    Generate many BB84 sessions using vectorized NumPy operations.

    Returns a pandas DataFrame containing the seven ML features,
    attack strength, and label.
    """
    if number_of_sessions <= 0:
        raise ValueError(
            "number_of_sessions must be greater than zero."
        )

    if number_of_bits <= 0:
        raise ValueError(
            "number_of_bits must be greater than zero."
        )

    rng = np.random.default_rng(seed)

    # 0 = Normal communication
    # 1 = Eve attack
    labels = rng.binomial(
        n=1,
        p=attack_ratio,
        size=number_of_sessions,
    ).astype(np.uint8)

    random_interception = rng.uniform(
        minimum_interception,
        maximum_interception,
        size=number_of_sessions,
    )

    interception_probability = np.where(
        labels == 1,
        random_interception,
        0.0,
    )

    noise_probability = rng.uniform(
        minimum_noise,
        maximum_noise,
        size=number_of_sessions,
    )

    requested_photon_loss = rng.uniform(
        minimum_photon_loss,
        maximum_photon_loss,
        size=number_of_sessions,
    )

    # Number of photons successfully received by Bob
    detected_photons = rng.binomial(
        n=number_of_bits,
        p=1.0 - requested_photon_loss,
        size=number_of_sessions,
    )

    detection_rate = (
        detected_photons / float(number_of_bits)
    )

    photon_loss_rate = 1.0 - detection_rate

    # Alice and Bob choose the same basis about 50% of the time
    sifted_key_length = rng.binomial(
        n=detected_photons,
        p=0.50,
    )

    # Split the sifted key between + and x bases
    plus_basis_count = rng.binomial(
        n=sifted_key_length,
        p=0.50,
    )

    cross_basis_count = (
        sifted_key_length - plus_basis_count
    )

    total_error_probability = (
        calculate_total_error_probability(
            noise_probability,
            interception_probability,
        )
    )

    # Generate total errors in the sifted key
    total_errors = rng.binomial(
        n=sifted_key_length,
        p=total_error_probability,
    )

    # Generate errors separately for the two bases
    plus_basis_errors = rng.binomial(
        n=plus_basis_count,
        p=total_error_probability,
    )

    cross_basis_errors = rng.binomial(
        n=cross_basis_count,
        p=total_error_probability,
    )

    qber = safe_divide(
        total_errors,
        sifted_key_length,
    )

    plus_basis_qber = safe_divide(
        plus_basis_errors,
        plus_basis_count,
    )

    cross_basis_qber = safe_divide(
        cross_basis_errors,
        cross_basis_count,
    )

    qber_variation = estimate_qber_variation(
        error_probability=total_error_probability,
        sifted_key_length=sifted_key_length,
        block_size=10,
    )

    dataframe = pd.DataFrame(
        {
            "qber": qber.astype(np.float32),
            "noise_level": noise_probability.astype(
                np.float32
            ),
            "photon_loss_rate": photon_loss_rate.astype(
                np.float32
            ),
            "detection_rate": detection_rate.astype(
                np.float32
            ),
            "plus_basis_qber": plus_basis_qber.astype(
                np.float32
            ),
            "cross_basis_qber": cross_basis_qber.astype(
                np.float32
            ),
            "qber_variation": qber_variation.astype(
                np.float32
            ),
            "interception_probability":
                interception_probability.astype(
                    np.float32
                ),
            "label": labels,
        }
    )

    return dataframe


def benchmark_generator(
    number_of_sessions=1_000_000,
    number_of_bits=1000,
):
    """
    Benchmark only the generation speed.

    This does not save the dataset to disk.
    """
    print("=" * 60)
    print("BB84 FAST SIMULATOR BENCHMARK")
    print("=" * 60)

    start_time = time.perf_counter()

    dataframe = generate_fast_bb84_sessions(
        number_of_sessions=number_of_sessions,
        number_of_bits=number_of_bits,
        seed=42,
    )

    elapsed_time = time.perf_counter() - start_time

    sessions_per_second = (
        number_of_sessions / elapsed_time
    )

    print(
        f"Sessions generated : "
        f"{number_of_sessions:,}"
    )

    print(
        f"Generation time    : "
        f"{elapsed_time:.4f} seconds"
    )

    print(
        f"Generation speed   : "
        f"{sessions_per_second:,.0f} sessions/sec"
    )

    print(
        f"DataFrame memory   : "
        f"{dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    print("\nFirst five rows:")
    print(dataframe.head())

    print("\nLabel counts:")
    print(dataframe["label"].value_counts())

    print("=" * 60)


if __name__ == "__main__":
    benchmark_generator(
        number_of_sessions=1_000_000,
        number_of_bits=1000,
    )