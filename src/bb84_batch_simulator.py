"""
Exact batched BB84 quantum key distribution simulator.

This version preserves the photon-by-photon simulation:

1. Alice generates every bit and basis.
2. Eve independently decides whether to intercept every photon.
3. Eve measures and resends intercepted photons.
4. Photon loss is applied to every photon.
5. Bob measures every detected photon.
6. Channel noise may flip every detected bit.
7. Alice and Bob sift their keys.
8. Seven machine-learning features are calculated.

The speed improvement comes from processing many sessions together
with NumPy instead of calling run_bb84() once for every session.
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


def validate_probability(value, name):
    """
    Make sure a probability is between zero and one.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0."
        )


def safe_divide(numerator, denominator):
    """
    Safely divide NumPy arrays.

    A result of zero is returned wherever the denominator is zero.
    """
    numerator = np.asarray(
        numerator,
        dtype=np.float64
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64
    )

    result = np.zeros_like(
        numerator,
        dtype=np.float64
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0
    )

    return result


def calculate_batch_qber_variation(
    error_mask,
    sifted_mask,
    block_size=10
):
    """
    Calculate exact block-based QBER variation for every session.

    This matches the original features.py behavior:

    1. Keep only sifted-key bits in their original order.
    2. Divide each sifted key into blocks.
    3. Calculate QBER for every block.
    4. Calculate population standard deviation of block QBER values.

    Parameters
    ----------
    error_mask:
        Boolean matrix where True means Alice's and Bob's bits differ.

    sifted_mask:
        Boolean matrix where True means the photon belongs to the
        sifted key.

    block_size:
        Number of sifted-key bits in each block.

    Returns
    -------
    np.ndarray
        One QBER-variation value for every simulated session.
    """
    if block_size <= 0:
        raise ValueError(
            "block_size must be greater than zero."
        )

    number_of_sessions, number_of_bits = (
        sifted_mask.shape
    )

    maximum_blocks = int(
        np.ceil(number_of_bits / block_size)
    )

    # The cumulative position of each sifted bit.
    #
    # Example sifted mask:
    # False, True, False, True, True
    #
    # Cumulative positions:
    # 0, 1, 1, 2, 3
    cumulative_sifted_positions = np.cumsum(
        sifted_mask,
        axis=1,
        dtype=np.int32
    )

    valid_rows, valid_columns = np.nonzero(
        sifted_mask
    )

    if valid_rows.size == 0:
        return np.zeros(
            number_of_sessions,
            dtype=np.float32
        )

    sifted_positions = (
        cumulative_sifted_positions[
            valid_rows,
            valid_columns
        ] - 1
    )

    block_numbers = (
        sifted_positions // block_size
    )

    flattened_block_indexes = (
        valid_rows * maximum_blocks
        + block_numbers
    )

    total_block_slots = (
        number_of_sessions * maximum_blocks
    )

    # Count the number of sifted bits in every block.
    block_bit_counts = np.bincount(
        flattened_block_indexes,
        minlength=total_block_slots
    ).reshape(
        number_of_sessions,
        maximum_blocks
    )

    valid_errors = error_mask[
        valid_rows,
        valid_columns
    ].astype(np.int8)

    # Count errors in every block.
    block_error_counts = np.bincount(
        flattened_block_indexes,
        weights=valid_errors,
        minlength=total_block_slots
    ).reshape(
        number_of_sessions,
        maximum_blocks
    )

    block_qbers = np.zeros(
        (
            number_of_sessions,
            maximum_blocks
        ),
        dtype=np.float64
    )

    np.divide(
        block_error_counts,
        block_bit_counts,
        out=block_qbers,
        where=block_bit_counts != 0
    )

    existing_blocks = block_bit_counts > 0

    number_of_blocks = np.count_nonzero(
        existing_blocks,
        axis=1
    )

    qber_sum = np.sum(
        block_qbers,
        axis=1
    )

    qber_squared_sum = np.sum(
        block_qbers * block_qbers,
        axis=1
    )

    average_block_qber = safe_divide(
        qber_sum,
        number_of_blocks
    )

    average_squared_qber = safe_divide(
        qber_squared_sum,
        number_of_blocks
    )

    variance = (
        average_squared_qber
        - average_block_qber**2
    )

    # Prevent tiny negative numbers caused by floating-point rounding.
    variance = np.maximum(
        variance,
        0.0
    )

    variation = np.sqrt(
        variance
    )

    # Original code returns zero when fewer than two blocks exist.
    variation[number_of_blocks < 2] = 0.0

    return variation.astype(
        np.float32
    )


def generate_exact_bb84_batch(
    number_of_sessions,
    number_of_bits=1000,
    attack_ratio=0.50,
    minimum_interception=0.10,
    maximum_interception=1.00,
    minimum_noise=0.00,
    maximum_noise=0.05,
    minimum_photon_loss=0.00,
    maximum_photon_loss=0.10,
    block_size=10,
    seed=None
):
    """
    Generate an exact batch of photon-by-photon BB84 sessions.

    Parameters
    ----------
    number_of_sessions:
        Number of complete BB84 sessions in this batch.

    number_of_bits:
        Number of photons transmitted in every session.

    attack_ratio:
        Probability that a session contains Eve.

    minimum_interception:
        Minimum fraction of photons intercepted in attack sessions.

    maximum_interception:
        Maximum fraction of photons intercepted in attack sessions.

    minimum_noise:
        Minimum channel-noise probability.

    maximum_noise:
        Maximum channel-noise probability.

    minimum_photon_loss:
        Minimum photon-loss probability.

    maximum_photon_loss:
        Maximum photon-loss probability.

    block_size:
        Block size used for QBER variation.

    seed:
        Optional random seed.

    Returns
    -------
    pandas.DataFrame
        Seven AI features, interception probability, and label.
    """
    if number_of_sessions <= 0:
        raise ValueError(
            "number_of_sessions must be greater than zero."
        )

    if number_of_bits <= 0:
        raise ValueError(
            "number_of_bits must be greater than zero."
        )

    validate_probability(
        attack_ratio,
        "attack_ratio"
    )

    validate_probability(
        minimum_interception,
        "minimum_interception"
    )

    validate_probability(
        maximum_interception,
        "maximum_interception"
    )

    validate_probability(
        minimum_noise,
        "minimum_noise"
    )

    validate_probability(
        maximum_noise,
        "maximum_noise"
    )

    validate_probability(
        minimum_photon_loss,
        "minimum_photon_loss"
    )

    validate_probability(
        maximum_photon_loss,
        "maximum_photon_loss"
    )

    if minimum_interception > maximum_interception:
        raise ValueError(
            "minimum_interception cannot be greater "
            "than maximum_interception."
        )

    if minimum_noise > maximum_noise:
        raise ValueError(
            "minimum_noise cannot be greater "
            "than maximum_noise."
        )

    if minimum_photon_loss > maximum_photon_loss:
        raise ValueError(
            "minimum_photon_loss cannot be greater "
            "than maximum_photon_loss."
        )

    rng = np.random.default_rng(seed)

    matrix_shape = (
        number_of_sessions,
        number_of_bits
    )

    # ---------------------------------------------------------
    # Session labels and communication conditions
    # ---------------------------------------------------------

    # 0 = normal communication
    # 1 = Eve attack
    labels = (
        rng.random(number_of_sessions)
        < attack_ratio
    ).astype(np.uint8)

    random_interception_probabilities = rng.uniform(
        minimum_interception,
        maximum_interception,
        size=number_of_sessions
    )

    interception_probabilities = np.where(
        labels == 1,
        random_interception_probabilities,
        0.0
    ).astype(np.float32)

    noise_probabilities = rng.uniform(
        minimum_noise,
        maximum_noise,
        size=number_of_sessions
    ).astype(np.float32)

    photon_loss_probabilities = rng.uniform(
        minimum_photon_loss,
        maximum_photon_loss,
        size=number_of_sessions
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Alice and Bob
    # ---------------------------------------------------------

    alice_bits = rng.integers(
        0,
        2,
        size=matrix_shape,
        dtype=np.int8
    )

    # 0 = plus basis
    # 1 = cross basis
    alice_bases = rng.integers(
        0,
        2,
        size=matrix_shape,
        dtype=np.int8
    )

    bob_bases = rng.integers(
        0,
        2,
        size=matrix_shape,
        dtype=np.int8
    )

    # ---------------------------------------------------------
    # Eve intercept-resend attack
    # ---------------------------------------------------------

    eve_bases = rng.integers(
        0,
        2,
        size=matrix_shape,
        dtype=np.int8
    )

    intercept_mask = (
        rng.random(matrix_shape)
        < interception_probabilities[:, None]
    )

    transmitted_bits = alice_bits.copy()
    transmitted_bases = alice_bases.copy()

    eve_wrong_basis_mask = (
        intercept_mask
        & (eve_bases != alice_bases)
    )

    number_of_wrong_eve_measurements = (
        np.count_nonzero(
            eve_wrong_basis_mask
        )
    )

    if number_of_wrong_eve_measurements > 0:
        transmitted_bits[
            eve_wrong_basis_mask
        ] = rng.integers(
            0,
            2,
            size=number_of_wrong_eve_measurements,
            dtype=np.int8
        )

    # Every intercepted photon is resent using Eve's basis.
    transmitted_bases[
        intercept_mask
    ] = eve_bases[
        intercept_mask
    ]

    # ---------------------------------------------------------
    # Photon loss
    # ---------------------------------------------------------

    photon_loss_mask = (
        rng.random(matrix_shape)
        < photon_loss_probabilities[:, None]
    )

    detected_mask = ~photon_loss_mask

    # ---------------------------------------------------------
    # Bob's measurement
    # ---------------------------------------------------------

    bob_bits = transmitted_bits.copy()

    bob_wrong_basis_mask = (
        detected_mask
        & (bob_bases != transmitted_bases)
    )

    number_of_wrong_bob_measurements = (
        np.count_nonzero(
            bob_wrong_basis_mask
        )
    )

    if number_of_wrong_bob_measurements > 0:
        bob_bits[
            bob_wrong_basis_mask
        ] = rng.integers(
            0,
            2,
            size=number_of_wrong_bob_measurements,
            dtype=np.int8
        )

    # Lost photons use -1.
    bob_bits[
        photon_loss_mask
    ] = -1

    # ---------------------------------------------------------
    # Channel noise
    # ---------------------------------------------------------

    noise_mask = (
        detected_mask
        & (
            rng.random(matrix_shape)
            < noise_probabilities[:, None]
        )
    )

    bob_bits[
        noise_mask
    ] ^= 1

    # ---------------------------------------------------------
    # Sifted keys and error masks
    # ---------------------------------------------------------

    sifted_mask = (
        detected_mask
        & (alice_bases == bob_bases)
    )

    error_mask = (
        sifted_mask
        & (alice_bits != bob_bits)
    )

    sifted_key_lengths = np.count_nonzero(
        sifted_mask,
        axis=1
    )

    total_errors = np.count_nonzero(
        error_mask,
        axis=1
    )

    qber = safe_divide(
        total_errors,
        sifted_key_lengths
    )

    # ---------------------------------------------------------
    # Detection and photon-loss features
    # ---------------------------------------------------------

    detected_photons = np.count_nonzero(
        detected_mask,
        axis=1
    )

    detection_rate = (
        detected_photons
        / float(number_of_bits)
    )

    photon_loss_rate = (
        1.0 - detection_rate
    )

    # ---------------------------------------------------------
    # Plus-basis QBER
    # ---------------------------------------------------------

    plus_basis_mask = (
        sifted_mask
        & (alice_bases == 0)
    )

    plus_basis_total = np.count_nonzero(
        plus_basis_mask,
        axis=1
    )

    plus_basis_errors = np.count_nonzero(
        plus_basis_mask
        & (alice_bits != bob_bits),
        axis=1
    )

    plus_basis_qber = safe_divide(
        plus_basis_errors,
        plus_basis_total
    )

    # ---------------------------------------------------------
    # Cross-basis QBER
    # ---------------------------------------------------------

    cross_basis_mask = (
        sifted_mask
        & (alice_bases == 1)
    )

    cross_basis_total = np.count_nonzero(
        cross_basis_mask,
        axis=1
    )

    cross_basis_errors = np.count_nonzero(
        cross_basis_mask
        & (alice_bits != bob_bits),
        axis=1
    )

    cross_basis_qber = safe_divide(
        cross_basis_errors,
        cross_basis_total
    )

    # ---------------------------------------------------------
    # Exact block-based QBER variation
    # ---------------------------------------------------------

    qber_variation = calculate_batch_qber_variation(
    error_mask=error_mask,
    sifted_mask=sifted_mask,
    block_size=block_size
)

    # ---------------------------------------------------------
    # Build final DataFrame
    # ---------------------------------------------------------

    dataframe = pd.DataFrame(
        {
            "qber": qber.astype(np.float32),
            "noise_level": noise_probabilities,
            "photon_loss_rate":
                photon_loss_rate.astype(np.float32),
            "detection_rate":
                detection_rate.astype(np.float32),
            "plus_basis_qber":
                plus_basis_qber.astype(np.float32),
            "cross_basis_qber":
                cross_basis_qber.astype(np.float32),
            "qber_variation":
                qber_variation,
            "interception_probability":
                interception_probabilities,
            "label": labels
        },
        columns=FEATURE_COLUMNS
    )

    return dataframe


def benchmark_batch_simulator(
    number_of_sessions=100_000,
    number_of_bits=1000,
    batch_size=10000,
    seed=42
):
    """
    Benchmark exact batched BB84 generation.

    Data is generated in multiple batches so memory use remains controlled.
    Nothing is saved to disk during the benchmark.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    print("=" * 65)
    print("EXACT BATCHED BB84 SIMULATOR BENCHMARK")
    print("=" * 65)

    print(
        f"Sessions requested : "
        f"{number_of_sessions:,}"
    )

    print(
        f"Photons per session: "
        f"{number_of_bits:,}"
    )

    print(
        f"Batch size         : "
        f"{batch_size:,}"
    )

    generated_sessions = 0
    batch_number = 0
    dataframes = []

    start_time = time.perf_counter()

    while generated_sessions < number_of_sessions:
        current_batch_size = min(
            batch_size,
            number_of_sessions
            - generated_sessions
        )

        current_seed = (
            seed + batch_number
            if seed is not None
            else None
        )

        batch_dataframe = generate_exact_bb84_batch(
            number_of_sessions=current_batch_size,
            number_of_bits=number_of_bits,
            seed=current_seed
        )

        dataframes.append(
            batch_dataframe
        )

        generated_sessions += current_batch_size
        batch_number += 1

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        current_speed = (
            generated_sessions
            / elapsed_time
        )

        print(
            f"Generated "
            f"{generated_sessions:,}/"
            f"{number_of_sessions:,} "
            f"({current_speed:,.0f} sessions/sec)"
        )

    dataframe = pd.concat(
        dataframes,
        ignore_index=True
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    sessions_per_second = (
        number_of_sessions
        / elapsed_time
    )

    memory_mb = (
        dataframe.memory_usage(
            deep=True
        ).sum()
        / 1024**2
    )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)

    print(
        f"Sessions generated : "
        f"{len(dataframe):,}"
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
        f"{memory_mb:.2f} MB"
    )

    print("\nLabel counts:")
    print(
        dataframe["label"].value_counts()
    )

    print("\nAverage feature values:")
    print(
        dataframe[
            [
                "qber",
                "photon_loss_rate",
                "detection_rate",
                "plus_basis_qber",
                "cross_basis_qber",
                "qber_variation"
            ]
        ].mean()
    )

    print("\nFirst five rows:")
    print(
        dataframe.head()
    )

    print("=" * 65)

    return dataframe


if __name__ == "__main__":
    benchmark_batch_simulator(
        number_of_sessions=100_000,
        number_of_bits=1000,
        batch_size=5000,
        seed=42
    )