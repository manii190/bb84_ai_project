import numpy as np

from channel import apply_channel_noise, apply_photon_loss
from features import calculate_qber, extract_features


rng = np.random.default_rng()


def generate_random_bits(number_of_bits):
    """
    Generate random bits.

    Values:
        0 = bit 0
        1 = bit 1
    """
    return rng.integers(
        0,
        2,
        size=number_of_bits,
        dtype=np.int8
    )


def generate_random_bases(number_of_bits):
    """
    Generate random BB84 bases.

    Values:
        0 = plus basis
        1 = cross basis
    """
    return rng.integers(
        0,
        2,
        size=number_of_bits,
        dtype=np.int8
    )


def eve_intercept(
    alice_bits,
    alice_bases,
    eve_bases,
    interception_probability
):
    """
    Simulate Eve's intercept-resend attack.
    """
    transmitted_bits = alice_bits.copy()
    transmitted_bases = alice_bases.copy()

    intercept_mask = (
        rng.random(alice_bits.size)
        < interception_probability
    )

    same_basis_mask = (
        intercept_mask
        & (eve_bases == alice_bases)
    )

    different_basis_mask = (
        intercept_mask
        & (eve_bases != alice_bases)
    )

    transmitted_bits[same_basis_mask] = (
        alice_bits[same_basis_mask]
    )

    random_eve_bits = rng.integers(
        0,
        2,
        size=np.count_nonzero(different_basis_mask),
        dtype=np.int8
    )

    transmitted_bits[different_basis_mask] = (
        random_eve_bits
    )

    transmitted_bases[intercept_mask] = (
        eve_bases[intercept_mask]
    )

    return transmitted_bits, transmitted_bases


def bob_measurement(
    received_bits,
    received_bases,
    bob_bases
):
    """
    Simulate Bob measuring the received photons.
    """
    bob_bits = received_bits.copy()

    detected_mask = received_bits != -1

    different_basis_mask = (
        detected_mask
        & (bob_bases != received_bases)
    )

    random_bob_bits = rng.integers(
        0,
        2,
        size=np.count_nonzero(different_basis_mask),
        dtype=np.int8
    )

    bob_bits[different_basis_mask] = random_bob_bits

    return bob_bits


def create_sifted_keys(
    alice_bits,
    bob_bits,
    alice_bases,
    bob_bases
):
    """
    Create Alice and Bob's sifted keys.
    """
    sifted_mask = (
        (bob_bits != -1)
        & (alice_bases == bob_bases)
    )

    alice_key = alice_bits[sifted_mask]
    bob_key = bob_bits[sifted_mask]

    return alice_key, bob_key


def run_bb84(
    number_of_bits=1000,
    eve_present=False,
    interception_probability=1.0,
    noise_probability=0.02,
    photon_loss_probability=0.05,
    show_output=True
):
    """
    Run one exact photon-by-photon BB84 session.
    """
    alice_bits = generate_random_bits(number_of_bits)
    alice_bases = generate_random_bases(number_of_bits)
    bob_bases = generate_random_bases(number_of_bits)

    if eve_present:
        eve_bases = generate_random_bases(number_of_bits)

        transmitted_bits, transmitted_bases = eve_intercept(
            alice_bits,
            alice_bases,
            eve_bases,
            interception_probability
        )

    else:
        transmitted_bits = alice_bits.copy()
        transmitted_bases = alice_bases.copy()
        interception_probability = 0.0

    received_bits, received_bases = apply_photon_loss(
        transmitted_bits,
        transmitted_bases,
        photon_loss_probability,
        rng
    )

    bob_bits = bob_measurement(
        received_bits,
        received_bases,
        bob_bases
    )

    bob_bits = apply_channel_noise(
        bob_bits,
        noise_probability,
        rng
    )

    alice_key, bob_key = create_sifted_keys(
        alice_bits,
        bob_bits,
        alice_bases,
        bob_bases
    )

    detected_photons = np.count_nonzero(
        bob_bits != -1
    )

    qber = calculate_qber(
        alice_key,
        bob_key
    )

    features = extract_features(
        alice_bits=alice_bits,
        bob_bits=bob_bits,
        alice_bases=alice_bases,
        bob_bases=bob_bases,
        alice_key=alice_key,
        bob_key=bob_key,
        number_of_bits=number_of_bits,
        detected_photons=detected_photons,
        noise_probability=noise_probability
    )

    if show_output:
        print("=" * 55)
        print("BB84 QUANTUM KEY DISTRIBUTION SIMULATION")
        print("=" * 55)

        print(f"Eve present        : {eve_present}")
        print(
            f"Eve interception   : "
            f"{interception_probability * 100:.2f}%"
        )
        print(f"Transmitted bits   : {number_of_bits}")
        print(f"Detected photons   : {detected_photons}")
        print(
            f"Noise level        : "
            f"{noise_probability * 100:.2f}%"
        )
        print(
            f"Photon loss        : "
            f"{photon_loss_probability * 100:.2f}%"
        )
        print(f"Sifted key length  : {alice_key.size}")
        print(f"QBER               : {qber * 100:.2f}%")
        print(
            f"Keys match         : "
            f"{np.array_equal(alice_key, bob_key)}"
        )

        print("\nSeven AI features:")

        for name, value in features.items():
            print(f"{name:22}: {value:.4f}")

        print("=" * 55)

    return features


if __name__ == "__main__":
    print("\nNORMAL COMMUNICATION\n")

    run_bb84(
        number_of_bits=1000,
        eve_present=False,
        interception_probability=0.0,
        noise_probability=0.02,
        photon_loss_probability=0.05
    )

    print("\nWEAK EVE ATTACK\n")

    run_bb84(
        number_of_bits=1000,
        eve_present=True,
        interception_probability=0.10,
        noise_probability=0.02,
        photon_loss_probability=0.05
    )

    print("\nMEDIUM EVE ATTACK\n")

    run_bb84(
        number_of_bits=1000,
        eve_present=True,
        interception_probability=0.50,
        noise_probability=0.02,
        photon_loss_probability=0.05
    )

    print("\nSTRONG EVE ATTACK\n")

    run_bb84(
        number_of_bits=1000,
        eve_present=True,
        interception_probability=1.0,
        noise_probability=0.02,
        photon_loss_probability=0.05
    )