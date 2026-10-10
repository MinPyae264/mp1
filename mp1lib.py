import numpy as np

def attempts_until_clean(n_trials, q, rng):
    """
    Returns the number of attempts until the first clean packet.
    The geometric distribution support starts at 1 attempt.
    """
    return rng.geometric(p=q, size=n_trials)

def bsc(bits, p, rng):
    """Flip each bit independently with probability p."""
    flips = rng.random(bits.shape) < p
    return np.bitwise_xor(bits, flips.astype(int))

def packet_errors(n_pkt, n_bits, p, rng):
    """Return the number of bit errors in each packet."""
    return rng.binomial(n_bits, p, n_pkt)

def tv_distance(pmf1, pmf2):
    """Return the total variation distance between two pmfs."""
    total_variation_distance = 0.5 * np.sum(np.abs(pmf1 - pmf2))
    return total_variation_distance
