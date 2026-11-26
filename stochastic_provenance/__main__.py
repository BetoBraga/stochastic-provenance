"""Run the paired reference and its negative control."""
import json

PREFIX = (1, 0, 1)
DRAWS = (
    (0, 15, 6, 12),
    (15, 0, 9, 2),
    (7, 7, 13, 1),
)


def prefill(prefix):
    return tuple(prefix)


def continuation(state, draws, *, perturb=False):
    """Sample explicit integer buckets from a two-token conditional law."""
    generated = []
    for draw in draws:
        if type(draw) is not int or not 0 <= draw < 16:
            raise ValueError("each draw must be an integer in [0, 16)")
        # token 0 gets threshold/16; token 1 gets the remaining mass.
        threshold = (sum(state) + len(state)) % 15 + 1
        if perturb and not generated:
            threshold = (threshold + 7) % 15 + 1
        token = int(draw >= threshold)
        generated.append(token)
        state += (token,)
    return tuple(generated)


def run(prefix, paths, *, shared, perturb=False):
    paths = tuple(paths)
    if not paths:
        return (), 0
    prefix = tuple(prefix)
    if shared:
        state = prefill(prefix)
        return tuple(continuation(state, p, perturb=perturb) for p in paths), 1
    return (
        tuple(continuation(prefill(prefix), p, perturb=perturb) for p in paths),
        len(paths),
    )


def main():
    reference, ref_prefills = run(PREFIX, DRAWS, shared=False)
    shared, shared_prefills = run(PREFIX, DRAWS, shared=True)
    perturbed, _ = run(PREFIX, DRAWS, shared=True, perturb=True)
    equal = reference == shared
    detected = perturbed != reference
    print(json.dumps({
        "scope": "offline synthetic fixture",
        "serial_prefills": ref_prefills,
        "shared_prefills": shared_prefills,
        "pathwise_equal": equal,
        "negative_control_detected": detected,
        "trajectories": shared,
    }, indent=2))
    return 0 if equal and detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
