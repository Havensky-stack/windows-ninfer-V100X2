"""Process sampling overrides shared by the two local server launchers."""

import math


SAMPLING_LIMITS = {
    'temperature': (0, 2), 'top_p': (0, 1), 'top_k': (0, 2147483647),
    'min_p': (0, 1), 'presence_penalty': (-2, 2), 'frequency_penalty': (-2, 2),
    'seed': (0, 18446744073709551615),
}


def validate_sampling(name: str, text: str) -> None:
    if not text.strip():
        return
    lo, hi = SAMPLING_LIMITS[name]
    try:
        integer = name in ('top_k', 'seed')
        if integer and not text.strip().isdecimal():
            raise ValueError()
        value = int(text) if integer else float(text)
        if (not integer and not math.isfinite(value)) or not lo <= value <= hi:
            raise ValueError()
    except ValueError:
        raise ValueError(f'Invalid sampling override: {name} ({lo}..{hi})') from None
