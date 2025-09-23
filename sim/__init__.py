"""Core tuning system generation logic for SIM/THIN.

This module implements the fundamental algorithms for generating various musical
tuning systems and scales, providing the mathematical foundation for both
SIM and THIN applications.

Supported Tuning Systems:
- Equal Temperament (ET): n-TET systems with configurable divisions per octave
- Geometric Progressions: Custom generator ratios with specified step counts
- Natural Intonation: 4:5:6 harmonic ratios and extensions
- Danielou Microtonal System: Traditional Indian microtonal intervals
- Pythagorean Tuning: Perfect fifth-based system with optional wolf intervals
- Just Intonation: Pure harmonic ratios and limit-based systems

Key Features:
- Exact rational arithmetic using fractions.Fraction for mathematical precision
- Flexible ratio reduction to octave intervals [1, 2) unless --no-reduce specified
- Support for microtonal base notes with quarter/eighth-tone alterations
- Octave transposition and custom frequency references
- Integration with Csound cpstun opcode format

Parsing and Validation:
- Intelligent interval parsing (integers as cents, fractions/floats as ratios)
- Comprehensive argument validation with multilingual error messages
- Safe mathematical operations with overflow protection
- Graceful handling of edge cases and degenerate scales

Mathematical Algorithms:
- Ratio normalization and simplification
- Geometric series generation with configurable bounds
- Harmonic series calculation with customizable limits
- Scale step sorting and deduplication

This module serves as the computational core that other modules build upon
for file export, audio analysis, and user interface functionality.
"""

import argparse
import functools
import math
from fractions import Fraction
from typing import List, Optional, Tuple, Union

import consts
import utils


def parse_interval_value(value: str) -> Union[Fraction, float]:
    """Parsa un valore di intervallo per --geometric.

    Regola: un intero senza suffisso è interpretato come cents; frazioni e float sono rapporti.
    Esempi: 700 -> cents; 3/2 -> rapporto; 2.0 -> rapporto.
    """
    if value is None:
        return Fraction(2, 1)

    s = str(value).strip()

    # 1) Prova int o frazione (int => cents; Fraction => rapporto)
    try:
        parsed_val = utils.int_or_fraction(s)
        if isinstance(parsed_val, int):
            ratio = utils.cents_to_fraction(float(parsed_val))
            if float(ratio) <= 1.0:
                raise argparse.ArgumentTypeError("L'intervallo in cents deve essere > 0")
            return ratio
        else:
            # è una Fraction -> rapporto
            ratio = parsed_val
            if float(ratio) <= 1.0:
                raise argparse.ArgumentTypeError("L'intervallo (rapporto) deve essere > 1")
            return ratio
    except argparse.ArgumentTypeError:
        # Continue to try float parsing
        pass

    # 2) Float come rapporto
    try:
        f = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Intervallo non valido: '{value}'")
    if not math.isfinite(f) or f <= 1.0:
        raise argparse.ArgumentTypeError("L'intervallo (rapporto) deve essere > 1")
    return f


def parse_danielou_tuple(value: str) -> Tuple[int, int, int]:
    """Parser per terne Danielou 'a,b,c'."""
    if value is None:
        return 0, 0, 1

    s = str(value).strip()

    # Remove outer parentheses
    if s and s[0] in '[({' and s[-1] in '])}':
        s = s[1:-1].strip()

    # Normalizza separatori
    for sep in (':', ';'):
        s = s.replace(sep, ',')

    # Split e pulizia
    s = s.strip(',')
    parts = [p.strip() for p in s.split(',') if p.strip()]

    # Caso singolo intero
    if len(parts) == 1:
        try:
            return int(parts[0]), 0, 1
        except ValueError:
            raise argparse.ArgumentTypeError("Formato non valido per --danielou")

    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Formato non valido per --danielou. Usa 'a,b,c'")

    try:
        a = int(parts[0])
        b = int(parts[1])
        c = int(parts[2])
        return a, b, c
    except ValueError:
        raise argparse.ArgumentTypeError("Esponenti non validi per --danielou")


@functools.lru_cache(maxsize=32)
def build_natural_ratios(a_max: int, b_max: int, reduce_octave: bool = True) -> List[float]:
    """Generates ratios for the natural system (4:5:6). Cached for performance."""
    three_over_two = Fraction(3, 2)
    five_over_four = Fraction(5, 4)
    vals = []

    for a in range(-abs(a_max), abs(a_max) + 1):
        for b in range(-abs(b_max), abs(b_max) + 1):
            r = utils.pow_fraction(three_over_two, a) * utils.pow_fraction(five_over_four, b)
            vals.append(r)

    return utils.normalize_ratios(vals, reduce_octave=reduce_octave)


@functools.lru_cache(maxsize=16)
def build_danielou_ratios(full_grid: bool = False, reduce_octave: bool = True) -> List[float]:
    """Generates ratios for the Danielou system. Cached for performance."""
    six_over_five = Fraction(6, 5)
    three_over_two = Fraction(3, 2)
    vals = []

    if full_grid:
        # Mapping based on Danielou's notes
        series_spec = {
            0: (5, 5),  # Prima serie su 1/1
            1: (3, 4),  # Seconda serie su 6/5
            -1: (4, 3),  # Terza serie su 5/3
            2: (4, 4),  # Quarta serie su 36/25
            -2: (4, 4),  # Quinta serie su 25/18
            3: (4, 0),  # Sesta serie su 216/125
            -3: (0, 4),  # Settima serie su 125/108
        }

        for a, (desc_n, asc_n) in series_spec.items():
            # Descending fifths
            for b in range(-desc_n, 0):
                r = utils.pow_fraction(six_over_five, a) * utils.pow_fraction(three_over_two, b)
                vals.append(r)
            # Series center
            vals.append(utils.pow_fraction(six_over_five, a))
            # Ascending fifths
            for b in range(1, asc_n + 1):
                r = utils.pow_fraction(six_over_five, a) * utils.pow_fraction(three_over_two, b)
                vals.append(r)
    else:
        # Demonstrative subset
        vals.append(Fraction(1, 1))
        # Circle (axis) of fifths
        for b in range(-5, 6):
            vals.append(utils.pow_fraction(three_over_two, b))
        # Harmonic minor thirds
        for k in range(1, 4):
            vals.append(utils.pow_fraction(six_over_five, k))
        # Harmonic major sixths
        five_over_three = Fraction(5, 3)
        for k in range(1, 4):
            vals.append(utils.pow_fraction(five_over_three, k))

    normalized = utils.normalize_ratios(vals, reduce_octave=reduce_octave)

    if full_grid:
        # Costruisci lista finale di gradi per griglia completa
        base = list(normalized)
        if not base or abs(base[0] - 1.0) > consts.RATIO_EPS:
            base.append(1.0)
            base = sorted(base)

        base_52_unique = []
        for v in base[:52]:
            if not base_52_unique or abs(v - base_52_unique[-1]) > consts.RATIO_EPS:
                base_52_unique.append(v)

        # Estendi se necessario
        if len(base_52_unique) < 52 and len(base) > len(base_52_unique):
            for v in base[len(base_52_unique):]:
                if all(abs(v - u) > consts.RATIO_EPS for u in base_52_unique):
                    base_52_unique.append(v)
                    if len(base_52_unique) >= 52:
                        break

        # Handle final ratio based on reduce_octave setting
        if reduce_octave:
            # When reducing to octave, don't add 2.0 - all ratios should already be in [1,2)
            normalized = base_52_unique[:52]
        else:
            # When not reducing to octave, add the octave (2.0) as the final ratio
            normalized = base_52_unique[:52] + [2.0]

    return normalized


@functools.lru_cache(maxsize=64)
def danielou_from_exponents(a: int, b: int, c: int, reduce_octave: bool = True) -> List[float]:
    """Calculates a single ratio for the Danielou system. Cached for performance."""
    six_over_five = Fraction(6, 5)
    three_over_two = Fraction(3, 2)
    two = Fraction(2, 1)

    r = (utils.pow_fraction(six_over_five, a) *
         utils.pow_fraction(three_over_two, b) *
         utils.pow_fraction(two, c))

    if reduce_octave:
        r = utils.reduce_to_octave(r)

    return [float(r)]



def process_tuning_system(args: argparse.Namespace, _basenote: float) -> Optional[Tuple[List[float], float]]:
    """Processes the tuning system and returns (ratios, interval) or None."""

    # Natural system
    if args.natural:
        try:
            a_max = int(args.natural[0])
            b_max = int(args.natural[1])
            if a_max < 0 or b_max < 0:
                print("A_MAX and B_MAX must be >= 0")
                return None
            ratios = build_natural_ratios(a_max, b_max, not args.no_reduce)
            return ratios, consts.DEFAULT_OCTAVE
        except (TypeError, ValueError):
            print("Invalid values for --natural")
            return None

    # Danielou system
    if args.danielou_all:
        ratios = build_danielou_ratios(True, not args.no_reduce)
        return ratios, consts.DEFAULT_OCTAVE

    if args.danielou is not None:
        ratios = []
        for (a, b, c) in args.danielou:
            ratios.extend(danielou_from_exponents(a, b, c, not args.no_reduce))
        return ratios, consts.DEFAULT_OCTAVE

    # Geometric system
    if args.geometric:
        parts = list(args.geometric)
        if len(parts) != 3:
            print("Usage: --geometric GEN STEPS INTERVAL")
            return None

        # Parse STEPS
        try:
            steps = int(parts[1])
        except (TypeError, ValueError):
            print("Invalid number of steps")
            return None
        if steps <= 0:
            print("Number of steps must be > 0")
            return None

        # Parse INTERVAL
        try:
            interval_ratio = parse_interval_value(parts[2])
        except argparse.ArgumentTypeError as e:
            print(f"Invalid interval: {e}")
            return None

        # Parse GEN come int o fraction, fallback a float
        try:
            gen_val = utils.int_or_fraction(parts[0])
            gen_ratio = Fraction(gen_val, 1) if isinstance(gen_val, int) else gen_val
        except (argparse.ArgumentTypeError, ValueError) as e:
            try:
                gen_ratio = float(parts[0])
            except ValueError:
                print(f"Geometric parsing error: {e}")
                return None

        if float(gen_ratio) <= 0:
            print("Generator must be > 0")
            return None

        # Use the new utility function for geometric series generation
        ratios = utils.generate_geometric_ratios(
            generator=gen_ratio,
            steps=steps,
            interval=interval_ratio,
            reduce_to_interval=not args.no_reduce
        )

        return ratios, float(interval_ratio)

    # ET (default)
    if args.et:
        index, cents = args.et
        if index <= 0 or (isinstance(cents, (int, float)) and cents <= 0):
            print("Invalid index or cents")
            return None

        if utils.is_fraction_type(cents):
            cents = utils.fraction_to_cents(cents)

        # For standard octave (1200 cents), use the utility function directly
        if cents == 1200:
            ratios = utils.generate_tet_ratios(index)
            interval_factor = 2.0
        else:
            # For non-standard intervals, calculate the interval factor and use geometric ratios
            try:
                interval_factor = math.exp(cents / consts.ELLIS_CONVERSION_FACTOR)
                ratio = interval_factor ** (1.0 / index)
                ratios = utils.generate_geometric_ratios(
                    generator=ratio,
                    steps=index,
                    interval=interval_factor,
                    reduce_to_interval=False  # ET ratios are already in correct range
                )
            except (ZeroDivisionError, ValueError):
                print("Division by zero or invalid value error")
                return None

        return ratios, float(interval_factor)

    return None
