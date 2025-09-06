#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIN - Sistemi di intonazione musicale
Copyright (c) 2025 Luca Bimbi
Distribuito secondo la licenza MIT - vedi il file LICENSE per i dettagli

Nome programma: THIN - Sistemi intonazione musicale
Versione: PHI
Autore: LUCA BIMBI
Data: 2025-09-06
"""

import argparse
import sys
import math
import re
import os
import shutil
import threading
import time
from fractions import Fraction
from typing import List, Tuple, Optional, Union, Iterable, Any


# Metadati di modulo / Module metadata (IT/EN)
__program_name__ = "THIN"  # Nuovo nome in onore di Walter Branchi / New name in honor of Walter Branchi
__version__ = "PHI"  # Version string literal 'PHI'
__author__ = "LUCA BIMBI"
__date__ = "2025-09-06"  # La data distingue le release / The date distinguishes releases
__license__ = "MIT"  # Vedi file LICENSE / See LICENSE file

# Costanti / Constants (IT/EN)
ELLIS_CONVERSION_FACTOR = 1200 / math.log(2)
RATIO_EPS = 1e-9
DEFAULT_DIAPASON = 440.0
DEFAULT_BASEKEY = 60
DEFAULT_OCTAVE = 2.0
MAX_HARMONIC_HZ = 10000.0
MIN_SUBHARMONIC_HZ = 16.0
PROXIMITY_THRESHOLD_HZ = 17.0
MIDI_MIN = 0
MIDI_MAX = 127
MIDI_A4 = 69
SEMITONES_PER_OCTAVE = 12

# Costanti colonne Excel
CUSTOM_COLUMN = "D"
HARM_COLUMN ="E"
SUB_COLUMN ="G"
TET_COLUMN ="I"

# Localizzazione semplice / Simple localization (IT/EN)
_LANG = "it"

# Pattern per parsing file .csd
PATTERN = re.compile(r"\bf\s*(\d+)\b")

# Tipo per valori numerici (int, float o Fraction)
Numeric = Union[int, float, Fraction]

def L(it_msg: str, en_msg: str) -> str:
    """Restituisce il messaggio nella lingua selezionata / Return message in selected language."""
    return it_msg if _LANG == "it" else en_msg


def _detect_lang_from_argv(argv: Optional[List[str]] = None) -> str:
    """Pre-scan di argv per estrarre --lang prima del parsing argparse."""
    try:
        av = list(argv if argv is not None else sys.argv[1:])
    except Exception:
        av = []
    # Support both "--lang it" and "--lang=it"
    for i, tok in enumerate(av):
        if tok == "--lang" and i + 1 < len(av):
            val = av[i + 1].lower()
            if val in ("it", "en"):
                return val
        elif tok.startswith("--lang="):
            val = tok.split("=", 1)[1].strip().lower()
            if val in ("it", "en"):
                return val
    return "it"


def print_banner() -> None:
    """Stampa sempre le info di programma: nome, versione, data, autore, licenza."""
    # Etichette localizzate
    lbl_ver = L("Versione", "Version")
    lbl_date = L("Rilascio", "Release")
    lbl_auth = L("Autore", "Author")
    lbl_lic = L("Licenza", "License")
    info_line = f"{__program_name__}  |  {lbl_ver}: {__version__}  |  {lbl_date}: {__date__}  |  {lbl_auth}: {__author__}  |  {lbl_lic}: {__license__}"
    print(info_line)
    # Copyright / License line (IT/EN)
    copyright_line = L(
        "Copyright (c) 2025 Luca Bimbi. Distribuito secondo la licenza MIT. \nVedi il file LICENSE per i dettagli.",
        "Copyright (c) 2025 Luca Bimbi. Distributed under the MIT License. See the LICENSE file for details."
    )
    print(copyright_line)

def apply_cents(freq_hz: float, cents: float) -> float:
    """Applica offset in cents a una frequenza / Apply cents offset to a frequency."""
    return float(freq_hz) * (2.0 ** (float(cents) / 1200.0))

def is_fraction_type(value: Any) -> bool:
    """Verifica se value è una frazione."""
    return isinstance(value, Fraction)


def fraction_to_cents(ratio: Fraction) -> int:
    """Converte un rapporto razionale in cents."""
    if ratio.denominator == 0:
        raise ValueError(L("Denominatore zero nella frazione", "Zero denominator in fraction"))
    decimal = math.log(ratio.numerator / ratio.denominator)
    return round(decimal * ELLIS_CONVERSION_FACTOR)


def cents_to_fraction(cents: float) -> Fraction:
    """Converte cents in rapporto razionale approssimato."""
    return Fraction(math.exp(cents / ELLIS_CONVERSION_FACTOR)).limit_denominator(10000)


def reduce_to_octave(value: Numeric) -> Numeric:
    """Riduce un rapporto nell'ambito di un'ottava [1, 2)."""
    if isinstance(value, Fraction):
        two = Fraction(2, 1)
        while value >= two:
            value /= two
        while value < 1:
            value *= two
    else:
        value = float(value)
        while value >= 2.0:
            value /= 2.0
        while value < 1.0:
            value *= 2.0
    return value


def reduce_to_interval(value: Numeric, interval: Numeric) -> Numeric:
    """Riduce un rapporto nell'intervallo [1, interval)."""
    try:
        interval_num = float(interval)
    except (TypeError, ValueError):
        return value

    if not math.isfinite(interval_num) or interval_num <= 1.0:
        return value

    if isinstance(value, Fraction) and isinstance(interval, Fraction):
        one = Fraction(1, 1)
        while value >= interval:
            value /= interval
        while value < one:
            value *= interval
    else:
        v = float(value)
        i = float(interval)
        while v >= i:
            v /= i
        while v < 1.0:
            v *= i
        return v

    return value


def normalize_ratios(ratios: Iterable[Numeric], reduce_octave: bool = True) -> List[float]:
    """
    Normalizza una lista di rapporti:
    - Opzionalmente riduce nell'ottava [1, 2)
    - Elimina duplicati con tolleranza
    - Ordina in modo crescente
    """
    processed = []
    seen = []

    for r in ratios:
        v = reduce_to_octave(r) if reduce_octave else r
        v_float = float(v)

        # Controlla duplicati con tolleranza
        if not any(abs(v_float - s) <= RATIO_EPS for s in seen):
            seen.append(v_float)
            processed.append(v_float)

    return sorted(processed)


def repeat_ratios(ratios: List[float], span: int, interval_factor: float) -> List[float]:
    """Ripete la serie di rapporti su un certo ambitus."""
    try:
        s = int(span)
    except (TypeError, ValueError):
        s = 1

    if s <= 1:
        return list(ratios)

    if not isinstance(interval_factor, (int, float)) or interval_factor <= 0:
        return list(ratios)

    out = []
    for k in range(s):
        factor_k = interval_factor ** k
        out.extend(float(r) * factor_k for r in ratios)

    return out


def pow_fraction(fr: Numeric, k: int) -> Numeric:
    """Eleva una Fraction/float ad un esponente intero."""
    k = int(k)
    return fr ** k if isinstance(fr, Fraction) else float(fr) ** k


def build_natural_ratios(a_max: int, b_max: int, reduce_octave: bool = True) -> List[float]:
    """Genera rapporti del sistema naturale (4:5:6)."""
    three_over_two = Fraction(3, 2)
    five_over_four = Fraction(5, 4)
    vals = []

    for a in range(-abs(a_max), abs(a_max) + 1):
        for b in range(-abs(b_max), abs(b_max) + 1):
            r = pow_fraction(three_over_two, a) * pow_fraction(five_over_four, b)
            vals.append(r)

    return normalize_ratios(vals, reduce_octave=reduce_octave)


def build_danielou_ratios(full_grid: bool = False, reduce_octave: bool = True) -> List[float]:
    """Genera rapporti del sistema Danielou."""
    six_over_five = Fraction(6, 5)
    three_over_two = Fraction(3, 2)
    vals = []

    if full_grid:
        # Mappatura basata sugli appunti di Danielou
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
            # Quinte discendenti
            for b in range(-desc_n, 0):
                r = pow_fraction(six_over_five, a) * pow_fraction(three_over_two, b)
                vals.append(r)
            # Centro serie
            vals.append(pow_fraction(six_over_five, a))
            # Quinte ascendenti
            for b in range(1, asc_n + 1):
                r = pow_fraction(six_over_five, a) * pow_fraction(three_over_two, b)
                vals.append(r)
    else:
        # Sottoinsieme dimostrativo
        vals.append(Fraction(1, 1))
        # Asse delle quinte
        for b in range(-5, 6):
            vals.append(pow_fraction(three_over_two, b))
        # Terze minori armoniche
        for k in range(1, 4):
            vals.append(pow_fraction(six_over_five, k))
        # Seste maggiori armoniche
        five_over_three = Fraction(5, 3)
        for k in range(1, 4):
            vals.append(pow_fraction(five_over_three, k))

    normalized = normalize_ratios(vals, reduce_octave=reduce_octave)

    if full_grid and reduce_octave:
        # Costruisci lista finale di 53 gradi
        base = list(normalized)
        if not base or abs(base[0] - 1.0) > RATIO_EPS:
            base.append(1.0)
            base = sorted(base)

        base_52_unique = []
        for v in base[:52]:
            if not base_52_unique or abs(v - base_52_unique[-1]) > RATIO_EPS:
                base_52_unique.append(v)

        # Estendi se necessario
        if len(base_52_unique) < 52 and len(base) > len(base_52_unique):
            for v in base[len(base_52_unique):]:
                if all(abs(v - u) > RATIO_EPS for u in base_52_unique):
                    base_52_unique.append(v)
                    if len(base_52_unique) >= 52:
                        break

        normalized = base_52_unique[:52] + [2.0]

    return normalized


def danielou_from_exponents(a: int, b: int, c: int, reduce_octave: bool = True) -> List[float]:
    """Calcola un singolo rapporto del sistema Danielou."""
    six_over_five = Fraction(6, 5)
    three_over_two = Fraction(3, 2)
    two = Fraction(2, 1)

    r = (pow_fraction(six_over_five, a) *
         pow_fraction(three_over_two, b) *
         pow_fraction(two, c))

    if reduce_octave:
        r = reduce_to_octave(r)

    return [float(r)]


def ratio_et(index: int, cents: Union[int, Fraction]) -> float:
    """Calcola il rapporto della radice per un temperamento equabile."""
    decimal_number = math.exp(cents / ELLIS_CONVERSION_FACTOR)
    return decimal_number ** (1 / index)


def int_or_fraction(value: str) -> Union[int, Fraction]:
    """Parser per interi o frazioni."""
    try:
        return int(value)
    except ValueError:
        try:
            return Fraction(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"'{value}' non è un intero o frazione valida. / '{value}' is not a valid integer or fraction."
            )


def parse_interval_value(value: str) -> Union[Fraction, float]:
    """Parsa un valore di intervallo per --geometric.

    Regola: un intero senza suffisso è interpretato come cents; frazioni e float sono rapporti.
    Esempi: 700 -> cents; 700c -> cents; 3/2 -> rapporto; 2.0 -> rapporto.
    """
    if value is None:
        return Fraction(2, 1)

    s = str(value).strip().lower()

    # Controlla suffissi cents
    cents_suffixes = ['cents', 'cent', 'c']
    for suffix in cents_suffixes:
        if s.endswith(suffix):
            num_str = s[:-len(suffix)].strip()
            try:
                cents_val = float(num_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Valore cents non valido: '{value}'")
            ratio = cents_to_fraction(cents_val)
            if float(ratio) <= 1.0:
                raise argparse.ArgumentTypeError("L'intervallo in cents deve essere > 0")
            return ratio

    # Prova int o frazione
    try:
        val = int_or_fraction(s)
        if isinstance(val, int):
            # Intero puro => cents
            cents_val = float(val)
            ratio = cents_to_fraction(cents_val)
            if float(ratio) <= 1.0:
                raise argparse.ArgumentTypeError("L'intervallo in cents deve essere > 0")
        else:
            ratio = val
            if float(ratio) <= 1.0:
                raise argparse.ArgumentTypeError("L'intervallo (rapporto) deve essere > 1")
        return ratio
    except argparse.ArgumentTypeError:
        # Prova float come rapporto
        try:
            f = float(s)
            if not math.isfinite(f) or f <= 1.0:
                raise argparse.ArgumentTypeError("L'intervallo (rapporto) deve essere > 1")
            return f
        except ValueError:
            raise argparse.ArgumentTypeError(f"Intervallo non valido: '{value}'")


def parse_danielou_tuple(value: str) -> Tuple[int, int, int]:
    """Parser per terne Danielou 'a,b,c'."""
    if value is None:
        return (0, 0, 1)

    s = str(value).strip()

    # Rimuove parentesi esterne
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
            return (int(parts[0]), 0, 1)
        except ValueError:
            raise argparse.ArgumentTypeError("Formato non valido per --danielou / Invalid format for --danielou")

    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Formato non valido per --danielou. Usa 'a,b,c'")

    try:
        a = int(parts[0])
        b = int(parts[1])
        c = int(parts[2])
        return (a, b, c)
    except ValueError:
        raise argparse.ArgumentTypeError("Esponenti non validi per --danielou")


def note_name_or_frequency(value: str) -> Union[float, str]:
    """Parser per nome nota o frequenza."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return str(value)


def convert_note_name_to_midi(note_name: str) -> int:
    """Converte nome nota in valore MIDI. Supporta # e b/B. / Convert note name to MIDI (supports # and b/B)."""
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    s = note_name.strip()
    if not s:
        raise ValueError(L("Nome nota vuoto", "Empty note name"))

    s_up = s.upper()
    note = s_up[0]
    if note not in note_map:
        raise ValueError(L(f"Nome nota non valido: {note_name}", f"Invalid note name: {note_name}"))

    alteration = 0
    idx = 1

    # gestisci # e b/B / handle # and b
    if idx < len(s_up):
        if s_up[idx] == '#':
            alteration += 1
            idx += 1
        elif s_up[idx] in ('B', 'b'):
            alteration -= 1
            idx += 1

    try:
        octave = int(s_up[idx:])
    except (ValueError, IndexError):
        raise ValueError(L(f"Formato ottava non valido in: {note_name}", 
                           f"Invalid octave format in: {note_name}"))

    midi_value = (octave + 1) * SEMITONES_PER_OCTAVE + note_map[note] + alteration

    if not (MIDI_MIN <= midi_value <= MIDI_MAX):
        raise ValueError(L(f"Nota MIDI fuori range: {midi_value}", 
                           f"MIDI note out of range: {midi_value}"))

    return midi_value


def parse_note_with_microtones(note_name: str) -> Tuple[int, float]:
    """
    Parsing nota con microtoni per basenote e fondamentali.
    Simboli: # (diesis), b/B (bemolle), + (quarto di tono su, +50c), - (quarto giù, -50c),
             ! (ottavo su, +25c), . (ottavo giù, -25c).
    Restituisce (midi_int, cents_offset). / Returns (midi_int, cents_offset).
    """
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    s = note_name.strip()
    if not s:
        raise ValueError(L("Nome nota vuoto", "Empty note name"))
    s_up = s.upper()
    note = s_up[0]
    if note not in note_map:
        raise ValueError(L(f"Nome nota non valida: {note_name}", f"Invalid note name: {note_name}"))
    idx = 1
    semitones = 0
    cents = 0.0

    # Accidenti base / basic accidentals
    while idx < len(s_up) and s_up[idx] in ('#', 'B', 'b', '+', '-', '!', '.'):
        ch = s_up[idx]
        if ch == '#':
            semitones += 1
        elif ch in ('B', 'b'):
            semitones -= 1
        elif ch == '+':
            cents += 50.0
        elif ch == '-':
            cents -= 50.0
        elif ch == '!':
            cents += 25.0
        elif ch == '.':
            cents -= 25.0
        idx += 1

    try:
        octave = int(s_up[idx:])
    except (ValueError, IndexError):
        raise ValueError(L(f"Formato ottava non valido in: {note_name}", 
                           f"Invalid octave format in: {note_name}"))

    midi_value = (octave + 1) * SEMITONES_PER_OCTAVE + note_map[note] + semitones
    if not (MIDI_MIN <= midi_value <= MIDI_MAX):
        raise ValueError(L(f"Nota MIDI fuori range: {midi_value}", 
                           f"MIDI note out of range: {midi_value}"))

    return midi_value, cents


def convert_midi_to_hz(midi_value: int, diapason_hz: float = DEFAULT_DIAPASON) -> float:
    """Converte valore MIDI in frequenza Hz."""
    return diapason_hz * (2 ** ((float(midi_value) - MIDI_A4) / SEMITONES_PER_OCTAVE))


def file_exists(file_path: str) -> bool:
    """Verifica esistenza file."""
    exists = os.path.exists(file_path)
    it_word = "esistente" if exists else "non esistente"
    en_word = "exists" if exists else "not found"
    print(L(f"File {it_word}: {file_path}", f"File {en_word}: {file_path}"))
    return exists


def parse_file(file_name: str) -> int:
    """Trova il massimo numero di tabella 'f N' nel file."""
    max_num = 0
    try:
        with open(file_name, "r") as file:
            for line in file:
                for match in PATTERN.finditer(line):
                    num = int(match.group(1))
                    max_num = max(max_num, num)
    except FileNotFoundError:
        print(L(f"Errore: file non trovato: {file_name}", f"Error: file not found: {file_name}"))
    return max_num


def ensure_midi_fit(ratios: List[float], basekey: int,
                    prefer_truncate: bool) -> Tuple[List[float], int]:
    """Assicura che i rapporti rientrino nel range MIDI."""
    n = len(ratios)

    try:
        bk = int(basekey)
    except (TypeError, ValueError):
        bk = 0

    if n > 128:
        if not prefer_truncate:
            print(L(f"WARNING: numero di passi ({n}) eccede 128. Verranno mantenuti solo i primi 128.",
                     f"WARNING: number of steps ({n}) exceeds 128. Only the first 128 will be kept."))
            prefer_truncate = True
        n = 128
        ratios = ratios[:n]

    if prefer_truncate:
        bk = max(MIDI_MIN, min(MIDI_MAX, bk))
        max_len = 128 - bk
        if len(ratios) > max_len:
            print(L(f"WARNING: serie eccede il limite MIDI. Troncati {len(ratios) - max_len} passi.",
                    f"WARNING: series exceeds the MIDI limit. Truncated {len(ratios) - max_len} steps."))
            ratios = ratios[:max_len]
        return ratios, bk
    else:
        allowed_min = MIDI_MIN
        allowed_max = MIDI_MAX - (n - 1)
        if allowed_max < allowed_min:
            allowed_max = allowed_min

        bk_eff = bk
        if bk < allowed_min or bk > allowed_max:
            bk_eff = max(allowed_min, min(allowed_max, bk))
            print(L(f"WARNING: basekey adattata da {bk} a {bk_eff} per includere tutti i passi.",
                     f"WARNING: basekey adjusted from {bk} to {bk_eff} to include all steps."))

        return ratios, bk_eff


def write_cpstun_table(output_base: str, ratios: List[float], basekey: int,
                       basefrequency: float, interval_value: Optional[float] = None) -> Tuple[int, bool]:
    """Crea o appende una tabella cpstun in un file Csound .csd."""
    csd_path = f"{output_base}.csd"
    skeleton = (
        "<CsoundSynthesizer>\n"
        "<CsOptions>\n\n</CsOptions>\n"
        "<CsInstruments>\n\n</CsInstruments>\n"
        "<CsScore>\n\n</CsScore>\n"
        "</CsoundSynthesizer>\n"
    )

    existed_before = file_exists(csd_path)

    if not existed_before:
        try:
            with open(csd_path, "w") as f:
                f.write(skeleton)
        except IOError as e:
            print(L(f"Errore creazione file CSD: {e}", f"Error creating CSD file: {e}"))
            return 0, existed_before

    try:
        with open(csd_path, "r", encoding="utf-8") as f:
            content = f.read()
    except IOError as e:
        print(L(f"Errore lettura CSD: {e}", f"Error reading CSD: {e}"))
        return 0, existed_before

    fnum = parse_file(csd_path) + 1

    # Ordina i rapporti
    try:
        ratios_sorted = sorted(float(r) for r in ratios)
    except (TypeError, ValueError):
        ratios_sorted = [float(r) for r in ratios]

    # Determina parametri cpstun
    numgrades = len(ratios_sorted)

    if interval_value is not None and isinstance(interval_value, (int, float)):
        interval = 0.0 if float(interval_value) <= 0 else float(interval_value)
    else:
        try:
            rmin = min(ratios_sorted) if ratios_sorted else 1.0
            rmax = max(ratios_sorted) if ratios_sorted else 1.0
            interval = 2.0 if (rmin >= 1.0 - RATIO_EPS and rmax <= 2.0 + RATIO_EPS) else 0.0
        except (TypeError, ValueError):
            interval = 0.0

    # Costruisci lista dati
    data_list = [
        str(numgrades),
        f"{float(interval):.10g}",
        f"{float(basefrequency):.10g}",
        str(int(basekey))
    ]
    data_list.extend(f"{r:.10f}" for r in ratios_sorted)

    size = len(data_list)

    # Costruisci righe
    prefix = f"f {fnum} 0 {size} -2 "
    positions = []
    col = len(prefix)
    for t in data_list:
        positions.append(col)
        col += len(t) + 1

    def build_aligned_comment(label_map: List[Tuple[int, str]]) -> str:
        line = ";"
        base_offset = 1
        for pos_idx, text in label_map:
            target = base_offset + (positions[pos_idx] if 0 <= pos_idx < len(positions) else len(prefix))
            if target < len(line):
                target = len(line) + 1
            line += " " * (target - len(line)) + text
        return line + "\n"

    header_comment = (
            build_aligned_comment([(0, "numgrades"), (2, "basefreq"), (4, "rapporti-di-intonazione .......")]) +
            build_aligned_comment([(1, "interval"), (3, "basekey")]) +
            f"; tabella cpstun generata | basekey={basekey} basefrequency={basefrequency:.6f}Hz\n"
    )
    f_line = prefix + " ".join(data_list) + "\n"

    # Inserisci prima di </CsScore>
    insert_marker = "</CsScore>"
    idx = content.rfind(insert_marker)
    if idx == -1:
        content += f"\n<CsScore>\n{header_comment}{f_line}</CsScore>\n"
    else:
        content = content[:idx] + header_comment + f_line + content[idx:]

    try:
        with open(csd_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(L(f"Tabella cpstun (f {fnum}) salvata in {csd_path}", f"cpstun table (f {fnum}) saved to {csd_path}"))
    except IOError as e:
        print(L(f"Errore scrittura CSD: {e}", f"Error writing CSD: {e}"))

    return fnum, existed_before


def write_tun_file(output_base: str, ratios: List[float], basekey: int,
                   basefrequency: float, tun_integer: bool = False) -> None:
    """Esporta un file .tun (AnaMark TUN) con valori espressi in cents assoluti riferiti a 8.1757989156437073336 Hz.
    Struttura: [Tuning] + 128 righe "note X=Y" (cents assoluti)."""
    tun_path = f"{output_base}.tun"
    lines = [
        "[Tuning]",
        # f"; basekey={basekey} basefrequency={basefrequency:.6f}Hz | valori in cents assoluti riferiti a 8.1757989156437073336 Hz"
    ]

    def tet_freq(offset_semitones: int) -> float:
        return basefrequency * (2.0 ** (offset_semitones / 12.0))

    # Ordina i rapporti per garantire valori crescenti nel segmento custom
    try:
        ratios_sorted = sorted(float(r) for r in ratios)
    except (TypeError, ValueError):
        ratios_sorted = [float(r) for r in ratios]

    def note_freq(n: int) -> float:
        if 0 <= basekey <= 127 and basekey <= n < basekey + len(ratios_sorted):
            idx = n - basekey
            if 0 <= idx < len(ratios_sorted):
                return basefrequency * float(ratios_sorted[idx])
            else:
                return tet_freq(n - basekey)
        else:
            return tet_freq(n - basekey)

    # Riferimento assoluto AnaMark
    f_ref = 8.1757989156437073336

    for n in range(128):
        f = note_freq(n)
        if isinstance(f, (int, float)) and f > 0:
            cents = 1200.0 * math.log2(f / f_ref)
        else:
            cents = 0.0 if n == 0 else 0.0
        if tun_integer:
            val = str(int(round(cents)))
        else:
            val = f"{cents:.2f}"
        lines.append(f"Note {n}={val}")

    try:
        with open(tun_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(L(f"File .tun salvato in {tun_path}", f".tun file saved to {tun_path}"))
    except IOError as e:
        print(L(f"Errore scrittura .tun: {e}", f"Error writing .tun: {e}"))


def export_system_tables(output_base: str, ratios: List[float], basekey: int,
                         basenote_hz: float) -> None:
    """Esporta tabelle del sistema generato."""
    # Calcola e ordina
    computed = [(basenote_hz * float(r), i, basekey + i, float(r))
                for i, r in enumerate(ratios)]
    computed.sort(key=lambda t: t[0])

    # Export testo
    txt_path = f"{output_base}_system.txt"
    headers = ["Step", "MIDI", "Ratio", "Hz"]
    try:
        rows = [[str(i), str(basekey + i), f"{r:.10f}", f"{hz:.6f}"]
                for i, (hz, _, _, r) in enumerate(computed)]

        # Calcola larghezze colonne
        widths = [len(h) for h in headers]
        for row in rows:
            for c, val in enumerate(row):
                widths[c] = max(widths[c], len(val))

        def fmt(vals: List[str]) -> str:
            return "  ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(fmt(headers) + "\n")
            for row in rows:
                f.write(fmt(row) + "\n")
        print(L(f"Esportato: {txt_path}", f"Exported: {txt_path}"))
    except IOError as e:
        print(L(f"Errore scrittura {txt_path}: {e}", f"Write error {txt_path}: {e}"))

    # Export Excel (opzionale)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill

        wb = Workbook()
        ws = wb.active
        ws.title = "System"
        ws.append(headers)

        header_fill = PatternFill(start_color="FFDDDDDD", end_color="FFDDDDDD",
                                  fill_type="solid")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill

        for i, (hz, _, _, r) in enumerate(computed):
            ws.append([i, basekey + i, r, hz])

        xlsx_path = f"{output_base}_system.xlsx"
        wb.save(xlsx_path)
        print(L(f"Esportato: {xlsx_path}", f"Exported: {xlsx_path}"))
    except ImportError:
        print(L("openpyxl non installato: export Excel saltato", "openpyxl not installed: Excel export skipped"))
    except Exception as e:
        print(L(f"Errore export Excel: {e}", f"Excel export error: {e}"))


def analyze_audio(audio_path: str,
                   method: str = "lpc",
                   frame_size: int = 1024,
                   hop_size: int = 512,
                   lpc_order: int = 12,
                   window_type: str = "hamming") -> Optional[dict]:
    """Analizza un file WAV per F0 e formanti usando librosa.
    Ritorna un dict: { 'f0_hz': float|None, 'formants': [(freq_hz, rel_amp_0_1), ...] }
    Se librosa non è disponibile o l'analisi fallisce, ritorna None.
    """
    try:
        import numpy as _np
        import librosa as _lb
        import warnings as _warnings
        try:
            from scipy.signal import find_peaks as _find_peaks  # optional
        except Exception:
            _find_peaks = None
    except Exception as e:
        print(L(f"librosa non disponibile ({e}): installa 'librosa' per l'analisi audio.",
                f"librosa not available ({e}): install 'librosa' for audio analysis."))
        return None

    try:
        # Load mono audio at native sampling rate
        y, sr = _lb.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(L(f"Errore nel caricamento audio: {e}", f"Audio loading error: {e}"))
        return None

    # Prepare window function
    win_type_in = (window_type or "hamming").lower()
    if win_type_in not in ("hamming", "hanning"):
        win_type_in = "hamming"
    win = _lb.filters.get_window("hann" if win_type_in == "hanning" else win_type_in, frame_size, fftbins=True)

    # Pitch estimation using pYIN median over frames; fallback to YIN
    f0_hz: Optional[float] = None
    try:
        fmin = 50.0
        fmax = max(2000.0, sr / 4.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                f0_series, _, _ = _lb.pyin(y, fmin=fmin, fmax=fmax, frame_length=frame_size, hop_length=hop_size, sr=sr)
                f0_vals = _np.asarray([v for v in _np.nan_to_num(f0_series, nan=_np.nan) if _np.isfinite(v) and v > 0])
            except Exception:
                f0_track = _lb.yin(y, fmin=fmin, fmax=fmax, frame_length=frame_size, hop_length=hop_size, sr=sr)
                f0_vals = _np.asarray([v for v in f0_track if _np.isfinite(v) and v > 0])
        if f0_vals.size > 0:
            f0_hz = float(_np.median(f0_vals))
    except Exception:
        f0_hz = None

    formant_freqs: list = []
    formant_amps: list = []

    try:
        if method.lower() == "lpc":
            # Iterate frames and compute LPC roots
            y_pad = _np.pad(y, (frame_size//2, frame_size//2), mode='reflect') if y.size >= frame_size else _np.pad(y, (0, frame_size - y.size), mode='constant')
            collect_f = []
            for start in range(0, max(1, len(y_pad) - frame_size + 1), hop_size):
                frame = y_pad[start:start+frame_size]
                if frame.size < frame_size:
                    break
                w = frame * win
                # Pre-emphasis can help formant detection
                w = _np.append(w[0], w[1:] - 0.97 * w[:-1])
                try:
                    a = _lb.lpc(w, order=max(2, int(lpc_order)))
                except Exception:
                    continue
                try:
                    roots = _np.roots(a)
                except Exception:
                    continue
                for r in roots:
                    if _np.abs(r) < 1.0 - 1e-6:
                        ang = _np.angle(r)
                        if ang > 0:
                            f = float(ang * sr / (2 * _np.pi))
                            if 90.0 <= f <= sr/2 - 100:
                                collect_f.append(f)
            if collect_f:
                # Cluster nearby frequencies (~70 Hz) and count as amplitude proxy
                freqs_sorted = _np.sort(_np.array(collect_f, dtype=float))
                centers = []
                counts = []
                for f in freqs_sorted:
                    if not centers or abs(f - centers[-1]) > 70.0:
                        centers.append(f)
                        counts.append(1)
                    else:
                        centers[-1] = (centers[-1] * counts[-1] + f) / (counts[-1] + 1)
                        counts[-1] += 1
                maxc = max(counts) if counts else 1
                formant_freqs = list(centers)
                formant_amps = [c / maxc for c in counts]
        else:
            # Spectral envelope approximation via STFT average and peak picking
            S = _np.abs(_lb.stft(y, n_fft=frame_size, hop_length=hop_size, window=win))
            if S.size == 0:
                return {"f0_hz": f0_hz, "formants": []}
            mag = _np.mean(S, axis=1)
            # Frequency axis in Hz
            freqs_axis = _np.linspace(0.0, sr/2.0, num=mag.shape[0])
            # Peak picking
            if _find_peaks is not None:
                peaks, _ = _find_peaks(mag, distance=max(1, int((200.0/(sr/2.0))*mag.shape[0])))
                # take top 10 by magnitude
                if peaks.size > 10:
                    idx = _np.argsort(mag[peaks])[-10:]
                    peaks = peaks[idx]
            else:
                # Fallback: take largest bins
                peaks = _np.argpartition(mag, -10)[-10:]
            # Map to Hz and normalize amplitudes
            amps = mag[peaks]
            if amps.size > 0:
                amps = amps / float(amps.max())
            fsel = freqs_axis[peaks]
            keep = (fsel >= 150.0) & (fsel <= sr/2.0 - 100.0)
            formant_freqs = list(map(float, fsel[keep]))
            formant_amps = list(map(float, amps[keep]))
    except Exception as e:
        print(L(f"Errore analisi audio: {e}", f"Audio analysis error: {e}"))
        return None

    # Normalize amplitudes 0..1
    pairs = []
    if formant_freqs:
        maxamp = max(formant_amps) if formant_amps else 1.0
        if maxamp <= 0:
            maxamp = 1.0
        for f, a in zip(formant_freqs, formant_amps):
            pairs.append((float(f), max(0.0, min(1.0, float(a) / maxamp))))
    return {"f0_hz": f0_hz, "formants": pairs}


def export_comparison_tables(output_base: str, ratios: List[float], basekey: int,
                             basenote_hz: float, diapason_hz: float,
                             compare_fund_hz: Optional[float] = None,
                             _tet_align: str = "same",
                             subharm_fund_hz: Optional[float] = None,
                             tet_divisions: int = 12,
                             analysis_result: Optional[dict] = None) -> None:
    """Esporta tabelle di confronto con TET (12/24/48), serie armonica, subarmonica e analisi audio opzionale. / 
    Export comparison with TET (12/24/48), harmonic, subharmonic, and optional audio analysis."""

    def freq_to_note_name(freq: float, a4_hz: float) -> str:
        """Converte frequenza in nome nota 12-TET (solo per labeling). / 
        Convert frequency to 12-TET note name (labeling only)."""
        try:
            if not (freq > 0 and a4_hz > 0):
                return ""
            midi = int(round(MIDI_A4 + SEMITONES_PER_OCTAVE * math.log2(freq / a4_hz)))
        except (ValueError, OverflowError):
            return ""
        midi = max(MIDI_MIN, min(MIDI_MAX, midi))
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = names[midi % SEMITONES_PER_OCTAVE]
        octave = (midi // SEMITONES_PER_OCTAVE) - 1
        return f"{name}{octave}"

    def tet_step_index(freq: float, base_freq: float, divs: int) -> int:
        """Restituisce l'indice del passo TET più vicino da base_freq. / 
        Return nearest TET step index from base_freq."""
        if freq <= 0 or base_freq <= 0:
            return 0
        return int(round(divs * math.log2(freq / base_freq)))

    base_cmp = compare_fund_hz if compare_fund_hz is not None else basenote_hz
    sub_base = subharm_fund_hz if subharm_fund_hz is not None else diapason_hz

    # Prepara dati ordinati
    computed = [(basenote_hz * float(r), i, basekey + i, float(r))
                for i, r in enumerate(ratios)]
    computed.sort(key=lambda t: t[0])

    custom_list = computed
    n_rows = len(custom_list)

    # Serie armonica
    harm_vals = []
    harm_n = 1
    while True:
        val = base_cmp * harm_n
        if val > MAX_HARMONIC_HZ:
            break
        harm_vals.append(val)
        harm_n += 1

    # Serie subarmonica
    sub_desc = []
    m = 1
    while True:
        val = sub_base / m
        if val < MIN_SUBHARMONIC_HZ:
            break
        sub_desc.append(val)
        m += 1
    sub_vals = list(reversed(sub_desc))

    # Filtro subarmoniche
    min_custom = custom_list[0][0] if n_rows > 0 else base_cmp
    cutoff_sub = max(MIN_SUBHARMONIC_HZ, min_custom)
    sub_vals = [v for v in sub_vals if v >= cutoff_sub]

    # Allineamento sequenze
    def align_sequence(seq: List[float], customs: List[float]) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * len(customs)
        p = 0
        for j in range(len(customs)):
            low = customs[j]
            high = customs[j + 1] if j + 1 < len(customs) else float('inf')
            if p < len(seq) and low <= seq[p] < high:
                out[j] = seq[p]
                p += 1
        return out

    custom_hz_list = [c[0] for c in custom_list]
    harm_aligned = align_sequence(harm_vals, custom_hz_list)
    sub_aligned = align_sequence(sub_vals, custom_hz_list)

    # Export testo
    txt_path = f"{output_base}_compare.txt"
    try:
        # Prepara allineamento analisi audio / Prepare audio analysis alignment
        audio_f0_idx = None
        audio_f0_val = None
        audio_formant_map = {}
        if analysis_result is not None:
            f0_val = analysis_result.get('f0_hz') if isinstance(analysis_result, dict) else None
            formants = analysis_result.get('formants') if isinstance(analysis_result, dict) else None
            if f0_val and f0_val > 0 and custom_hz_list:
                # indice più vicino / nearest index
                audio_f0_idx = int(min(range(len(custom_hz_list)), key=lambda k: abs(custom_hz_list[k] - f0_val)))
                audio_f0_val = float(f0_val)
            if formants and custom_hz_list:
                # Escludi formanti sotto f0 / Exclude formants below f0
                _ff_list = formants
                if f0_val and f0_val > 0:
                    _ff_list = [(ff, amp) for (ff, amp) in formants if ff >= f0_val]
                for (ff, amp) in sorted(_ff_list, key=lambda x: x[0]):
                    idx = int(min(range(len(custom_hz_list)), key=lambda k: abs(custom_hz_list[k] - ff)))
                    if idx not in audio_formant_map or amp > audio_formant_map[idx][1]:
                        audio_formant_map[idx] = (float(ff), float(max(0.0, min(1.0, amp))))

        headers = [
            "Step", "MIDI", "Ratio",
            "Custom_Hz", "Harmonic_Hz", "DeltaHz_Harm",
            "Subharm_Hz", "DeltaHz_Sub",
            "TET_Hz", "TET_Note", "DeltaHz_TET",
            "AudioF0_Hz", "AudioFormant_Hz", "Formant_RelAmp"
        ]

        rows = []
        for row_i, (custom_hz, _step_idx, _midi, r) in enumerate(custom_list):
            harm_val = harm_aligned[row_i]
            sub_val = sub_aligned[row_i]

            # N-TET (12/24/48)
            if custom_hz > 0 and base_cmp > 0:
                tet_val = base_cmp * (2.0 ** (tet_step_index(custom_hz, base_cmp, tet_divisions) / tet_divisions))
            else:
                tet_val = base_cmp

            # Formatta valori
            harm_str = f"{harm_val:.6f}" if harm_val is not None else ""
            d_har_str = f"{(custom_hz - harm_val):.6f}" if harm_val is not None else ""
            approx = "≈" if harm_val is not None and abs(custom_hz - harm_val) < PROXIMITY_THRESHOLD_HZ else ""

            sub_str = f"{sub_val:.6f}" if sub_val is not None else ""
            d_sub_str = f"{(custom_hz - sub_val):.6f}" if sub_val is not None else ""

            tet_str = f"{tet_val:.6f}"
            tet_note = freq_to_note_name(tet_val, diapason_hz)
            d_tet_str = f"{(custom_hz - tet_val):.6f}"

            # Audio alignments for this row
            f0_str = f"{audio_f0_val:.6f}" if (audio_f0_idx is not None and audio_f0_idx == row_i and audio_f0_val) else ""
            af = audio_formant_map.get(row_i)
            a_formant_str = f"{af[0]:.6f}" if af else ""
            a_amp_str = f"{af[1]:.3f}" if af else ""

            rows.append([
                str(row_i), str(basekey + row_i), f"{r:.10f}",
                f"{custom_hz:.6f}{approx}",
                f"{harm_str}{approx if harm_str else ''}",
                d_har_str, sub_str, d_sub_str,
                tet_str, tet_note, d_tet_str,
                f0_str, a_formant_str, a_amp_str
            ])

        # Calcola larghezze
        widths = [len(h) for h in headers]
        for row in rows:
            for c, val in enumerate(row):
                widths[c] = max(widths[c], len(val))

        def fmt(vals: List[str]) -> str:
            return "  ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(fmt(headers) + "\n")
            for row in rows:
                f.write(fmt(row) + "\n")
        print(L(f"Esportato: {txt_path}", f"Exported: {txt_path}"))
    except IOError as e:
        print(f"Errore scrittura {txt_path}: {e}")

    # Export Excel (opzionale)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill

        wb = Workbook()
        ws = wb.active
        ws.title = "Compare"

        headers_xl = [
            "Step", "MIDI", "Ratio",
            "Custom_Hz", "Harmonic_Hz", "|DeltaHz_Harm|",
            "Subharm_Hz", "|DeltaHz_Sub|",
            "TET_Hz", "TET_Note", "|DeltaHz_TET|",
            "AudioF0_Hz", "AudioFormant_Hz", "Formant_RelAmp"
        ]
        ws.append(headers_xl)

        header_fill = PatternFill(start_color="FFCCE5FF", end_color="FFCCE5FF",
                                  fill_type="solid")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill

        # Colori serie
        fill_custom = PatternFill(start_color="FFFFCCCC", end_color="FFFFCCCC",
                                  fill_type="solid")
        fill_harm = PatternFill(start_color="FFCCFFCC", end_color="FFCCFFCC",
                                fill_type="solid")
        fill_sub = PatternFill(start_color="FFFFFFCC", end_color="FFFFFFCC",
                               fill_type="solid")
        fill_tet = PatternFill(start_color="FFCCE5FF", end_color="FFCCE5FF",
                               fill_type="solid")

        # Prepara mappa analisi audio per Excel
        audio_f0_idx = None
        audio_formant_map = {}
        if analysis_result is not None:
            f0_val = analysis_result.get('f0_hz') if isinstance(analysis_result, dict) else None
            formants = analysis_result.get('formants') if isinstance(analysis_result, dict) else None
            if f0_val and f0_val > 0 and custom_hz_list:
                audio_f0_idx = int(min(range(len(custom_hz_list)), key=lambda k: abs(custom_hz_list[k] - f0_val)))
            if formants and custom_hz_list:
                # Escludi formanti sotto f0 / Exclude formants below f0
                _ff_list = formants
                if f0_val and f0_val > 0:
                    _ff_list = [(ff, amp) for (ff, amp) in formants if ff >= f0_val]
                for (ff, amp) in sorted(_ff_list, key=lambda x: x[0]):
                    idx = int(min(range(len(custom_hz_list)), key=lambda k: abs(custom_hz_list[k] - ff)))
                    if idx not in audio_formant_map or amp > audio_formant_map[idx][1]:
                        audio_formant_map[idx] = (float(ff), float(max(0.0, min(1.0, amp))))

        for row_i, (custom_hz, _step_idx, _midi, r) in enumerate(custom_list):
            harm_val = harm_aligned[row_i]
            sub_val = sub_aligned[row_i]

            if custom_hz > 0 and base_cmp > 0:
                tet_val = base_cmp * (2.0 ** (tet_step_index(custom_hz, base_cmp, tet_divisions) / tet_divisions))
            else:
                tet_val = base_cmp

            harm_cell = harm_val if harm_val is not None else None
            # d_har_cell = abs(custom_hz - harm_val) if harm_val is not None else None
            d_har_cell = (f"=IF({HARM_COLUMN}{row_i+2}<>\"\","
                          f"ABS({CUSTOM_COLUMN}{row_i+2}-{HARM_COLUMN}{row_i+2}),\"\")")
            sub_cell = sub_val if sub_val is not None else None
            #d_sub_cell = abs(custom_hz - sub_val) if sub_val is not None else None
            d_sub_cell = (f"=IF({SUB_COLUMN}{row_i + 2}<>\"\","
                          f"ABS({CUSTOM_COLUMN}{row_i + 2}-{SUB_COLUMN}{row_i + 2}),\"\")")
            tet_note = freq_to_note_name(tet_val, diapason_hz)
            d_tet_cell = (f"=IF({TET_COLUMN}{row_i + 2}<>\"\","
                          f"ABS({CUSTOM_COLUMN}{row_i + 2}-{TET_COLUMN}{row_i + 2}),\"\")")

            # Valori analisi per questa riga
            f0_cell = custom_hz if (audio_f0_idx is not None and audio_f0_idx == row_i) else None
            a_formant = audio_formant_map.get(row_i)
            a_formant_hz = a_formant[0] if a_formant else None
            a_formant_amp = a_formant[1] if a_formant else None

            ws.append([row_i, basekey + row_i, r, custom_hz, harm_cell, d_har_cell,
                       sub_cell, d_sub_cell, tet_val, tet_note, d_tet_cell,
                       f0_cell, a_formant_hz, a_formant_amp])

            row = ws.max_row
            ws.cell(row=row, column=4).fill = fill_custom
            ws.cell(row=row, column=5).fill = fill_harm
            ws.cell(row=row, column=7).fill = fill_sub
            ws.cell(row=row, column=9).fill = fill_tet
            ws.cell(row=row, column=10).fill = fill_tet

        xlsx_path = f"{output_base}_compare.xlsx"
        wb.save(xlsx_path)
        print(L(f"Esportato: {xlsx_path}", f"Exported: {xlsx_path}"))
    except ImportError:
        print(L("openpyxl non installato: export Excel saltato", "openpyxl not installed: Excel export skipped"))
    except Exception as e:
        print(L(f"Errore export Excel: {e}", f"Excel export error: {e}"))


def page_lines(lines: List[str], rows_per_page: Optional[int] = None) -> None:
    """Simple pager: prints lines with a --More-- prompt, Enter continues, q quits.
    Caps the page height at 80 rows by requirement.
    """
    # Determine terminal rows
    if rows_per_page is None:
        try:
            rows = shutil.get_terminal_size(fallback=(80, 24)).lines - 1
        except (OSError, AttributeError, ValueError):
            rows = 23
    else:
        rows = int(rows_per_page)

    # Cap at 80 rows and enforce minimum
    rows = max(5, min(80, rows))

    i = 0
    n = len(lines)
    while i < n:
        # print up to rows lines
        end = min(n, i + rows)
        for j in range(i, end):
            print(lines[j])
        i = end
        if i < n:
            prompt = L("--More-- (invio=continua, q=esci)", "--More-- (enter=continue, q=quit)")
            print(prompt, end="", flush=True)
            # Wait for Enter to continue or 'q' to quit
            ch = "\n"
            while True:
                try:
                    ch = sys.stdin.read(1)
                except (OSError, ValueError, AttributeError):
                    try:
                        ch = input()
                    except EOFError:
                        ch = "q"
                    ch = ch[0] if ch else "\n"
                if ch.lower() == 'q':
                    break
                if ch in ('\n', '\r'):
                    break
            # clear prompt line (assume 80 cols)
            print("\r" + " " * 80 + "\r", end="")
            if ch.lower() == 'q':
                break


def print_step_hz_table(ratios: List[float], basenote_hz: float) -> None:
    """Stampa tabella multi-colonna con Step/Hz."""
    pairs = [(str(i + 1), f"{basenote_hz * float(r):.2f}")
             for i, r in enumerate(ratios)]

    if not pairs:
        page_lines(["Step Hz"])
        return

    # Calcola larghezze
    step_w = max(len("Step"), max(len(p[0]) for p in pairs))
    hz_w = max(len("Hz"), max(len(p[1]) for p in pairs))
    cell_w = step_w + 1 + hz_w
    gap = 3

    # Larghezza terminale
    try:
        term_w = shutil.get_terminal_size(fallback=(80, 24)).columns
    except (AttributeError, ValueError):
        term_w = 80

    # Calcola colonne
    cols = max(1, (term_w + gap) // (cell_w + gap)) if cell_w < term_w else 1
    rows_count = math.ceil(len(pairs) / cols)

    # Compose lines instead of printing directly
    lines_out: List[str] = []
    header_cell = "Step".ljust(step_w) + " " + "Hz".ljust(hz_w)
    lines_out.append((" " * gap).join([header_cell] * cols))

    for r in range(rows_count):
        line_cells = []
        for c in range(cols):
            idx = c * rows_count + r
            if idx < len(pairs):
                s, h = pairs[idx]
                cell = s.rjust(step_w) + " " + h.rjust(hz_w)
                line_cells.append(cell)
        if line_cells:
            lines_out.append((" " * gap).join(line_cells))

    page_lines(lines_out)


def process_tuning_system(args: argparse.Namespace, _basenote: float) -> Optional[Tuple[List[float], float]]:
    """Processa il sistema di intonazione e ritorna (ratios, interval) o None."""

    # Sistema naturale
    if args.natural:
        try:
            a_max = int(args.natural[0])
            b_max = int(args.natural[1])
            if a_max < 0 or b_max < 0:
                print(L("A_MAX e B_MAX devono essere >= 0", "A_MAX and B_MAX must be >= 0"))
                return None
            ratios = build_natural_ratios(a_max, b_max, not args.no_reduce)
            return ratios, DEFAULT_OCTAVE
        except (TypeError, ValueError):
            print("Valori non validi per --natural")
            return None

    # Sistema Danielou
    if args.danielou_all:
        ratios = build_danielou_ratios(True, not args.no_reduce)
        return ratios, DEFAULT_OCTAVE

    if args.danielou is not None:
        ratios = []
        for (a, b, c) in args.danielou:
            ratios.extend(danielou_from_exponents(a, b, c, not args.no_reduce))
        return ratios, DEFAULT_OCTAVE

    # Sistema geometrico
    if args.geometric:
        parts = list(args.geometric)
        if len(parts) != 3:
            print("Uso: --geometric GEN STEPS INTERVAL")
            return None

        # Parse STEPS
        try:
            steps = int(parts[1])
        except (TypeError, ValueError):
            print("Numero di passi non valido")
            return None
        if steps <= 0:
            print("Numero di passi deve essere > 0")
            return None

        # Parse INTERVAL
        try:
            interval_ratio = parse_interval_value(parts[2])
        except argparse.ArgumentTypeError as e:
            print(f"Intervallo non valido: {e}")
            return None

        # Parse GEN as int or fraction, fallback to float
        try:
            gen_val = int_or_fraction(parts[0])
            gen_ratio = Fraction(gen_val, 1) if isinstance(gen_val, int) else gen_val
        except (argparse.ArgumentTypeError, ValueError) as e:
            try:
                gen_ratio = float(parts[0])
            except ValueError:
                print(f"Errore parsing geometrico: {e}")
                return None

        if float(gen_ratio) <= 0:
            print("Il generatore deve essere > 0")
            return None

        ratios = []
        for i in range(steps):
            r = pow_fraction(gen_ratio, i) if isinstance(gen_ratio, Fraction) else float(gen_ratio) ** i
            if not args.no_reduce:
                r = reduce_to_interval(r, interval_ratio)
            ratios.append(float(r))

        return ratios, float(interval_ratio)

    # Temperamento equabile (default)
    if args.et:
        index, cents = args.et
        if index <= 0 or (isinstance(cents, (int, float)) and cents <= 0):
            print("Indice o cents non valido")
            return None

        if is_fraction_type(cents):
            cents = fraction_to_cents(cents)


        try:
            ratio = ratio_et(index, cents)
        except ZeroDivisionError:
            print("Errore divisione per zero")
            return None

        ratios = [float(ratio ** i) for i in range(index)]
        interval_factor = math.exp(cents / ELLIS_CONVERSION_FACTOR)
        return ratios, interval_factor

    return None


def main():
    """Punto di ingresso principale / Main entry point (IT/EN)."""
    # Imposta lingua (pre-scan argv) e stampa banner sempre
    global _LANG
    _LANG = _detect_lang_from_argv()
    print_banner()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "IT: THIN – Generatore e comparatore di sistemi di intonazione (PHI). "
            "Supporta 12/24/48-TET, sistemi naturali, Danielou, geometrici, "
            "basenote microtonale (+ - ! .), esporta Csound/.tun/Excel (richiede openpyxl) e analisi audio (richiede librosa). "
            "EN: THIN – Generator and comparator of tuning systems (PHI). "
            "Supports 12/24/48-TET, natural, Danielou, geometric; microtonal basenote (+ - ! .); "
            "exports Csound/.tun/Excel (openpyxl requested) and audio analysis (librosa required)."
        ),
        epilog=(
            "IT - Note utili:\n"
            "• Notazione microtonale: + = +50c, - = -50c, ! = +25c, . = -25c (es.: C+4, A!3).\n"
            "• Confronto TET: --compare-tet 12|24|48, allineamento --compare-tet-align.\n"
            "• Analisi audio (consigliato 'lpc' per voce): --audio-file file.wav --analysis lpc|specenv "
            "--frame-size 1024 --hop-size 512 --lpc-order 12 --window hamming|hanning.\n"
            "• Output: out.csd (cpstun), out_system.txt/.xlsx, out_compare.txt/.xlsx, .tun opzionale.\n"
            "EN - Useful notes:\n"
            "• Microtonal notation: + = +50c, - = -50c, ! = +25c, . = -25c (e.g., C+4, A!3).\n"
            "• TET comparison: --compare-tet 12|24|48, alignment with --compare-tet-align.\n"
            "• Audio analysis (recommend 'lpc' for voice): --audio-file file.wav --analysis lpc|specenv "
            "--frame-size 1024 --hop-size 512 --lpc-order 12 --window hamming|hanning.\n"
            "• Outputs: out.csd (cpstun), out_system.txt/.xlsx, out_compare.txt/.xlsx, optional .tun.\n"
            "Esempi / Examples:\n"
            "  thin.py --lang it --et 24 1200 --basenote C+4 out\n"
            "  thin.py --geometric 3/2 12 2/1 --compare-tet 48 --audio-file voce.wav --analysis lpc out\n"
            "  thin.py --danielou-all --diapason 442 --export-tun out\n"
        )
    )

    # Lingua / Language
    parser.add_argument("--lang", choices=["it", "en"], default="it",
                        help="Lingua dell'interfaccia / Interface language")

    # Argomenti base / Basic arguments
    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {__version__}")
    parser.add_argument("--diapason", type=float, default=DEFAULT_DIAPASON,
                        help=(f"Diapason in Hz (default: {DEFAULT_DIAPASON}) / "
                              f"A4 reference (diapason) in Hz (default: {DEFAULT_DIAPASON})"))
    parser.add_argument("--basekey", type=int, default=DEFAULT_BASEKEY,
                        help=(f"Nota base MIDI (default: {DEFAULT_BASEKEY}) / "
                              f"Base MIDI note (default: {DEFAULT_BASEKEY})"))
    parser.add_argument("--basenote", type=note_name_or_frequency, default="C4",
                        help=(
                            "Nota di riferimento o frequenza in Hz; supporta microtoni (+ - ! .). / "
                            "Reference note or frequency in Hz; supports microtones (+ - ! .)."
                        ))

    # Sistemi di intonazione
    parser.add_argument("--et", nargs=2, type=int_or_fraction, default=(12, 1200),
                        metavar=("INDEX", "INTERVAL"),
                        help=("Temperamento equabile / Equal temperament"))
    parser.add_argument("--geometric", nargs=3, metavar=("GEN", "STEPS", "INTERVAL"),
                        help=(
                            "Sistema geometrico: INTERVAL intero=cents (es. 700); per rapporto usare 2/1 o 2.0; "
                            "accetta anche suffisso 'c' (es. 200c). / "
                            "Geometric system: INTERVAL integer=cents (e.g., 700); for ratios use 2/1 or 2.0; "
                            "also accepts 'c' suffix (e.g., 200c)."
                        ))
    parser.add_argument("--natural", nargs=2, type=int, metavar=("A_MAX", "B_MAX"),
                        help="Sistema naturale 4:5:6 / Natural system 4:5:6")
    # Nota: per esponenti negativi usare virgolette, es.: --danielou "-1,2,0"
    parser.add_argument("--danielou", action="append", type=parse_danielou_tuple,
                        default=None, help=("Sistema Danielou manuale (usa virgolette per esponenti negativi, es.: \"-1,2,0\") / "
                                           "Manual Danielou system (quote negative exponents, e.g.: \"-1,2,0\")"))
    parser.add_argument("--danielou-all", action="store_true",
                        help="Griglia completa Danielou / Full Danielou grid")

    # Opzioni
    parser.add_argument("--no-reduce", action="store_true",
                        help="Non ridurre all'ottava / Do not apply octave reduction")
    parser.add_argument("--span", "--ambitus", dest="span", type=int, default=1,
                        help="Ripetizioni dell'intervallo / Interval repetitions (span)")
    parser.add_argument("--interval-zero", action="store_true",
                        help="Imposta interval=0 in cpstun / Set interval=0 in cpstun")
    parser.add_argument("--export-tun", action="store_true",
                        help="Esporta file .tun / Export .tun file")
    parser.add_argument("--tun-integer", action="store_true",
                        help=".tun: arrotonda i cents al valore intero più vicino (default: due decimali) / .tun: round cents to nearest integer (default: two decimals)")

    # Confronto
    parser.add_argument("--compare-fund", nargs="?", type=note_name_or_frequency,
                        default="basenote", const="basenote",
                        help=("Fondamentale per confronto / Fundamental for comparison"))
    parser.add_argument("--compare-tet", type=int, choices=[12, 24, 48], default=12,
                        help=("Divisioni del TET per confronto (12, 24 o 48). / TET divisions for comparison (12, 24 or 48)."))
    parser.add_argument("--compare-tet-align", choices=["same", "nearest"],
                        default="same", help=("Allineamento TET / TET alignment"))
    parser.add_argument("--subharm-fund", type=note_name_or_frequency,
                        default="A5", help=("Fondamentale subarmoniche / Subharmonic fundamental"))
    parser.add_argument("--midi-truncate", action="store_true",
                        help=("Forza troncamento MIDI / Force MIDI truncation"))

    # Analisi audio (librosa) / Audio analysis (librosa)
    parser.add_argument("--audio-file", dest="audio_file", default=None,
                        help=("Percorso del file WAV da analizzare / Path to WAV file to analyze"))
    parser.add_argument("--analysis", choices=["lpc", "specenv"], default="lpc",
                        help=("Metodo analisi formanti: lpc (consigliato per voce) o specenv / Formant analysis method: lpc (recommended for voice) or specenv"))
    parser.add_argument("--frame-size", type=int, default=1024,
                        help=("Frame size (default: 1024)"))
    parser.add_argument("--hop-size", type=int, default=512,
                        help=("Hop size (default: 512)"))
    parser.add_argument("--lpc-order", type=int, default=12,
                        help=("Ordine LPC (default: 12) / LPC order (default: 12)"))
    parser.add_argument("--window", choices=["hamming", "hanning"], default="hamming",
                        help=("Finestra: hamming (default) o hanning / Window: hamming (default) or hanning"))

    # Output
    parser.add_argument("output_file", nargs="?", default=None,
                        help="File di output (default: out) / Output file (default: out)")

    # Mostra help se nessun argomento o se richiesto con -h/--help: usa pager "more"
    if len(sys.argv) == 1 or any(a in ("-h", "--help") for a in sys.argv[1:]):
        help_text = parser.format_help()
        page_lines(help_text.splitlines())
        return

    args = parser.parse_args()

    # Imposta lingua globale / Set global language
    _LANG = args.lang

    # Validazione base
    if isinstance(args.basenote, (int, float)) and args.basenote < 0:
        print(L("Nota di riferimento non valida", "Invalid reference note"))
        return

    if args.span is None or args.span < 1:
        args.span = 1

    # Calcola frequenza base (supporto microtoni) / Compute base frequency (microtonal support)
    if isinstance(args.basenote, float):
        basenote = args.basenote
    else:
        try:
            midi_base, cents_off = parse_note_with_microtones(str(args.basenote))
            basenote = apply_cents(convert_midi_to_hz(midi_base, args.diapason), cents_off)
        except ValueError as e:
            print(L(f"Errore conversione nota: {e}", f"Note conversion error: {e}"))
            return

    # Fondamentali per confronto
    def parse_fund_hz(value, default):
        if isinstance(value, str) and value.lower() == "basenote":
            return basenote
        elif isinstance(value, float):
            return value
        else:
            try:
                midi_f, cents_f = parse_note_with_microtones(str(value))
                return apply_cents(convert_midi_to_hz(midi_f, args.diapason), cents_f)
            except ValueError:
                return default

    compare_fund_hz = parse_fund_hz(args.compare_fund, basenote)
    subharm_fund_hz = parse_fund_hz(args.subharm_fund, args.diapason)


    # Processa sistema di intonazione
    result = process_tuning_system(args, basenote)
    if result is None:
        print(L("Nessun sistema di intonazione valido specificato", "No valid tuning system specified"))
        return

    ratios, interval = result

    # Applica span
    ratios_spanned = repeat_ratios(ratios, args.span, interval)
    ratios_eff, basekey_eff = ensure_midi_fit(ratios_spanned, args.basekey,
                                              args.midi_truncate)

    # Riepilogo esecuzione / Run summary
    output_base = args.output_file or "out"
    yes_label = L("sì", "yes")
    no_label = L("no", "no")

    # Descrizione sistema
    sys_desc = ""
    if args.natural:
        sys_desc = L("Sistema: Naturale 4:5:6", "System: Natural 4:5:6")
    elif args.danielou_all:
        sys_desc = L("Sistema: Danielou (griglia completa)", "System: Danielou (full grid)")
    elif args.danielou is not None:
        sys_desc = L("Sistema: Danielou (manuale)", "System: Danielou (manual)")
    elif args.geometric:
        sys_desc = L("Sistema: Geometrico", "System: Geometric")
    elif args.et:
        try:
            et_idx, et_int = args.et
            et_cents = fraction_to_cents(et_int) if is_fraction_type(et_int) else float(et_int)
        except Exception:
            et_idx = args.et[0] if args.et else 12
            et_cents = 1200
        sys_desc = L(f"Sistema: ET index={et_idx}, intervallo={et_cents}c",
                     f"System: ET index={et_idx}, interval={et_cents}c")

    # Analisi audio: linea localizzata
    if getattr(args, 'audio_file', None):
        analysis_line = L(f"Analisi audio: {yes_label} ({args.analysis}) file={args.audio_file}",
                          f"Audio analysis: {yes_label} ({args.analysis}) file={args.audio_file}")
    else:
        analysis_line = L(f"Analisi audio: {no_label}", f"Audio analysis: {no_label}")

    summary_lines = [
        L("— Riepilogo esecuzione —", "— Run summary —"),
        sys_desc,
        L(f"Nota di base (Hz): {basenote:.2f}", f"Basenote (Hz): {basenote:.2f}"),
        L(f"Diapason (A4): {args.diapason:.2f} Hz", f"Diapason (A4): {args.diapason:.2f} Hz"),
        L(f"Basekey MIDI: {basekey_eff}", f"MIDI basekey: {basekey_eff}"),
        L(f"Ripetizioni (span): {args.span}", f"Span (repetitions): {args.span}"),
        L(f"Riduzione all'ottava: {yes_label if not args.no_reduce else no_label}",
          f"Octave reduction: {yes_label if not args.no_reduce else no_label}"),
        L(f"Troncamento MIDI: {yes_label if args.midi_truncate else no_label}",
          f"MIDI truncation: {yes_label if args.midi_truncate else no_label}"),
        L(f"Confronto – fondamentale: {compare_fund_hz:.2f} Hz",
          f"Compare – fundamental: {compare_fund_hz:.2f} Hz"),
        L(f"TET: {args.compare_tet} ({args.compare_tet_align})",
          f"TET: {args.compare_tet} ({args.compare_tet_align})"),
        L(f"Subarmonica – fondamentale: {subharm_fund_hz:.2f} Hz",
          f"Subharmonic – fundamental: {subharm_fund_hz:.2f} Hz"),
        L(f"Base output: {output_base}", f"Output base: {output_base}"),
        analysis_line,
        ""
    ]
    page_lines(summary_lines)

    # Stampa tabella
    print_step_hz_table(sorted(ratios_eff), basenote)

    # Prepara dati per CSD
    if args.interval_zero:
        csd_input = ratios_spanned
        csd_interval = 0.0
    else:
        csd_input = ratios
        csd_interval = interval

    csd_ratios, csd_basekey = ensure_midi_fit(csd_input, args.basekey,
                                              args.midi_truncate)
    output_base = args.output_file or "out"
    fnum, existed = write_cpstun_table(output_base, csd_ratios,
                                       csd_basekey, basenote, csd_interval)

    # Export
    export_base = (output_base if not existed
                   else f"{output_base}_{fnum}")

    export_system_tables(export_base, ratios_eff, basekey_eff, basenote)

    # Analisi audio opzionale / Optional audio analysis
    analysis_data = None
    if getattr(args, 'audio_file', None):
        # Live spinner during audio analysis
        base_msg = L("Analisi audio in corso", "Audio analysis in progress")
        print(base_msg + " ", end="", flush=True)
        _stop_evt = threading.Event()
        def _spinner_worker():
            seq = ['|','/','-','\\']
            i = 0
            while not _stop_evt.is_set():
                ch = seq[i % len(seq)]
                print("\r" + base_msg + " " + ch, end="", flush=True)
                i += 1
                try:
                    time.sleep(0.2)
                except Exception:
                    break
        _th = threading.Thread(target=_spinner_worker, daemon=True)
        _th.start()
        try:
            analysis_data = analyze_audio(
                audio_path=args.audio_file,
                method=args.analysis,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                lpc_order=args.lpc_order,
                window_type=args.window,
            )
        finally:
            _stop_evt.set()
            try:
                _th.join(timeout=1.0)
            except Exception:
                pass
            print("\r" + " " * (len(base_msg) + 4) + "\r", end="")
            print("")  # newline after spinner
        if analysis_data is None:
            print(L("Analisi audio non disponibile o fallita: le tabelle saranno generate senza risultati di analisi.",
                    "Audio analysis unavailable or failed: tables will be generated without analysis results."))
        else:
            print(L("Analisi audio completata. Generazione delle tabelle di confronto...",
                    "Audio analysis completed. Generating comparison tables..."))
    else:
        print(L("Nessuna analisi audio: genero subito le tabelle di confronto.",
                "No audio analysis: generating comparison tables immediately."))

    export_comparison_tables(export_base, ratios_eff, basekey_eff, basenote,
                             args.diapason, compare_fund_hz,
                             args.compare_tet_align, subharm_fund_hz,
                             tet_divisions=args.compare_tet,
                             analysis_result=analysis_data)

    if args.export_tun:
        write_tun_file(export_base, ratios_eff, basekey_eff, basenote, args.tun_integer)


if __name__ == "__main__":
    main()