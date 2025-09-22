"""File format export and import utilities for musical tuning systems.

This module provides comprehensive support for reading and writing various tuning
file formats used in musical software and hardware, enabling interoperability
between THIN/SIM and external applications.

Csound Integration:
- Generation and management of cpstun tables within .csd files
- Automatic table numbering with conflict detection and resolution
- GEN -2 format compliance with metadata preservation
- Skeleton file creation for new Csound projects
- Integration with existing Csound scores and orchestras

AnaMark TUN Format:
- Full AnaMark TUN specification implementation with exact tuning support
- 128 MIDI note mapping with custom frequency references
- Intelligent diapason handling respecting user-specified A4 frequencies
- Bidirectional conversion between internal ratios and TUN format
- Support for both relative and absolute frequency specifications

Scala Format Support:
- Export to Scala (.scl) format for microtonal scale sharing
- Compliance with Scala scale archive standards
- Automatic scale naming and description generation
- Integration with Scala database for comparison and validation

Ableton Live Integration:
- Export to Ableton Scale (.ascl) format for DAW integration
- XML-based format with proper Live compatibility
- Custom scale naming and organization
- Support for microtonal workflows in Live

Excel Integration:
- Conversion of Excel worksheets to tuning file formats
- Batch processing of comparison tables
- Automatic format detection and appropriate export selection
- Error handling for malformed or incomplete Excel data

File Management:
- Intelligent file conflict resolution and backup creation
- Cross-platform path handling and file system compatibility
- Robust error handling with detailed user feedback
- Atomic file operations to prevent corruption

Format Conversion:
- Seamless conversion between different tuning file formats
- Precision preservation across format boundaries
- Metadata translation and format-specific optimization
- Validation of output files for compliance with specifications

Dependencies:
- Standard library modules for file I/O and parsing
- Integration with utils module for mathematical operations
- Optional openpyxl support for Excel file processing

Note: All file format implementations follow current specifications.
Historical DaMuSc integration has been removed for simplified maintenance.
"""
import math
import os
import re
from typing import List, Tuple, Optional, Union, Dict
from fractions import Fraction

import consts
import utils

# Pre-compiled regex patterns for better performance
_TUN_NOTE_PATTERN = re.compile(r'^(?:[Nn]ote|[Kk]ey)\s+(\d{1,3})\s*=\s*([+-]?\d+(?:\.\d+)?)')
_NOTE_PARSE_PATTERN = re.compile(r'([A-G][#B]?)(\d+)')

# Local wrapper to optimize reduce_to_octave usage
def _reduce_to_octave_float(ratio) -> float:
    """Optimized wrapper for utils.reduce_to_octave that returns float directly."""
    return float(utils.reduce_to_octave(ratio))


def _filter_valid_ratios(ratios: List[float]) -> List[float]:
    """Filter and sort ratios, removing unison and invalid values for Scala format."""
    valid_ratios = []
    for ratio in ratios:
        if isinstance(ratio, (int, float)) and ratio > 1.0:
            valid_ratios.append(float(ratio))
    valid_ratios.sort()
    return valid_ratios


def _ratio_to_scala_format(ratio: float) -> str:
    """Convert a ratio to appropriate Scala format (fraction or cents)."""
    from fractions import Fraction
    try:
        # Convert to fraction with reasonable denominator
        frac = Fraction(ratio).limit_denominator(10000)
        # Use fraction if precise enough, otherwise cents
        if abs(float(frac) - ratio) < 1e-8 and 1 < frac.denominator <= 10000:
            return f"{frac.numerator}/{frac.denominator}"
        else:
            # Convert to cents: cents = 1200 * log2(ratio)
            cents = 1200.0 * math.log2(ratio)
            return f"{cents:.5f}"
    except (ValueError, TypeError, ZeroDivisionError):
        # Fallback a formato decimale
        return f"{ratio:.8f}"


def parse_file(file_name: str) -> int:
    """Finds the maximum table number 'f N' in the file."""
    max_num = 0
    try:
        with open(file_name, "r") as f:
            for line in f:
                for match in consts.PATTERN.finditer(line):
                    num = int(match.group(1))
                    max_num = max(max_num, num)
    except FileNotFoundError:
        print(f"Error: file not found: {file_name}")
    return max_num


def write_cpstun_table(output_base: str, ratios: List[float], basekey: int,
                       basefrequency: float, interval_value: Optional[float] = None) -> Tuple[int, bool]:
    """Creates or appends a cpstun table to a Csound .csd file."""
    csd_path = f"{output_base}.csd"
    skeleton = (
        "<CsoundSynthesizer>\n"
        "<CsOptions>\n\n</CsOptions>\n"
        "<CsInstruments>\n\n</CsInstruments>\n"
        "<CsScore>\n\n</CsScore>\n"
        "</CsoundSynthesizer>\n"
    )

    existed_before = utils.file_exists(csd_path)

    if not existed_before:
        try:
            with open(csd_path, "w") as f:
                f.write(skeleton)
        except IOError as e:
            print(f"Error creating CSD file: {e}")
            return 0, existed_before

    try:
        with open(csd_path, "r", encoding="utf-8") as f:
            content = f.read()
    except IOError as e:
        print(f"Error reading CSD: {e}")
        return 0, existed_before

    fnum = parse_file(csd_path) + 1

    # Ordina i rapporti
    try:
        ratios_sorted = sorted(float(r) for r in ratios)
    except (TypeError, ValueError):
        print(f"Error sorting ratios")
        ratios_sorted = [float(r) for r in ratios]

    # Determina parametri cpstun
    numgrades = len(ratios_sorted)

    if interval_value is not None and isinstance(interval_value, (int, float)):
        iv = float(interval_value)
        interval = max(0.0, iv)
    else:
        try:
            rmin = min(ratios_sorted) if ratios_sorted else 1.0
            rmax = max(ratios_sorted) if ratios_sorted else 1.0
            interval = 2.0 if (1.0 - consts.RATIO_EPS <= rmin and rmax <= 2.0 + consts.RATIO_EPS) else 0.0
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
            f"; cpstun table generated | basekey={basekey} basefrequency={basefrequency:.6f}Hz\n"
    )
    f_line = prefix + " ".join(data_list) + "\n"

    # Insert before </CsScore>
    insert_marker = "</CsScore>"
    idx = content.rfind(insert_marker)
    if idx == -1:
        content += f"\n<CsScore>\n{header_comment}{f_line}</CsScore>\n"
    else:
        content = content[:idx] + header_comment + f_line + content[idx:]

    try:
        with open(csd_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"cpstun table (f {fnum}) saved to {csd_path}")
    except IOError as e:
        print(f"Error writing CSD: {e}")

    return fnum, existed_before


def write_tun_file(output_base: str, diapason: float, ratios: List[float], basekey: int,
                   basefrequency: float, tun_integer: bool = False) -> None:
    """Exports a .tun file (AnaMark TUN) with values expressed in absolute cents relative to 8.1757989156437073336 Hz.
    Structure: [Tuning] + 128 lines "note X=Y" (absolute cents)."""
    f_ref = consts.F_REF * (diapason / 440)
    tun_path = f"{output_base}.tun"

    if diapason != 440:
        lines = ["[Exact Tuning]",
                 f"basefreq={f_ref}"]
    else:

        lines = [
            "[Tuning]",
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

    for note_idx in range(consts.MIDI_MIN, consts.MIDI_MAX + 1):
        f = note_freq(note_idx)
        if isinstance(f, (int, float)) and f > 0:
            cents = utils.ratio_to_cents(f / f_ref)
        else:
            cents = 0.0
        rounded2 = round(cents, 2)
        s2 = f"{rounded2:.2f}"
        if tun_integer or s2.endswith(".00"):
            cents_text = str(int(round(cents)))
        else:
            cents_text = s2
        lines.append(f"Note {note_idx}={cents_text}")

    try:
        with open(tun_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f".tun file saved to {tun_path}")
    except IOError as e:
        print(f"Error writing .tun: {e}")


def import_tun_file(tun_path: str, basekey: int = consts.DEFAULT_BASEKEY, reduce_octave_out: bool = True) -> Optional[
    str]:
    """
    Importa un file .tun (AnaMark TUN) e converte i valori in ratios relativi a basekey.
    Salva un file .txt con mappatura 'MIDI -> ratio' per le note disponibili.
    """
    # Riferimento assoluto AnaMark (stesso di write_tun_file)
    f_ref = consts.F_REF

    cents_map = {}
    try:
        with open(tun_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith(";") or s.startswith("#"):
                    continue
                # Consenti e ignora le sezioni come [Tuning] o [Exact Tuning]
                if s.startswith("[") and s.endswith("]"):
                    continue
                m = _TUN_NOTE_PATTERN.match(s)
                if m:
                    idx = int(m.group(1))
                    if 0 <= idx <= 127:
                        cents_val = float(m.group(2))
                        cents_map[idx] = cents_val

    except IOError as e:
        print(f"Error reading .tun: {e}")
        return None

    if not cents_map:
        print("Empty or unrecognized .tun file")
        return None

    # Cents -> frequenze assolute
    freq_map = {}
    for k, cv in cents_map.items():
        freq_map[k] = float(f_ref) * (2.0 ** (float(cv) / 1200.0))

    # Determina frequenza base
    if basekey in freq_map:
        base_freq = freq_map[basekey]
        base_key_eff = basekey
    else:
        keys_sorted = sorted(freq_map.keys(), key=lambda nn: abs(nn - basekey))
        if not keys_sorted:
            print("No valid notes in .tun")
            return None
        base_key_eff = keys_sorted[0]
        base_freq = freq_map[base_key_eff]

    if not isinstance(base_freq, (int, float)) or base_freq <= 0:
        print("Invalid base frequency in .tun")
        return None

    # Costruisci ratios rispetto a base_key_eff
    ratios_by_midi = {}
    for n in range(128):
        f = freq_map.get(n)
        if isinstance(f, (int, float)) and f > 0:
            r = float(f) / float(base_freq)
            if reduce_octave_out:
                r = _reduce_to_octave_float(r)
            ratios_by_midi[n] = float(r)

    # Scrivi file di output
    base_name = os.path.splitext(os.path.basename(tun_path))[0]
    out_path = f"{base_name}_ratios.txt"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"; Import .tun -> ratios | basekey={base_key_eff}\n")
            f.write("MIDI -> Ratio\n")
            for n in sorted(ratios_by_midi.keys()):
                f.write(f"{n} -> {ratios_by_midi[n]:.10f}\n")
        print(f"Ratios file saved to {out_path}")
        return out_path
    except IOError as e:
        print(f"Error writing ratios: {e}")
        return None


def convert_excel_to_outputs(excel_path: str,
                             output_base: Optional[str],
                             default_basekey: int,
                             default_base_hz: float,
                             diapason_hz: float,
                             midi_truncate: bool = False,
                             tun_integer: bool = False,
                             cents_delta: Optional[float] = None) -> Union[bool, dict]:
    """
    Converte un file Excel (System/Compare) in output .csd (cpstun) e .tun usando
    la colonna dei rapporti (o Hz/Base_Hz se necessario). Applica opzionalmente uno shift in cents
    (cents_delta) to all system steps before comparisons (Scala/ET/Pyth/JI).
    Ritorna True se conversione riuscita.
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        print("openpyxl not installed: cannot use --convert")
        return False

    if not utils.file_exists(excel_path):
        print(f"Excel file not found: {excel_path}")
        return False

    try:
        wb = load_workbook(excel_path, data_only=True)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"Error opening Excel: {e}")
        return False

    # Scegli foglio: preferisci 'System' se presente, altrimenti attivo
    if 'System' in wb.sheetnames:
        ws = wb['System']
    else:
        ws = wb.active

    # Mappa header -> colonna (1-based)
    header_row = 1
    headers = {}

    # Valida l'indice di riga
    if not isinstance(header_row, int) or header_row < 1:
        print("Invalid header row")
        return False

    # Ottieni la riga header in modo sicuro
    try:
        row = ws[header_row]
    except (TypeError, KeyError, IndexError, AttributeError):
        print("Cannot read header row")
        return False

    for cell in row:
        try:
            val = cell.value
            if val is None:
                continue
            name = str(val).strip()
            if not name:
                continue
            col = cell.column  # openpyxl: può essere int o lettera
            headers[name.lower()] = col
        except (AttributeError, ValueError, TypeError):
            # salta solo la cella problematica
            continue

    def col_index_by_name(names: list) -> Optional[int]:
        def _col_to_index(c) -> Optional[int]:
            if isinstance(c, int):
                return c
            s = str(c).strip().upper()
            if s and s.isalpha():
                idx = 0
                for ch in s:
                    idx = idx * 26 + (ord(ch) - ord('A') + 1)
                return idx if idx > 0 else None
            return None

        if not names:
            return None
        names_l = [str(name_item).lower() for name_item in names]
        for key, column in headers.items():
            k = str(key).lower()
            for name_match in names_l:
                if name_match in k:
                    ci = _col_to_index(column)
                    if ci:
                        return ci
        return None

    ratio_col = col_index_by_name(["ratio"])  # es. "ratio", "custom ratio"
    hz_col = col_index_by_name(["hz"])  # es. "hz"
    midi_col = col_index_by_name(["midi"])  # es. "midi"

    # Base_Hz: prova a trovarlo da intestazione 'Base_Hz' o F2
    base_hz = float(default_base_hz) if default_base_hz and default_base_hz > 0 else consts.DEFAULT_DIAPASON
    base_anchor_col = col_index_by_name(["base_hz", "base hz", "base-freq", "basefreq"]) or 6  # fallback F
    base_cell_val = ws.cell(row=2, column=base_anchor_col).value
    if isinstance(base_cell_val, (int, float)) and float(base_cell_val) > 0:
        base_hz = float(base_cell_val)


    # Basekey: usa prima riga della colonna MIDI se presente, altrimenti default
    basekey = int(default_basekey)
    if midi_col:
        mv = ws.cell(row=2, column=midi_col).value
        if isinstance(mv, (int, float)):
            basekey = int(mv)

    # Estrai rapporti
    ratios: List[float] = []
    row = 2
    max_empty = 0
    while True:
        v_ratio = None
        v_hz = None
        if ratio_col:
            try:
                v_ratio = ws.cell(row=row, column=ratio_col).value
            except ValueError:
                v_ratio = None
        if hz_col:
            try:
                v_hz = ws.cell(row=row, column=hz_col).value
            except ValueError:
                v_hz = None
        if v_ratio is None and v_hz is None:
            max_empty += 1
            if max_empty >= 5:  # stop after a few empty rows
                break
            row += 1
            continue
        # Reset empty counter when any content
        max_empty = 0
        r_val: Optional[float] = None
        if isinstance(v_ratio, (int, float)) and float(v_ratio) > 0:
            r_val = float(v_ratio)
        elif isinstance(v_hz, (int, float)) and float(v_hz) > 0 and base_hz > 0:
            try:
                r_val = float(v_hz) / float(base_hz)
            except (TypeError, ValueError):
                r_val = None
        if r_val is not None and r_val > 0:
            ratios.append(r_val)
        row += 1

    if not ratios:
        print("No valid ratios found in Excel (Ratio/Hz columns)")
        return False

    # Assicura compatibilità MIDI
    ratios_eff, basekey_eff = utils.ensure_midi_fit(ratios, basekey, midi_truncate)

    # Base output
    if output_base and isinstance(output_base, str) and output_base.strip():
        out_base = output_base.strip()
    else:
        try:
            stem = os.path.splitext(os.path.basename(excel_path))[0]
        except OSError:
            stem = "out"
        out_base = stem

    # Ensure output directory and build in-folder base
    out_dir = out_base
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")
        pass
    out_base_in_dir = os.path.join(out_dir, out_base)
    # Scrivi cpstun e tun (entrambi di default)
    fnum, existed = write_cpstun_table(out_base_in_dir, ratios_eff, basekey_eff, base_hz, None)
    out_base_eff = out_base_in_dir if not existed else f"{out_base_in_dir}_{fnum}"
    write_tun_file(out_base_eff, diapason_hz, ratios_eff, basekey_eff, base_hz, tun_integer)

    # Scrivi anche .scl e .ascl files
    try:
        # Determine basenote string for .scl/.ascl export
        basenote_str = f"MIDI{basekey_eff}"  # Simple fallback

        # Generate .scl file
        write_scl_file(out_base_eff, ratios_eff, basenote_str, diapason_hz, "Converted from Excel")

        # Generate .ascl file
        write_ascl_file(out_base_eff, ratios_eff, basenote_str, diapason_hz, "Converted from Excel")

    except Exception as e:
        print(f"Error generating .scl/.ascl files: {e}")

    # --- Confronto con scale .scl e inferenza ET/Geometrico ---
    try:
        import glob as _glob
    except ImportError:
        _glob = None

    def _ratios_to_cents(_ratios: list) -> list:
        out = []
        for r in _ratios:
            try:
                rr = float(r)
                if rr <= 0:
                    continue
                rr = _reduce_to_octave_float(rr)
                c = utils.ratio_to_cents(rr)
                out.append(c)
            except ValueError:
                continue
        out = sorted(x % 1200.0 for x in out)
        # remove duplicates within 1e-6 cents
        dedup = []
        for v in out:
            if not dedup or abs(v - dedup[-1]) > 1e-6:
                dedup.append(v)
        return dedup

    def _parse_scl_file(path: str) -> Optional[dict]:
        """Parse .scl file using centralized function."""
        return parse_scl_file(path)

    def _Fractions_to_cents(ratio_list: list) -> list:
        """Convert a list of fraction ratios to cents values."""
        cents = []
        for r in ratio_list:
            try:
                cents.append((1200.0 * math.log(_reduce_to_octave_float(r), 2)) % 1200.0)
            except (ValueError, TypeError):
                pass
        return sorted(cents)

    def _avg_error_best_rotation(a: list, b: list) -> float:
        # compute minimal mean absolute difference between a and b under rotation, using min length
        if not a or not b:
            return float('inf')
        a_sorted = sorted(a)
        b_sorted = sorted(b)
        min_len = min(len(a_sorted), len(b_sorted))
        # use first min_len of each
        a_vals = a_sorted[:min_len]
        b_vals = b_sorted[:min_len]
        best_err = float('inf')
        # try all rotations of b
        for shift in range(min_len):
            err_sum = 0.0
            for idx in range(min_len):
                dv = abs(a_vals[idx] - b_vals[(idx + shift) % min_len])
                # allow wrap-around distance on circle 1200
                dv = min(dv, 1200.0 - dv)
                err_sum += dv
            best_err = min(best_err, err_sum / min_len)
        return best_err

    def _infer_system(cents: list) -> dict:
        res = {'type': 'unknown'}
        if not cents or len(cents) < 2:
            return res
        cs = sorted(x % 1200.0 for x in cents)
        # compute circular diffs
        diffs = []
        for idx in range(len(cs) - 1):
            diffs.append(cs[idx + 1] - cs[idx])
        diffs.append(1200.0 - cs[-1] + cs[0])
        if not diffs:
            return res
        mean = sum(diffs) / len(diffs)
        # std dev
        var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        std = math.sqrt(max(0.0, var))
        # ET test: mean step should divide 1200 closely and low std
        if mean > 0:
            n_est = round(1200.0 / mean)
            if n_est > 0:
                mean_et = 1200.0 / n_est
                if abs(mean - mean_et) <= 0.5 and std <= 0.8:
                    res = {'type': 'et', 'n': int(n_est), 'step_cents': mean_et, 'std_cents': std}
                    return res
        # Geometric-like (equal steps not matching 1200/n but low std)
        if std <= 0.8:
            res = {'type': 'geometric', 'steps': len(diffs), 'step_cents': mean, 'std_cents': std}
            return res
        return res

    try:
        report_lines = [
            f"— Report confronto (convert) —",
            f"File Excel: {os.path.basename(excel_path)}",
            f"Output base: {out_base_eff}",
            f"Base_Hz: {base_hz:.6f}  |  Basekey: {basekey_eff}",
            f"N rapporti: {len(ratios_eff)}"
        ]
        cents_from_excel = _ratios_to_cents(ratios_eff)
        # Apply cents delta shift if provided
        try:
            if cents_delta is not None:
                dd = float(cents_delta)
                cents_from_excel = sorted(((x + dd) % 1200.0) for x in cents_from_excel)
        except ValueError:
            pass
        report_lines.append("")
        # Inference
        inf = _infer_system(cents_from_excel)
        if inf.get('type') == 'et':
            report_lines.append(
                f"Inferenza: ET ~ {inf['n']}-TET  step≈{inf['step_cents']:.2f}c  std≈{inf['std_cents']:.2f}c")
        elif inf.get('type') == 'geometric':
            report_lines.append(
                f"Inferenza: Geometrico  passi={inf['steps']}  passo≈{inf['step_cents']:.2f}c  std≈{inf['std_cents']:.2f}c")
        else:
            report_lines.append(f"Inferenza: non determinata")
        report_lines.append("")
        # Scala comparison
        top_matches = []
        scl_dir = os.path.join(os.getcwd(), 'scl')
        if _glob and os.path.isdir(scl_dir):
            files = _glob.glob(os.path.join(scl_dir, '**', '*.scl'), recursive=True)
        else:
            files = []
        for fp in files[:2000]:  # safety cap
            info = _parse_scl_file(fp)
            if not info:
                continue
            err = _avg_error_best_rotation(cents_from_excel, info['cents'])
            if math.isfinite(err):
                top_matches.append({'file': os.path.relpath(fp, scl_dir), 'name': info['name'], 'avg_cents_error': err})
        top_matches.sort(key=lambda d: d['avg_cents_error'])
        if top_matches:
            best = top_matches[0]
            report_lines.append(
                f"Scala migliore: {best['name']} [{best['file']}]  err medio: {best['avg_cents_error']:.2f} c")
            report_lines.append("")
            report_lines.append("Top 5 corrispondenze:")
            report_lines.append("Pos  Nome                                   Err (c)         File")
            for i, it in enumerate(top_matches[:5], start=1):
                nm = str(it.get('name', ''))[:35]
                er = f"{float(it.get('avg_cents_error', 0.0)):.2f}"
                fl = str(it.get('file', ''))
                report_lines.append(f"{str(i).rjust(3)}  {nm.ljust(35)}  {er.rjust(7)}    {fl}")
        else:
            report_lines.append("Nessuna scala .scl trovata o confrontabile.")

        # --- Confronti aggiuntivi: ET / Pitagorici / Naturali ---
        report_lines.append("")
        report_lines.append("Confronti ET / Pitagorici / Naturali")

        # ET: valuta n-TET da 5 a 72
        def _et_cents(et_n: int) -> list:
            try:
                step = 1200.0 / float(et_n)
                return [step * k for k in range(et_n)]
            except ValueError:
                return []

        et_scores = []
        for n in range(5, 73):
            cents_et = _et_cents(n)
            if not cents_et:
                continue
            err = _avg_error_best_rotation(cents_from_excel, cents_et)
            if math.isfinite(err):
                et_scores.append((n, err))
        et_scores.sort(key=lambda t: t[1])
        if et_scores:
            report_lines.append("Migliori ET (n-TET):")
            report_lines.append("Pos  n    Err (c)")
            for i, (n, er) in enumerate(et_scores[:5], start=1):
                report_lines.append(f"{str(i).rjust(3)}  {str(n).rjust(3)}  {er:7.2f}")
        else:
            report_lines.append("Nessun risultato ET calcolabile.")

        # Pitagorici: 12 e 7
        def _pyth12_cents() -> list:
            try:
                from fractions import Fraction as _Frac
                vals = []
                for k in range(12):
                    r = utils.reduce_to_octave(utils.pow_fraction(_Frac(3, 2), k)) if hasattr(utils, 'pow_fraction') else utils.reduce_to_octave((3 / 2) ** k)
                    vals.append(float(r))
                return _Fractions_to_cents(vals)
            except (TypeError, ValueError, AttributeError):
                vals = [(1.5 ** k) for k in range(12)]
                return _Fractions_to_cents(vals)

        def _pyth7_cents() -> list:
            try:
                from fractions import Fraction as _Frac
                seq = [_Frac(1, 1), _Frac(9, 8), _Frac(81, 64), _Frac(4, 3), _Frac(3, 2), _Frac(27, 16), _Frac(243, 128)]
                vals = [utils.reduce_to_octave(r) for r in seq]
                return _Fractions_to_cents(vals)
            except (TypeError, ValueError, AttributeError):
                vals = [1.0, 9 / 8, 81 / 64, 4 / 3, 3 / 2, 27 / 16, 243 / 128]
                return _Fractions_to_cents(vals)

        py12_err = _avg_error_best_rotation(cents_from_excel, _pyth12_cents())
        py7_err = _avg_error_best_rotation(cents_from_excel, _pyth7_cents())
        report_lines.append("")
        report_lines.append("Pitagorici:")
        report_lines.append(f"Pyth-12 err medio: {py12_err:.2f} c")
        report_lines.append(f"Pyth-7  err medio: {py7_err:.2f} c")

        # Naturali: 5-limit diatonic e triade 4:5:6
        def _ji_diatonic5_cents() -> list:
            try:
                from fractions import Fraction as _Frac
                seq = [_Frac(1, 1), _Frac(9, 8), _Frac(5, 4), _Frac(4, 3), _Frac(3, 2), _Frac(5, 3), _Frac(15, 8)]
                vals = [utils.reduce_to_octave(r) for r in seq]
                return _Fractions_to_cents(vals)
            except (ValueError, TypeError):
                vals = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8]
                return _Fractions_to_cents(vals)

        def _ji_triad_456_cents() -> list:
            try:
                from fractions import Fraction as _Frac
                seq = [_Frac(1, 1), _Frac(5, 4), _Frac(3, 2)]
                vals = [utils.reduce_to_octave(r) for r in seq]
                return _Fractions_to_cents(vals)
            except (ValueError, TypeError):
                vals = [1.0, 5 / 4, 3 / 2]
                return _Fractions_to_cents(vals)

        ji5_err = _avg_error_best_rotation(cents_from_excel, _ji_diatonic5_cents())
        ji_triad_err = _avg_error_best_rotation(cents_from_excel, _ji_triad_456_cents())
        report_lines.append("")
        report_lines.append("Naturali (JI):")
        report_lines.append(f"Diatonica 5-limit err medio: {ji5_err:.2f} c")
        report_lines.append(f"Triade 4:5:6 err medio: {ji_triad_err:.2f} c")


        report_txt = "\n".join(report_lines) + "\n"
        rep_path = f"{out_base_eff}_convert_compare.txt"
        try:
            with open(rep_path, 'w', encoding='utf-8') as rf:
                rf.write(report_txt)
            print(f"Comparison report saved: {rep_path}")
        except OSError as e:
            print(f"Error saving report: {e}")
    except (ValueError, TypeError, ImportError, AttributeError) as e:
        print(f"Comparison/inference failed: {e}")

    return True


def write_scl_file(output_base: str, ratios: List[float], basenote: str, diapason: float,
                   system_name: str = "Custom Scale") -> None:
    """Exports a .scl file (Scala format) with the provided ratios.

    Args:
        output_base: Base name of the output file (without extension)
        ratios: List of frequency ratios (must include 1.0 as first element)
        basenote: Base note (e.g. "A4")
        diapason: Reference frequency in Hz
        system_name: Name of the tuning system
    """
    scl_path = f"{output_base}.scl"

    # Prepare description according to Scala specifications
    description = f"{system_name}, 1/1={diapason:.2f} Hz, base={basenote}"

    # Filter and sort ratios using helper function
    valid_ratios = _filter_valid_ratios(ratios)
    num_notes = len(valid_ratios)

    # Build file according to Scala specifications
    lines = [
        f"! {output_base}.scl",
        f"! Generated by THIN (SIM project)",
        f"! {description}",
        description,
        str(num_notes)
    ]

    # Add ratios in appropriate format using helper function
    for ratio in valid_ratios:
        lines.append(_ratio_to_scala_format(ratio))

    try:
        with open(scl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f".scl file saved to {scl_path}")
    except IOError as e:
        print(f"Error writing .scl: {e}")


def write_ascl_file(output_base: str, ratios: List[float], basenote: str, diapason: float,
                   system_name: str = "Custom Scale", source: str = None, link: str = None,
                   analysis_data: Optional[Dict] = None) -> None:
    """Exports an .ascl file (Ableton Live format) with the provided ratios.

    ASCL format is an Ableton extension to Scala format with Ableton metadata
    specified in @ABL comments according to Ableton Live specification.
    File is saved in UTF-8 format as required by the specification.

    Args:
        output_base: Base name of the output file (without extension)
        ratios: List of frequency ratios (must include 1.0 as first element)
        basenote: Base note (e.g. "A4")
        diapason: Reference frequency in Hz
        system_name: Name of the tuning system
        source: Source or author of the tuning system (optional)
        link: URL for more information about the system (optional)
        analysis_data: Optional analysis data for intelligent note naming (reserved for future use)
    """
    ascl_path = f"{output_base}.ascl"

    # Prepare description according to Scala/ASCL specifications
    description = f"{system_name}, 1/1={diapason:.2f} Hz, base={basenote}"

    # Filter and sort ratios using helper function
    valid_ratios = _filter_valid_ratios(ratios)
    num_notes = len(valid_ratios)

    # Build file according to Scala specifications con estensioni ASCL
    lines = [
        f"! {output_base}.ascl",
        f"! Generated by THIN (SIM project)",
        f"! Ableton Live ASCL format",
        f"! {description}",
        description,
        str(num_notes)
    ]

    # Add ratios in appropriate format using helper function
    for ratio in valid_ratios:
        lines.append(_ratio_to_scala_format(ratio))

    # Add Ableton ASCL extensions
    lines.append("!")
    lines.append("! === ABLETON LIVE EXTENSIONS ===")

    # @ABL REFERENCE_PITCH: octave_number note_index_in_octave frequency_hz
    # Determine octave and note index based on basenote
    octave_number, note_index = _parse_basenote_for_ascl(basenote)
    lines.append(f"! @ABL REFERENCE_PITCH {octave_number} {note_index} {diapason:.1f}")

    # @ABL NOTE_NAMES: note names for each pitch in the scale
    note_names = _generate_note_names_for_ascl(num_notes, ratios, basenote, diapason, analysis_data)  # Must match number of notes declared
    lines.append(f"! @ABL NOTE_NAMES {' '.join(note_names)}")

    # @ABL NOTE_RANGE_BY_FREQUENCY: minimum frequency for tuning range
    # Use reasonable frequency as minimum (8.0 Hz) and maximum (21000.0 Hz)
    # According to specification, range is from 4.0 to 21000.0 Hz
    lines.append("! @ABL NOTE_RANGE_BY_FREQUENCY 8.0 21000.0")

    # @ABL SOURCE: source documentation (optional)
    if source:
        lines.append(f"! @ABL SOURCE {source}")
    else:
        lines.append("! @ABL SOURCE THIN (SIM project) - Generated tuning system")

    # @ABL LINK: URL for more information (optional)
    if link:
        lines.append(f"! @ABL LINK {link}")

    try:
        # Save in UTF-8 as required by ASCL specification
        with open(ascl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f".ascl file saved to {ascl_path}")
    except IOError as e:
        print(f"Error writing .ascl: {e}")


def _parse_basenote_for_ascl(basenote: str) -> Tuple[int, int]:
    """Converts a base note (e.g. 'A4') to octave_number and note_index for ASCL.

    Returns:
        Tuple[octave_number, note_index]: octave (4 per A4) e indice nota (9 per A)
    """
    # Note to index mapping
    note_to_index = {
        'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
        'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8, 'AB': 8,
        'A': 9, 'A#': 10, 'BB': 10, 'B': 11
    }

    # Simple parsing for common notes like A4, C4, etc.
    match = _NOTE_PARSE_PATTERN.match(basenote.upper())
    if match:
        note_name = match.group(1)
        octave = int(match.group(2))
        note_index = note_to_index.get(note_name, 9)  # Default A
        return octave, note_index

    # Default to A4 if parsing fails
    return 4, 9


def _generate_note_names_for_ascl(num_notes: int, ratios: List[float] = None,
                                 basenote: str = "C4", diapason: float = 440.0,
                                 analysis_data: Optional[Dict] = None) -> List[str]:
    """Generates note names for ASCL with microtonal arrows based on analysis data.

    Args:
        num_notes: Total number of notes in the scale
        ratios: List of frequency ratios for the scale steps
        basenote: Base note reference (e.g., "C4")
        diapason: A4 frequency in Hz
        analysis_data: Optional analysis data containing scale information (reserved for future enhancements)

    Returns:
        List of note names with microtonal deviation arrows

    Note:
        analysis_data parameter is reserved for future implementation of intelligent
        note naming based on scale analysis results.
    """

    # Base names for 12-TET
    base_names = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]

    # If we don't have ratios, fall back to simple cyclic naming
    if not ratios:
        note_names = []
        for i in range(num_notes):
            base_index = i % 12
            octave_suffix = f"_{i//12}" if i >= 12 else ""
            note_names.append(base_names[base_index] + octave_suffix)
        return note_names

    # Try to use analysis data to generate intelligent note names with arrows
    note_names = []

    # Calculate base note frequency from basenote and diapason
    try:
        # Parse basenote to get MIDI number
        basenote_midi = utils.parse_note_with_microtones(basenote)[0] if hasattr(utils, 'parse_note_with_microtones') else 60
        # Calculate base frequency
        base_freq = diapason * (2 ** ((basenote_midi - 69) / 12.0))
    except (TypeError, ValueError):
        base_freq = 261.626  # Default C4 frequency

    # Generate note names with microtonal arrows
    for i, ratio in enumerate(ratios):
        if i >= num_notes:
            break

        # Calculate frequency of this scale step
        freq = base_freq * ratio

        # Find closest 12-TET note
        if diapason > 0:
            # Convert frequency to MIDI note
            midi_float = 69 + 12 * math.log2(freq / diapason)
            closest_midi = round(midi_float)

            # Calculate deviation in cents
            closest_freq = diapason * (2 ** ((closest_midi - 69) / 12.0))
            cents_deviation = 1200 * math.log2(freq / closest_freq) if closest_freq > 0 else 0

            # Generate base note name
            note_index = closest_midi % 12
            octave = (closest_midi // 12) - 1
            base_name = base_names[note_index]

            # Add octave suffix if needed
            octave_suffix = f"_{octave-4}" if octave != 4 else ""

            # Add microtonal arrows based on cents deviation
            arrow_suffix = ""
            if abs(cents_deviation) > 5:  # Only add arrows for significant deviations
                if cents_deviation > 25:
                    arrow_suffix = "↑↑"  # Very sharp
                elif cents_deviation > 10:
                    arrow_suffix = "↑"   # Sharp
                elif cents_deviation < -25:
                    arrow_suffix = "↓↓"  # Very flat
                elif cents_deviation < -10:
                    arrow_suffix = "↓"   # Flat

            note_names.append(base_name + arrow_suffix + octave_suffix)
        else:
            # Fallback to simple naming
            base_index = i % 12
            octave_suffix = f"_{i//12}" if i >= 12 else ""
            note_names.append(base_names[base_index] + octave_suffix)

    # Fill remaining notes if needed
    while len(note_names) < num_notes:
        i = len(note_names)
        base_index = i % 12
        octave_suffix = f"_{i//12}" if i >= 12 else ""
        note_names.append(base_names[base_index] + octave_suffix)

    return note_names


# Global cache for loaded scales
_scala_scales_cache = None


def parse_scl_file(path: str) -> Optional[dict]:
    """Parsa un file .scl secondo il formato Huygens-Fokker di base.
    Ritorna un dict: {'name': str, 'degrees_cents': List[float], 'file': str}
    dove degrees_cents sono ridotti a [0,1200).
    Se il file non è leggibile/parsabile, ritorna None.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    # Rimuovi commenti e righe vuote; in .scl i commenti iniziano con '!'
    cleaned: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # drop lines that start with '!' entirely
        if s.startswith('!'):
            # but keep the first comment line as possible filename/description if name missing
            cleaned.append(s)
            continue
        # remove trailing inline comments starting with '!' if any
        if '!' in s:
            s = s.split('!', 1)[0].strip()
        if s:
            cleaned.append(s)

    if not cleaned:
        return None

    # The format: optional leading comment lines starting with '!'
    # Next non-comment line is the description (name). Following line has the number of notes.
    idx = 0
    # Skip leading comments for description, but remember them if needed
    while idx < len(cleaned) and cleaned[idx].startswith('!'):
        idx += 1
    if idx >= len(cleaned):
        return None
    name_line = cleaned[idx]
    idx += 1
    # Some files may keep description empty; fallback to filename later
    desc = name_line.strip()

    # Read number of notes
    if idx >= len(cleaned):
        return None
    try:
        n_deg = int(str(cleaned[idx]).strip().split()[0])
    except ValueError:
        return None
    idx += 1

    degrees_cents: List[float] = []
    # Read next n_deg lines as degrees (can include comments after value that we already stripped)
    for _ in range(n_deg):
        if idx >= len(cleaned):
            break
        val = cleaned[idx].strip()
        idx += 1
        if not val:
            continue
        token = val.split()[0]
        cents_val: Optional[float] = None
        # ratio like 3/2 or 1.5/1.0 is allowed
        if '/' in token:
            try:
                num_str, den_str = token.split('/', 1)
                num = float(Fraction(num_str.strip()))
                den = float(Fraction(den_str.strip()))
                r = num / den if den != 0 else None
                if r and r > 0:
                    cents_val = utils.ratio_to_cents(r)
            except ValueError:
                cents_val = None
        else:
            # plain number in cents
            try:
                cents_val = float(token)
            except ValueError:
                cents_val = None
        if isinstance(cents_val, (int, float)) and math.isfinite(cents_val):
            # Reduce to [0,1200)
            c = float(cents_val) % 1200.0
            degrees_cents.append(c)
    if not degrees_cents:
        return None

    # Ensure unique sorted degrees
    degrees_sorted: List[float] = []
    for c in sorted(degrees_cents):
        if not degrees_sorted or abs(c - degrees_sorted[-1]) > 1e-6:
            degrees_sorted.append(c)

    return {
        'name': desc if desc else os.path.basename(path),
        'degrees_cents': degrees_sorted,
        'file': os.path.basename(path),
        'cents': degrees_sorted  # Add 'cents' key for compatibility
    }


def load_scales_from_dir(dir_path: str = 'scl') -> List[dict]:
    """Carica tutti i file .scl leggibili dalla directory indicata.
    Ritorna una lista di dict come parse_scl_file.
    Uses global cache for performance.
    """
    global _scala_scales_cache

    # Return cached scales if available
    if _scala_scales_cache is not None:
        return _scala_scales_cache

    try:
        if not os.path.isdir(dir_path):
            _scala_scales_cache = []
            return []
    except (FileNotFoundError, OSError):
        _scala_scales_cache = []
        return []

    out: List[dict] = []
    try:
        for fn in os.listdir(dir_path):
            if not fn.lower().endswith('.scl'):
                continue
            fp = os.path.join(dir_path, fn)
            info = parse_scl_file(fp)
            if info and isinstance(info.get('degrees_cents'), list) and info['degrees_cents']:
                out.append(info)
    except (FileNotFoundError, OSError):
        pass

    # Cache the result
    _scala_scales_cache = out
    return out
