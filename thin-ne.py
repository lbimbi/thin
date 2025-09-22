#!/usr/bin/env python3
"""THIN - Version PHI

Advanced musical intonation systems tool with audio analysis and Scala integration.
"""

import argparse
import glob
import math
import os
import sys
from typing import List, cast

import audio_analysis as aa
import consts
import sim
import tables
import tun_csd
import utils


def main() -> None:
    """Main entry point"""
    # Setup warning suppression and logging
    utils.setup_warning_logging("thin_warnings.log")
    utils.suppress_tensorflow_warnings()

    # Theme pre-scan and apply before any styled output
    utils.clear_screen()
    utils.print_banner()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "THIN – Advanced Generator and Analyzer of Musical Tuning Systems (PHI)\n"
            "\n"
            "Generate, analyze and compare microtonal tuning systems with audio analysis.\n"
            "Supports mathematical generation (ET, Geometric, Natural, Danielou),\n"
            "audio-based extraction, Scala database integration, and export to\n"
            "multiple professional formats (.csd/.tun/.scl/.ascl).\n"
        ),
        epilog=(
            "QUICK REFERENCE:\n\n"

            "NOTATION:\n"
            "  Microtonal: + = +50c, - = -50c, ! = +25c, . = -25c\n"
            "  Examples: C+4 (C4+50c), A!3 (A3+25c), --basenote 261.626\n\n"

            "TUNING SYSTEMS:\n"
            "  --et 19 1200             19-TET equal temperament\n"
            "  --geometric 3/2 12 2/1   Pythagorean (perfect fifths)\n"
            "  --natural 8 8            Natural 4:5:6 harmonics\n"
            "  --danielou \"2,3,1\" --danielou \"1,0,1\"  Multiple custom ratios\n"
            "  --danielou-all           Complete Danielou grid\n\n"

            "EXPORT FORMATS:\n"
            "  .csd (Csound cpstun)    Always generated\n"
            "  .tun (AnaMark TUN)      --export-tun\n"
            "  .scl (Scala)            --export-scl\n"
            "  .ascl (Ableton Live)    --export-ableton\n"
            "  .xlsx (Excel analysis)  Automatic with comparisons\n\n"

            "AUDIO ANALYSIS:\n"
            "  --audio-file voice.wav --analysis cqt|lpc\n"
            "  --diapason-analysis     Estimate A4 and infer tuning\n"
            "  --render                Export analysis as audio\n"
            "  --scala/--inferred      Force system selection\n\n"

            "CLUSTERING PARAMETERS:\n"
            "  --tolerance 1.5         Clustering tolerance (default: 2.0%)\n"
            "  --min-ratio-count 2     Min occurrences for ratio inclusion\n"
            "  --confidence-threshold 0.8  CREPE F0 confidence filter\n"
            "  --clustering-method adaptive  simple|weighted|adaptive\n"
            "  --scale-size-hint 12    Expected scale degrees (adaptive mode)\n"
            "  --duration-weighting    Weight by note duration (weighted mode)\n"
            "  --harmonic-bias         Favor simple integer ratios\n\n"

            "EXAMPLES:\n"
            "  thin-ne.py --et 19 1200 --export-scl my_19tet\n"
            "  thin-ne.py --audio-file voice.wav --diapason-analysis out_voice\n"
            "  thin-ne.py --convert my_tuning.xlsx my_output\n\n"

            "DEPENDENCIES:\n"
            "  Core: openpyxl\n"
            "  Audio: librosa scipy matplotlib\n"
            "  High-quality F0: crepe tensorflow\n"
        )
    )

    # Groups for clearer help layout
    grp_base = parser.add_argument_group("Base", "Base")
    grp_tuning = parser.add_argument_group("Tuning systems", "Tuning systems")
    grp_opts = parser.add_argument_group("Extra options", "Extra options")
    grp_cmp = parser.add_argument_group("Comparison", "Comparison")
    grp_audio = parser.add_argument_group("Audio analysis (librosa)", "Audio analysis (librosa)")
    grp_out = parser.add_argument_group("Output", "Output")

    # Language
    grp_base.add_argument("-v", "--version", action="version",
                          version=f"%(prog)s {consts.__version__}")
    grp_base.add_argument("--diapason", type=float, default=consts.DEFAULT_DIAPASON,
                          help=f"Diapason in Hz (default: {consts.DEFAULT_DIAPASON})")
    grp_base.add_argument("--basekey", type=int, default=consts.DEFAULT_BASEKEY,
                          help=f"MIDI base note (default: {consts.DEFAULT_BASEKEY})")
    grp_base.add_argument("--basenote", type=utils.note_name_or_frequency, default="C4",
                          help="Reference note or frequency in Hz; supports microtones (+ - ! .)")

    # Tuning systems
    grp_tuning.add_argument("--et", nargs=2, type=utils.int_or_fraction, default=(12, 1200),
                            metavar=("DIVISIONS", "INTERVAL"),
                            help="Equal temperament with N divisions of interval in cents "
                                 "(e.g., --et 19 1200 for 19-TET)")
    grp_tuning.add_argument("--geometric", nargs=3, metavar=("GENERATOR", "STEPS", "INTERVAL"),
                            help="Geometric progression: GENERATOR as ratio (3/2) or decimal "
                                 "(1.5), STEPS count, INTERVAL as cents (1200) or ratio (2/1). "
                                 "Examples: --geometric 3/2 12 2/1 (Pythagorean)")
    grp_tuning.add_argument("--natural", nargs=2, type=int, metavar=("A_MAX", "B_MAX"),
                            help="Natural intonation system based on 4:5:6 ratios with "
                                 "harmonics up to A_MAX:B_MAX limits")
    # Note: for negative exponents use quotes, e.g.: --danielou "-1,2,0"
    grp_tuning.add_argument("--danielou", action="append", type=sim.parse_danielou_tuple,
                            default=None, help="Manual Danielou microtonal system with custom "
                                               "exponent triplets (a,b,c) for ratios 2^a * 3^b * 5^c. "
                                               "Can be used multiple times for multiple ratios. "
                                               "Use quotes for negative values: \"-1,2,0\"")
    grp_tuning.add_argument("--danielou-all", action="store_true",
                            help="Generate complete Danielou 53-comma microtonal grid")

    # Extra options
    grp_opts.add_argument("--no-reduce", action="store_true",
                          help="Don't reduce to octave")
    grp_opts.add_argument("--span", "--ambitus", dest="span", type=int, default=1,
                          help="Interval repetitions (span)")
    grp_opts.add_argument("--interval-zero", action="store_true",
                          help="Set interval=0 in cpstun")
    grp_opts.add_argument("--export-tun", action="store_true",
                          help="Export .tun file")
    grp_opts.add_argument("--export-scl", action="store_true",
                          help="Export .scl file (Scala format)")
    grp_opts.add_argument("--export-ableton", action="store_true",
                          help="Export .ascl file (Ableton Live format)")
    grp_opts.add_argument("--tun-integer", action="store_true",
                          help=".tun: round cents to nearest integer value (default: two decimals)")
    grp_opts.add_argument("--convert", metavar="FILE.xlsx", dest="convert", default=None,
                          help="Convert an Excel sheet (System/Compare) to .csd/.tun/.scl/.ascl using 'Ratio' column or Hz/Base_Hz")
    grp_opts.add_argument("--import-tun", metavar="FILE.tun", dest="import_tun", default=None,
                          help="Import a .tun file and save a .txt with MIDI->ratio mapping")

    # Comparison
    grp_cmp.add_argument("--compare-fund", nargs="?", type=utils.note_name_or_frequency,
                         default="basenote", const="basenote",
                         help="Fundamental for comparison")
    grp_cmp.add_argument("--compare-tet", type=int, choices=[12, 24, 48], default=12,
                         help="TET divisions for comparison (12, 24 or 48).")
    grp_cmp.add_argument("--compare-tet-align", choices=["same", "nearest"],
                         default="same", help="TET alignment")
    grp_cmp.add_argument("--subharm-fund", type=utils.note_name_or_frequency,
                         default="A5", help="Subharmonic fundamental")
    grp_cmp.add_argument("--midi-truncate", action="store_true",
                         help="Force MIDI truncation")
    grp_cmp.add_argument("--cents-delta", type=float, default=None,
                         help="Analysis with cents tolerance for Scala comparisons; generates <output>_cents_delta.txt")
    grp_cmp.add_argument("--tolerance", type=float, default=2.0,
                         help="Clustering tolerance percentage for ratio grouping (default: 2.0). "
                              "Lower values increase precision, higher values group more ratios together. "
                              "Example: 1.0 for 1%%, 0.5 for 0.5%%")
    grp_cmp.add_argument("--min-ratio-count", type=int, default=1,
                         help="Minimum number of occurrences required for a ratio cluster to be included (default: 1). "
                              "Higher values filter out rare ratios, keeping only frequently occurring ones")
    grp_cmp.add_argument("--confidence-threshold", type=float, default=0.0,
                         help="Minimum confidence threshold for F0 detection with CREPE (default: 0.0). "
                              "Range: 0.0-1.0. Higher values use only high-confidence F0 estimates")
    grp_cmp.add_argument("--clustering-method", choices=["simple", "weighted", "adaptive"], default="simple",
                         help="Clustering algorithm: 'simple' (basic proximity), 'weighted' (duration-based), "
                              "'adaptive' (variable tolerance based on scale size)")
    grp_cmp.add_argument("--scale-size-hint", type=int, default=None,
                         help="Expected number of degrees in the scale (used with adaptive clustering). "
                              "Helps adaptive algorithm adjust tolerance for better results")
    grp_cmp.add_argument("--duration-weighting", action="store_true",
                         help="Enable duration-based weighting in clustering. Longer notes get more weight "
                              "(requires --clustering-method weighted)")
    grp_cmp.add_argument("--harmonic-bias", action="store_true",
                         help="Apply harmonic bias to favor simple integer ratios in clustering results")

    # Audio analysis (librosa)
    grp_audio.add_argument("--audio-file", dest="audio_file", default=None,
                           help="Path to audio file for analysis (WAV recommended, "
                                "supports most formats via librosa)")
    grp_audio.add_argument("--analysis", choices=["cqt", "lpc"], default="cqt",
                           help="Formant analysis method: 'cqt' for Constant-Q Transform "
                                "(better for musical content), 'lpc' for Linear Predictive "
                                "Coding (better for speech/vocals)")
    grp_audio.add_argument("--frame-size", type=int, default=1024,
                           help="Analysis frame size in samples (default: 1024). Used for "
                                "F0 tracking with librosa (basic analysis) and STFT fallback. "
                                "Note: --diapason-analysis uses CREPE (ignores this setting)")
    grp_audio.add_argument("--hop-size", type=int, default=512,
                           help="Hop size between analysis frames in samples (default: 512). "
                                "Smaller = smoother time tracking, larger = faster processing")
    grp_audio.add_argument("--window", choices=["hamming", "hanning"], default="hamming",
                           help="Window function: 'hamming' (default, general purpose) "
                                "or 'hanning' (better frequency resolution)")
    grp_audio.add_argument("--cqt-bins", dest="cqt_bins", type=int, default=None,
                           help="Total number of bins for CQT analysis (override). "
                                "If omitted, automatic calculation.")
    grp_audio.add_argument("--cqt-bpo", dest="cqt_bpo", type=int, default=None,
                           help="Bins per octave for CQT analysis (override). "
                                "Internal default: 48.")
    grp_audio.add_argument("--cqt-fmin", dest="cqt_fmin", type=float, default=None,
                           help="Minimum frequency for CQT analysis (Hz). "
                                "Internal default: 55.0 Hz.")
    grp_audio.add_argument("--lpc-order", dest="lpc_order", type=int, default=None,
                           help="LPC order (number of coefficients). If omitted, "
                                "auto-calculated based on sample rate.")
    grp_audio.add_argument("--lpc-preemph", dest="lpc_preemph", type=float, default=0.97,
                           help="Pre-emphasis coefficient for LPC analysis (default: 0.97).")
    grp_audio.add_argument("--lpc-window-ms", dest="lpc_window_ms", type=float, default=None,
                           help="LPC window length in milliseconds. If omitted, "
                                "auto-calculated based on frame size.")
    grp_audio.add_argument("--diapason-analysis", action="store_true",
                           help="Estimate diapason (A4) from audio using F0+CQT analysis "
                                "and attempt to infer the tuning system (ratios). Results "
                                "exported to dedicated text file and Excel sheet.")
    grp_audio.add_argument("--f0-only", action="store_true",
                           help="With --diapason-analysis: use only F0 tracking, exclude "
                                "CQT data (faster but less accurate for complex signals).")
    grp_audio.add_argument("--digits", type=int, default=None, metavar="N",
                           help="Round output values to N decimal places for display "
                                "(requires --diapason-analysis). Analysis internally uses "
                                "full precision. If omitted, shows full precision.")
    grp_audio.add_argument("--render", action="store_true",
                           help="With --audio-file: export WAV files of frequency-controlled "
                                "sinusoids for auditory feedback of the analysis.")
    grp_audio.add_argument("--scala-cent", type=float, default=None,
                           help="List all .scl scales with avg error <= threshold (cents).")
    grp_audio.add_argument("--cross-inference", nargs="?", metavar="FILE.wav", dest="cross_inference_file",
                           default=None, const=True,
                           help="Cross-inference between CQT and F0 data to determine scale "
                                "notes. Optional FILE.wav for F0 extraction; if omitted, "
                                "uses --audio-file for both CQT and F0")

    # Export system choice (mutually exclusive)
    export_choice = grp_audio.add_mutually_exclusive_group()
    export_choice.add_argument("--scala", action="store_true",
                               help="Force exports to use Scala match system "
                                    "(requires --diapason-analysis and any export option)")
    export_choice.add_argument("--inferred", action="store_true",
                               help="Force exports to use inferred system "
                                    "(requires --diapason-analysis and any export option)")

    # Output
    grp_out.add_argument("output_file", nargs="?", default=None, help="Output file (default: out)")

    # Show help if no arguments or if requested with -h/--help: print and exit (no pager)
    if len(sys.argv) == 1 or any(a in ("-h", "--help") for a in sys.argv[1:]):
        help_text = parser.format_help()
        print(help_text)
        return

    args = parser.parse_args()

    # Detect if the user actually provided --diapason on CLI
    try:
        user_diapason_set = any(opt == "--diapason" for opt in sys.argv[1:])
    except (AttributeError, TypeError, IndexError):
        user_diapason_set = False

    # Validate export system choice options
    if getattr(args, 'scala', False) or getattr(args, 'inferred', False):
        has_export = (getattr(args, 'export_tun', False) or
                      getattr(args, 'export_scl', False) or
                      getattr(args, 'export_ableton', False))
        has_diapason = getattr(args, 'diapason_analysis', False)
        if not (has_export and has_diapason):
            print(
                "Error: --scala and --inferred options require --diapason-analysis and at least one of --export-tun/--export-scl/--export-ableton")
            return

    # Validate cross-inference option
    if getattr(args, 'cross_inference_file', None) is not None:
        if not (getattr(args, 'audio_file', None) and getattr(args, 'diapason_analysis', False)):
            print("Error: --cross-inference requires both --audio-file and --diapason-analysis")
            return

    # Validate f0-only option
    if getattr(args, 'f0_only', False):
        if not getattr(args, 'diapason_analysis', False):
            print("Error: --f0-only requires --diapason-analysis")
            return

    # Validate digits option
    if getattr(args, 'digits', None) is not None:
        if not getattr(args, 'diapason_analysis', False):
            print("Error: --digits requires --diapason-analysis")
            return
        if args.digits < 0:
            print("Error: --digits must be a non-negative integer")
            return

    # Extract commonly used getattr values to avoid duplication
    audio_params = utils.extract_audio_params(args)

    # Import .tun if requested
    if hasattr(args, 'import_tun') and args.import_tun:
        tun_csd.import_tun_file(args.import_tun, basekey=int(args.basekey))
        return

    # Basic validation
    if isinstance(args.basenote, (int, float)) and args.basenote < 0:
        print(f"Invalid reference note")
        return

    if args.span is None or args.span < 1:
        args.span = 1

    # Compute base frequency (microtonal support)
    if isinstance(args.basenote, float):
        basenote = args.basenote
    else:
        try:
            midi_base, cents_off = utils.parse_note_with_microtones(str(args.basenote))
            basenote = utils.apply_cents(utils.convert_midi_to_hz(midi_base, args.diapason), cents_off)
        except ValueError as e:
            print(f"Note conversion error: {e}")
            return

    # Se richiesta conversione Excel -> .csd/.tun, esegui e termina
    if hasattr(args, 'convert') and args.convert:
        tun_csd.convert_excel_to_outputs(
            excel_path=args.convert,
            output_base=args.output_file,
            default_basekey=args.basekey,
            default_base_hz=basenote,
            diapason_hz=args.diapason,
            midi_truncate=args.midi_truncate,
            tun_integer=args.tun_integer,
            cents_delta=audio_params.get('cents_delta'),
        )
        return

    # Fundamentals for comparison
    compare_fund_hz = utils.parse_fundamental_frequency(args.compare_fund, basenote, args.diapason, basenote)
    subharm_fund_hz = utils.parse_fundamental_frequency(args.subharm_fund, basenote, args.diapason, args.diapason)

    # Process tuning system
    result = sim.process_tuning_system(args, basenote)
    if result is None:
        print("No valid tuning system specified")
        return

    ratios, interval = result

    # Applica span
    ratios_spanned = utils.repeat_ratios(ratios, args.span, interval)
    ratios_eff, basekey_eff = utils.ensure_midi_fit(ratios_spanned, args.basekey,
                                                    args.midi_truncate)

    # Run summary
    output_base = args.output_file or "out"
    yes_label = "yes"
    no_label = "no"

    # System description
    sys_desc = ""
    if args.natural:
        sys_desc = "System: Natural 4:5:6"
    elif args.danielou_all:
        sys_desc = "System: Danielou (complete grid)"
    elif args.danielou is not None:
        sys_desc = "System: Danielou (manual)"
    elif args.geometric:
        sys_desc = "System: Geometric"
    elif args.et:
        try:
            et_idx, et_int = args.et
            et_cents = utils.fraction_to_cents(et_int) if utils.is_fraction_type(et_int) else float(et_int)
        except (TypeError, ValueError):
            et_idx = args.et[0] if args.et else 12
            et_cents = 1200
        sys_desc = f"System: ET index={et_idx}, interval={et_cents}c"

    # Audio analysis: line
    if hasattr(args, 'audio_file') and args.audio_file:
        analysis_line = f"Audio analysis: {yes_label} ({args.analysis}) file={args.audio_file}"
    else:
        analysis_line = f"Audio analysis: {no_label}"

    summary_lines = [
        "— Run Summary —",
        sys_desc,
        f"Base note (Hz): {basenote:.2f}",
        f"Diapason (A4): {args.diapason:.2f} Hz",
        f"Basekey MIDI: {basekey_eff}",
        f"Repetitions (span): {args.span}",
        f"Octave reduction: {'yes' if not args.no_reduce else 'no'}",
        f"MIDI truncation: {'yes' if args.midi_truncate else 'no'}",
        f"Comparison – fundamental: {compare_fund_hz:.2f} Hz",
        f"TET: {args.compare_tet} ({args.compare_tet_align})",
        f"Subharmonic – fundamental: {subharm_fund_hz:.2f} Hz",
        f"Base output: {output_base}",
        analysis_line,
        ""
    ]
    utils.page_lines(summary_lines)

    # Print table
    utils.print_step_hz_table(sorted(ratios_eff), basenote)

    # Prepare data for CSD
    if args.interval_zero:
        csd_input = ratios_spanned
        csd_interval = 0.0
    else:
        csd_input = ratios
        csd_interval = interval

    csd_ratios, csd_basekey = utils.ensure_midi_fit(csd_input, args.basekey,
                                                    args.midi_truncate)
    output_base = args.output_file or "out"
    # Ensure output directory named after out_base exists and use it as parent
    out_dir = output_base
    try:
        os.makedirs(out_dir, exist_ok=True)
    except (OSError, PermissionError):
        pass
    out_base_in_dir = os.path.join(out_dir, output_base)
    fnum, existed = tun_csd.write_cpstun_table(out_base_in_dir, csd_ratios,
                                               csd_basekey, basenote, csd_interval)

    # Export
    export_base = (out_base_in_dir if not existed
                   else f"{out_base_in_dir}_{fnum}")

    tables.export_system_tables(export_base, ratios_eff, basekey_eff, basenote)

    # Optional audio analysis
    analysis_data = None
    if hasattr(args, 'audio_file') and args.audio_file:
        # Use spinner animation from utils
        with utils.SpinnerAnimation("Audio analysis in progress"):
            analysis_data = aa.analyze_audio(
                audio_path=args.audio_file,
                method=args.analysis,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                window_type=args.window,
                cqt_n_bins=audio_params.get('cqt_bins'),
                cqt_bins_per_octave=audio_params.get('cqt_bpo'),
                cqt_fmin=audio_params.get('cqt_fmin'),
                f0_only=audio_params.get('f0_only', False),
                lpc_order=audio_params.get('lpc_order'),
                lpc_preemph=audio_params.get('lpc_preemph', 0.97),
                lpc_window_ms=audio_params.get('lpc_window_ms'),
            )
        if analysis_data is None:
            msg = "Audio analysis unavailable or failed: tables will be generated without analysis results."
            print(msg)
            # Run diapason analysis anyway if requested
            if audio_params.get('diapason_analysis'):
                est = aa.estimate_diapason_and_ratios(
                    audio_path=args.audio_file,
                    base_hint_hz=basenote,
                    initial_a4_hz=args.diapason,
                    frame_size=args.frame_size,
                    hop_size=args.hop_size,
                    scala_cent=audio_params.get('scala_cent'),
                    digits=audio_params.get('digits'),
                    f0_only=audio_params.get('f0_only', False),
                    cqt_analysis=None,  # No CQT data available when analysis_data is None
                    formant_method=args.analysis,
                    lpc_order=audio_params.get('lpc_order'),
                    lpc_preemph=audio_params.get('lpc_preemph', 0.97),
                    lpc_window_ms=audio_params.get('lpc_window_ms'),
                    tolerance_percent=args.tolerance,
                    min_ratio_count=args.min_ratio_count,
                    confidence_threshold=args.confidence_threshold,
                    clustering_method=args.clustering_method,
                    scale_size_hint=args.scale_size_hint,
                    duration_weighting=args.duration_weighting,
                    harmonic_bias=args.harmonic_bias,
                )
                if est:
                    analysis_data = dict(est)
                    a4v = est.get('diapason_est')
                    if a4v and isinstance(a4v, (int, float)):
                        print(f"Diapason estimated from audio (A4): {a4v:.2f} Hz")
        else:
            msg = "Audio analysis completed. Generating comparison tables..."
            print(msg)
            # Diapason analysis (optional): estimate A4 and ratios from audio without 12-TET assumptions
            if audio_params.get('diapason_analysis'):
                est = aa.estimate_diapason_and_ratios(
                    audio_path=args.audio_file,
                    base_hint_hz=basenote,
                    initial_a4_hz=args.diapason,
                    frame_size=args.frame_size,
                    hop_size=args.hop_size,
                    scala_cent=audio_params.get('scala_cent'),
                    digits=audio_params.get('digits'),
                    f0_only=audio_params.get('f0_only', False),
                    cqt_analysis=analysis_data,  # Pass CQT data for enhanced diapason estimation
                    formant_method=args.analysis,
                    lpc_order=audio_params.get('lpc_order'),
                    lpc_preemph=audio_params.get('lpc_preemph', 0.97),
                    lpc_window_ms=audio_params.get('lpc_window_ms'),
                    tolerance_percent=args.tolerance,
                    min_ratio_count=args.min_ratio_count,
                    confidence_threshold=args.confidence_threshold,
                    clustering_method=args.clustering_method,
                    scale_size_hint=args.scale_size_hint,
                    duration_weighting=args.duration_weighting,
                    harmonic_bias=args.harmonic_bias,
                )
                if est:
                    if analysis_data is None:
                        analysis_data = {}
                    analysis_data.update(est)
                    a4v = est.get('diapason_est')
                    if a4v and isinstance(a4v, (int, float)):
                        print(f"Diapason estimated from audio (A4): {a4v:.2f} Hz")
                        # Suggested basenote from analysis (nearest 12-TET)
                        bn_hz = est.get('basenote_est_hz')
                        bn_midi = est.get('basenote_est_midi')
                        bn_name = est.get('basenote_est_name_12tet')

                        if any([bn_hz, bn_midi, bn_name]):
                            hz_txt = f"{bn_hz:.2f} Hz" if isinstance(bn_hz, (int, float)) else ""
                            midi_txt = f"MIDI {bn_midi}" if bn_midi is not None else ""
                            parts = [str(bn_name or ''), midi_txt, hz_txt]
                            line = f"Basenote suggested from analysis: {' '.join(p for p in parts if p)}"
                            print(line)
                        # Audio Analysis Results Section
                        print("\n" + "=" * 60)
                        print("                AUDIO ANALYSIS RESULTS")
                        print("=" * 60)

                        # Display A4 confidence information if available
                        diapason_est = est.get('diapason_est')
                        diapason_ci = est.get('diapason_confidence_interval')
                        diapason_std = est.get('diapason_confidence_std')
                        diapason_samples = est.get('diapason_sample_count')
                        diapason_delta_hz = est.get('diapason_delta_440_hz')
                        diapason_delta_percent = est.get('diapason_delta_440_percent')

                        if diapason_est:
                            print(f"\nEstimated A4 (diapason): {diapason_est:.2f} Hz")

                            # Show delta from standard A4 = 440 Hz
                            if diapason_delta_hz is not None and diapason_delta_percent is not None:
                                sign = "+" if diapason_delta_hz >= 0 else ""
                                print(
                                    f"  └─ Delta from 440 Hz: {sign}{diapason_delta_hz:.2f} Hz ({sign}{diapason_delta_percent:.2f}%)")

                            if diapason_samples and diapason_samples > 0:
                                print(f"  └─ Samples used: {diapason_samples}")

                            if diapason_std:
                                print(f"  └─ Confidence std: ±{diapason_std:.2f} Hz")

                            if diapason_ci:
                                print(f"  └─ 95% CI: [{diapason_ci[0]:.2f}, {diapason_ci[1]:.2f}] Hz")

                        # Scala matches are now saved only to files (print statements removed)

                        # Refined scale matches are now saved only to files (print statements removed)

                        # Top-5 comparison output removed as per requirements

                        # Close audio analysis results section
                        print("=" * 60)
    else:
        _noaud = "No audio analysis: generating comparison tables immediately."
        print(_noaud)

    # Cross-inference analysis if requested
    if getattr(args, 'cross_inference_file', None) is not None and analysis_data:
        # Determine F0 source file: use specified file or same as --audio-file
        f0_source_file = args.cross_inference_file if isinstance(args.cross_inference_file, str) else args.audio_file

        print("Performing cross-inference between CQT and F0 data...")
        if f0_source_file == args.audio_file:
            print(f"Using same file for both CQT and F0: {f0_source_file}")
        else:
            print(f"CQT from: {args.audio_file}, F0 from: {f0_source_file}")

        cross_inference_data = aa.cross_inference_cqt_f0(
            cqt_analysis=analysis_data,
            f0_audio_path=f0_source_file
        )
        if cross_inference_data:
            num_corr = cross_inference_data.get('num_correlations', 0)
            f0_fund = cross_inference_data.get('f0_fundamental', 0)
            scale_ratios = cross_inference_data.get('inferred_scale_ratios', [])
            print(f"Cross-inference completed: {num_corr} correlations found")
            print(f"F0 fundamental: {f0_fund:.2f} Hz")
            print(f"Inferred scale: {len(scale_ratios)} degrees")

            # Update analysis_data with cross-inference results
            if analysis_data is None:
                analysis_data = {}
            analysis_data['cross_inference'] = cross_inference_data

            # Export cross-inference results to text file
            try:
                cross_output_path = f"{export_base}_cross_inference.txt"
                with open(cross_output_path, 'w', encoding='utf-8') as f:
                    f.write("— Cross-Inference Analysis Report —\n\n")
                    f.write(f"CQT source file: {args.audio_file}\n")
                    f.write(f"F0 source file: {f0_source_file}\n")
                    f.write(f"F0 fundamental: {f0_fund:.2f} Hz\n")
                    f.write(f"Number of correlations: {num_corr}\n")
                    f.write(f"Inferred scale degrees: {len(scale_ratios)}\n\n")

                    # List correlations
                    correlations = cross_inference_data.get('correlations', [])
                    if correlations:
                        f.write("CQT-F0 Correlations:\n")
                        f.write("CQT Freq (Hz)  Harmonic#  Expected (Hz)  Error (c)  Ratio\n")
                        for corr in correlations:
                            f.write(f"{corr['cqt_freq']:11.2f}  {corr['harmonic_number']:9d}  "
                                    f"{corr['expected_freq']:11.2f}  {corr['error_cents']:8.2f}  "
                                    f"{corr['ratio']:6.4f}\n")
                        f.write("\n")

                    # List scale ratios and cents
                    if scale_ratios:
                        f.write("Inferred Scale (reduced to octave):\n")
                        f.write("Ratio     Cents\n")
                        scale_cents = cross_inference_data.get('inferred_scale_cents', [])
                        for i, ratio in enumerate(scale_ratios):
                            cents = scale_cents[i] if i < len(scale_cents) else 0.0
                            f.write(f"{ratio:6.4f}   {cents:6.2f}\n")

                utils.log_export_success(cross_output_path)
            except (OSError, IOError) as e:
                print(f"Failed to export cross-inference results: {e}")
        else:
            print("Cross-inference failed or no correlations found")

    # Decide effective diapason for computations: if user did not specify --diapason and we have an estimate,
    # use the estimated A4; otherwise use the user-provided value.
    comp_diapason = args.diapason
    try:
        est_a4 = (analysis_data or {}).get('diapason_est') if isinstance(analysis_data, dict) else None
        if audio_params.get('diapason_analysis') and (not user_diapason_set) and isinstance(est_a4, (int,
                                                                                                     float)) and est_a4 > 0:
            comp_diapason = float(est_a4)
    except (TypeError, ValueError):
        comp_diapason = args.diapason

        # Attach user diapason metadata for display purposes in Diapason reports/sheets

        if analysis_data is None:
            analysis_data = {}
        if isinstance(analysis_data, dict):
            analysis_data['user_diapason_set'] = bool(user_diapason_set)
            if isinstance(args.diapason, (int, float)):
                analysis_data['user_diapason_value'] = float(args.diapason)

    tables.export_comparison_tables(export_base, ratios_eff, basekey_eff, basenote,
                                    comp_diapason, compare_fund_hz, subharm_fund_hz,
                                    tet_divisions=args.compare_tet,
                                    analysis_result=analysis_data,
                                    delta_threshold_hz=audio_params.get('delta_threshold_hz', 0.0))

    # Export separate diapason Excel file if analysis data is available
    if analysis_data:
        tables.export_diapason_excel(export_base, analysis_data, basenote, comp_diapason)

    # Always generate textual Diapason report when --diapason-analysis is active
    try:
        if audio_params.get('diapason_analysis') and analysis_data:
            # For the Diapason text report, show A4_utente as follows:
            #   - if user explicitly set --diapason: use that value
            #   - otherwise: use the estimated A4 from analysis (if available), falling back to default
            try:
                a4_user_display = float(args.diapason)
                if not user_diapason_set:
                    est = analysis_data.get('diapason_est') if isinstance(analysis_data, dict) else None
                    if isinstance(est, (int, float)) and est > 0:
                        a4_user_display = float(est)
            except (TypeError, ValueError, AttributeError):
                a4_user_display = float(args.diapason)
            utils.generate_diapason_text_fallback(export_base, float(a4_user_display), float(basenote), analysis_data,
                                                  args.digits)
    except (ValueError, TypeError, OSError, IOError, AttributeError):
        pass

    # --cents-delta report (Scala within given cents tolerance)
    try:
        if audio_params.get('cents_delta') is not None:
            delta = float(args.cents_delta)
            # Build cents from current system ratios
            cents_sys = []
            for r in (ratios_eff or []):
                try:
                    rr = float(r)
                    if rr <= 0:
                        continue
                    # reduce to octave and convert to cents
                    rr = float(utils.reduce_to_octave(rr))
                    c = (1200.0 * math.log(rr, 2.0)) % 1200.0
                    cents_sys.append(float(c))
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
            cents_sys = sorted(cents_sys)

            # Use centralized function from utils

            # Use centralized function from utils

            # Scan Scala files
            scl_matches = []
            scl_dir = os.path.join(os.getcwd(), 'scl')
            scl_files = glob.glob(os.path.join(scl_dir, '**', '*.scl'), recursive=True) if os.path.isdir(
                scl_dir) else []
            for fp in scl_files:
                info = tun_csd.parse_scl_file(fp)
                if not info:
                    continue
                err = utils.avg_error_rotation(cents_sys, cast(List[float], info['cents']))
                if math.isfinite(err) and err <= delta:
                    scl_matches.append(
                        {'file': os.path.relpath(fp, scl_dir), 'name': info['name'], 'avg_cents_error': err})
            scl_matches.sort(key=lambda d: d['avg_cents_error'])

            # Compose report
            lines = [
                "— Cents-delta Report —",
                f"Output base: {export_base}",
                f"Delta (cents): {delta:.2f}",
                "",
                f"Scala entro <= {delta:.2f} cents (N={len(scl_matches)})"
            ]
            if scl_matches:
                lines.append("Pos  Name                                   Err (c)         File")
                for pos_num, it in enumerate(scl_matches[:50], start=1):
                    lines.append(utils.format_scala_match_line(pos_num, it))
            else:
                lines.append("(none)")
            lines.append("")
            out_path = f"{export_base}_cents_delta.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
            utils.log_export_success(out_path)
    except (OSError, IOError, PermissionError, ValueError) as e:
        print(f"Error generating --cents-delta report: {e}")

    # PNG plot export for diapason analysis
    if audio_params.get('diapason_analysis') and analysis_data:
        _png_path = aa.export_diapason_plot_png(export_base, analysis_data, basenote, comp_diapason)
        if _png_path:
            print(f"Diapason PNG plot saved: {_png_path}")
        else:
            print("Diapason PNG export unavailable or failed")

        # Export CQT plot at 300 DPI
        _cqt_path = aa.export_cqt_plot_png(export_base, analysis_data, dpi=300)
        if _cqt_path:
            print(f"CQT PNG plot saved: {_cqt_path}")
        else:
            print("CQT PNG export unavailable or failed")

    # Render audio (frequency-controlled sinusoid) if requested
    if audio_params.get('render_enabled') and getattr(args, 'audio_file', None) and analysis_data:
        # Ensure a fallback diapason value is present in analysis data to allow render in absence of F0
        try:
            if isinstance(analysis_data, dict):
                dv = analysis_data.get('diapason_est')
                if not isinstance(dv, (int, float)) or not (float(dv) > 0):
                    if isinstance(args.diapason, (int, float)) and float(args.diapason) > 0:
                        analysis_data['diapason_est'] = float(args.diapason)
        except (ValueError, TypeError, AttributeError, KeyError):
            pass
        _render_path = aa.render_analysis_to_wav(args.audio_file, analysis_data, export_base)
        if _render_path:
            print(f"Audio render saved: {_render_path}")
        else:
            print("Audio render failed or unavailable")
        # Additionally render constant diapason tone for 30s
        try:
            a4_freq = aa.estimate_a4_from_analysis(analysis_data, default_a4=float(args.diapason))
        except (TypeError, ValueError, AttributeError):
            a4_freq = float(args.diapason) if isinstance(args.diapason, (int, float)) else consts.DEFAULT_DIAPASON
        _dia_path = None
        if isinstance(a4_freq, (int, float)) and float(a4_freq) > 0:
            _dia_path = aa.render_constant_tone_wav(export_base, float(a4_freq), duration_s=30.0, sr=48000)
        if _dia_path:
            print(f"Diapason render (30s sine) saved: {_dia_path}")
        else:
            print("Diapason render unavailable or failed")

    if args.export_tun:
        # Use diapason analysis ratios if available, otherwise use default system ratios
        tun_ratios = ratios_eff
        tun_basekey = basekey_eff
        tun_basenote = basenote
        tun_diapason = args.diapason

        if audio_params.get('diapason_analysis') and analysis_data:
            # Check user preference for system choice
            force_inferred = getattr(args, 'inferred', False)
            force_scala = getattr(args, 'scala', False)

            use_inferred_system = False
            inferred_ratios = None

            if force_inferred:
                # User explicitly wants inferred system
                tuning_inferred = analysis_data.get('tuning_inferred')
                if tuning_inferred and isinstance(tuning_inferred, dict):
                    scale_steps = analysis_data.get('scale_steps')
                    if scale_steps and len(scale_steps) > 0:
                        inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                        use_inferred_system = True
                        print(
                            f"Using inferred system (error: {tuning_inferred.get('avg_cents_error', 'unknown'):.2f}c)")
                    else:
                        print("Warning: Inferred system requested but no scale_steps available")
                else:
                    print("Warning: Inferred system requested but no tuning_inferred data available")

            elif force_scala:
                # User explicitly wants Scala match
                scala_match = analysis_data.get('scala_match_info')
                if scala_match and isinstance(scala_match, dict):
                    scala_steps = analysis_data.get('scala_match_steps')
                    if scala_steps and len(scala_steps) > 0:
                        inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                        use_inferred_system = True
                        print(f"Using Scala match system (error: {scala_match.get('avg_cents_error', 'unknown'):.2f}c)")
                    else:
                        print("Warning: Scala match requested but no scala_match_steps available")
                else:
                    print("Warning: Scala match requested but no scala_match_info available")

            else:
                # Automatic selection based on < 2 cents error (existing logic)
                # Check inferred tuning system first
                tuning_inferred = analysis_data.get('tuning_inferred')
                if tuning_inferred and isinstance(tuning_inferred, dict):
                    inferred_error = tuning_inferred.get('avg_cents_error', float('inf'))
                    if isinstance(inferred_error, (int, float)) and inferred_error < 2.0:
                        scale_steps = analysis_data.get('scale_steps')
                        if scale_steps and len(scale_steps) > 0:
                            # Extract ratios from scale_steps: (index, ratio, count)
                            inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                            use_inferred_system = True
                            print(f"Auto-selecting inferred system (error: {inferred_error:.2f}c < 2.0c)")

                # If no good inferred system, check scala match
                if not use_inferred_system:
                    scala_match = analysis_data.get('scala_match_info')
                    if scala_match and isinstance(scala_match, dict):
                        scala_error = scala_match.get('avg_cents_error', float('inf'))
                        if isinstance(scala_error, (int, float)) and scala_error < 2.0:
                            scala_steps = analysis_data.get('scala_match_steps')
                            if scala_steps and len(scala_steps) > 0:
                                # Extract ratios from scala_match_steps: (index, ratio, count)
                                inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                                use_inferred_system = True
                                print(f"Auto-selecting Scala match system (error: {scala_error:.2f}c < 2.0c)")

            if use_inferred_system and inferred_ratios:
                tun_ratios = inferred_ratios
            else:
                # Fallback to raw analysis ratios
                diapason_ratios = analysis_data.get('ratios_reduced')
                if diapason_ratios and len(diapason_ratios) > 0:
                    tun_ratios = diapason_ratios
                    if not (force_inferred or force_scala):
                        print("Using raw analysis ratios (no system with error < 2.0c)")
                    else:
                        print("Fallback: Using raw analysis ratios")

            # Use estimated diapason from analysis only if user didn't specify one
            est_diapason = analysis_data.get('diapason_est')
            if est_diapason and isinstance(est_diapason, (int, float)) and est_diapason > 0 and not user_diapason_set:
                tun_diapason = float(est_diapason)
                print(f"Using estimated diapason from analysis: {est_diapason:.2f} Hz")
            elif user_diapason_set:
                print(f"Using user-specified diapason: {args.diapason:.2f} Hz")
            # Use estimated basenote from analysis if available
            est_basenote_hz = analysis_data.get('basenote_est_hz')
            if est_basenote_hz and isinstance(est_basenote_hz, (int, float)) and est_basenote_hz > 0:
                tun_basenote = float(est_basenote_hz)
            est_basenote_midi = analysis_data.get('basenote_est_midi')
            if est_basenote_midi and isinstance(est_basenote_midi, int) and 0 <= est_basenote_midi <= 127:
                tun_basekey = est_basenote_midi

        tun_csd.write_tun_file(export_base, tun_diapason, tun_ratios, tun_basekey, tun_basenote, args.tun_integer)

    # Export SCL file if requested
    if args.export_scl:
        # Use diapason analysis ratios if available, otherwise use default system ratios
        scl_ratios = ratios_eff

        # Apply same diapason logic as TUN export
        scl_diapason = args.diapason  # Default
        if audio_params.get('diapason_analysis') and analysis_data:
            est_diapason = analysis_data.get('diapason_est')
            if est_diapason and isinstance(est_diapason, (int, float)) and est_diapason > 0 and not user_diapason_set:
                scl_diapason = float(est_diapason)
                print(f"SCL: Using estimated diapason from analysis: {est_diapason:.2f} Hz")
            elif user_diapason_set:
                print(f"SCL: Using user-specified diapason: {args.diapason:.2f} Hz")

        # Generate system name based on the selected options
        if hasattr(args, 'et') and args.et:
            # args.et might be a list [divisions, interval], so extract the divisions
            et_divisions = args.et[0] if isinstance(args.et, list) else args.et
            scl_system_name = f"Equal Temperament {et_divisions}-TET"
        elif hasattr(args, 'geometric') and args.geometric:
            scl_system_name = "Geometric progression"
        elif hasattr(args, 'natural') and args.natural:
            scl_system_name = "Natural intonation"
        elif hasattr(args, 'danielou') and args.danielou:
            scl_system_name = "Danielou microtonal system"
        else:
            scl_system_name = "Custom Scale"

        if audio_params.get('diapason_analysis') and analysis_data:
            # Check user preference for system choice (same logic as TUN export)
            force_inferred = getattr(args, 'inferred', False)
            force_scala = getattr(args, 'scala', False)

            use_inferred_system = False
            inferred_ratios = None

            if force_inferred:
                # User explicitly wants inferred system
                tuning_inferred = analysis_data.get('tuning_inferred')
                if tuning_inferred and isinstance(tuning_inferred, dict):
                    scale_steps = analysis_data.get('scale_steps')
                    if scale_steps and len(scale_steps) > 0:
                        inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                        use_inferred_system = True
                        scl_system_name = f"Inferred ({tuning_inferred.get('type', 'Unknown')})"
                    else:
                        print("Warning: Inferred system requested but no scale_steps available for SCL export")
                else:
                    print("Warning: Inferred system requested but no tuning_inferred data available for SCL export")

            elif force_scala:
                # User explicitly wants Scala match
                scala_match = analysis_data.get('scala_match_info')
                if scala_match and isinstance(scala_match, dict):
                    scala_steps = analysis_data.get('scala_match_steps')
                    if scala_steps and len(scala_steps) > 0:
                        inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                        use_inferred_system = True
                        scl_system_name = f"Scala match ({scala_match.get('name', 'Unknown')})"
                    else:
                        print("Warning: Scala match requested but no scala_match_steps available for SCL export")
                else:
                    print("Warning: Scala match requested but no scala_match_info available for SCL export")

            else:
                # Automatic selection based on < 2 cents error (same logic as TUN)
                tuning_inferred = analysis_data.get('tuning_inferred')
                if tuning_inferred and isinstance(tuning_inferred, dict):
                    inferred_error = tuning_inferred.get('avg_cents_error', float('inf'))
                    if isinstance(inferred_error, (int, float)) and inferred_error < 2.0:
                        scale_steps = analysis_data.get('scale_steps')
                        if scale_steps and len(scale_steps) > 0:
                            inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                            use_inferred_system = True
                            scl_system_name = f"Auto-inferred ({tuning_inferred.get('type', 'Unknown')})"

                # If inferred system wasn't used, check Scala match
                if not use_inferred_system:
                    scala_match = analysis_data.get('scala_match_info')
                    if scala_match and isinstance(scala_match, dict):
                        scala_error = scala_match.get('avg_cents_error', float('inf'))
                        if isinstance(scala_error, (int, float)) and scala_error < 2.0:
                            scala_steps = analysis_data.get('scala_match_steps')
                            if scala_steps and len(scala_steps) > 0:
                                inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                                use_inferred_system = True
                                scl_system_name = f"Auto-Scala ({scala_match.get('name', 'Unknown')})"

            if use_inferred_system and inferred_ratios:
                scl_ratios = inferred_ratios

            # Use estimated diapason from analysis if available (unless user specified one)
            est_diapason = analysis_data.get('diapason_est')
            if (not user_diapason_set) and (
                    est_diapason and isinstance(est_diapason, (int, float)) and est_diapason > 0):
                scl_diapason = float(est_diapason)

        tun_csd.write_scl_file(export_base, scl_ratios, str(basenote), scl_diapason, scl_system_name)

    # Export ASCL file if requested
    if args.export_ableton:
        # Use diapason analysis ratios if available, otherwise use default system ratios
        ascl_ratios = ratios_eff

        # Apply same diapason logic as SCL export
        ascl_diapason = args.diapason  # Default
        if audio_params.get('diapason_analysis') and analysis_data:
            est_diapason = analysis_data.get('diapason_est')
            if est_diapason and isinstance(est_diapason, (int, float)) and est_diapason > 0 and not user_diapason_set:
                ascl_diapason = float(est_diapason)
                print(f"ASCL: Using estimated diapason from analysis: {est_diapason:.2f} Hz")
            elif user_diapason_set:
                print(f"ASCL: Using user-specified diapason: {args.diapason:.2f} Hz")

        # Generate system name based on the selected options (same logic as SCL)
        if hasattr(args, 'geometric') and args.geometric:
            ascl_system_name = "Geometric progression"
        elif hasattr(args, 'natural') and args.natural:
            ascl_system_name = "Natural intonation"
        elif hasattr(args, 'danielou') and args.danielou:
            ascl_system_name = "Danielou microtonal system"
        elif hasattr(args, 'pythagorean') and args.pythagorean:
            ascl_system_name = "Pythagorean"
        elif hasattr(args, 'just') and args.just:
            ascl_system_name = "Just intonation"
        elif hasattr(args, 'et') and args.et:
            # args.et might be a list [divisions, interval], so extract the divisions
            et_divisions = args.et[0] if isinstance(args.et, list) else args.et
            ascl_system_name = f"Equal Temperament {et_divisions}-TET"
        else:
            ascl_system_name = "Custom Scale"

        # Handle diapason analysis system selection (inferred vs Scala) - same logic as SCL export
        if audio_params.get('diapason_analysis') and analysis_data:
            # Check user preference for system choice
            force_inferred = getattr(args, 'inferred', False)
            force_scala = getattr(args, 'scala', False)

            use_inferred_system = False
            inferred_ratios = None

            if force_inferred:
                # User explicitly wants inferred system (same as SCL export)
                tuning_inferred = analysis_data.get('tuning_inferred')
                if tuning_inferred and isinstance(tuning_inferred, dict):
                    scale_steps = analysis_data.get('scale_steps')
                    if scale_steps and len(scale_steps) > 0:
                        inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                        use_inferred_system = True
                        ascl_system_name = f"Inferred ({tuning_inferred.get('type', 'Unknown')})"
                    else:
                        print("Warning: Inferred system requested but no scale_steps available for ASCL export")
                else:
                    print("Warning: Inferred system requested but no tuning_inferred data available for ASCL export")

            elif force_scala:
                # User explicitly wants Scala match (same as SCL export)
                scala_match = analysis_data.get('scala_match_info')
                if scala_match and isinstance(scala_match, dict):
                    scala_steps = analysis_data.get('scala_match_steps')
                    if scala_steps and len(scala_steps) > 0:
                        inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                        use_inferred_system = True
                        ascl_system_name = f"Scala match ({scala_match.get('name', 'Unknown')})"
                    else:
                        print("Warning: Scala match requested but no scala_match_steps available for ASCL export")
                else:
                    print("Warning: Scala match requested but no scala_match_info available for ASCL export")

            else:
                # Automatic selection based on < 2 cents error (same logic as SCL/TUN)
                tuning_inferred = analysis_data.get('tuning_inferred')
                if tuning_inferred and isinstance(tuning_inferred, dict):
                    inferred_error = tuning_inferred.get('avg_cents_error', float('inf'))
                    if isinstance(inferred_error, (int, float)) and inferred_error < 2.0:
                        scale_steps = analysis_data.get('scale_steps')
                        if scale_steps and len(scale_steps) > 0:
                            inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                            use_inferred_system = True
                            ascl_system_name = f"Auto-inferred ({tuning_inferred.get('type', 'Unknown')})"

                # If inferred system wasn't used, check Scala match
                if not use_inferred_system:
                    scala_match = analysis_data.get('scala_match_info')
                    if scala_match and isinstance(scala_match, dict):
                        scala_error = scala_match.get('avg_cents_error', float('inf'))
                        if isinstance(scala_error, (int, float)) and scala_error < 2.0:
                            scala_steps = analysis_data.get('scala_match_steps')
                            if scala_steps and len(scala_steps) > 0:
                                inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                                use_inferred_system = True
                                ascl_system_name = f"Auto-Scala ({scala_match.get('name', 'Unknown')})"

            if use_inferred_system and inferred_ratios:
                ascl_ratios = inferred_ratios

            # Use estimated diapason from analysis if available (unless user specified one)
            est_diapason = analysis_data.get('diapason_est')
            if (not user_diapason_set) and (
                    est_diapason and isinstance(est_diapason, (int, float)) and est_diapason > 0):
                ascl_diapason = float(est_diapason)

        tun_csd.write_ascl_file(export_base, ascl_ratios, str(basenote), ascl_diapason, ascl_system_name,
                                analysis_data=analysis_data)

    # When --diapason-analysis is active, also generate a cpstun table (.csd) using the same
    # selection logic as for .tun/.scl (inferred/Scala/raw analysis ratios and estimated basenote),
    # so that the .csd reflects the analysis outcome as requested.
    if audio_params.get('diapason_analysis') and analysis_data:
        csd_ratios = ratios
        csd_basekey_out = basekey_eff
        csd_basenote_out = basenote

        # Reuse the selection logic from TUN export
        force_inferred = getattr(args, 'inferred', False)
        force_scala = getattr(args, 'scala', False)

        use_inferred_system = False
        inferred_ratios = None

        if force_inferred:
            tuning_inferred = analysis_data.get('tuning_inferred')
            if tuning_inferred and isinstance(tuning_inferred, dict):
                scale_steps = analysis_data.get('scale_steps')
                if scale_steps and len(scale_steps) > 0:
                    inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                    use_inferred_system = True
                else:
                    print("Warning: Inferred system requested but no scale_steps available for CSD export")
            else:
                print("Warning: Inferred system requested but no tuning_inferred data available for CSD export")
        elif force_scala:
            scala_match = analysis_data.get('scala_match_info')
            if scala_match and isinstance(scala_match, dict):
                scala_steps = analysis_data.get('scala_match_steps')
                if scala_steps and len(scala_steps) > 0:
                    inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                    use_inferred_system = True
                else:
                    print("Warning: Scala match requested but no scala_match_steps available for CSD export")
            else:
                print("Warning: Scala match requested but no scala_match_info available for CSD export")
        else:
            tuning_inferred = analysis_data.get('tuning_inferred')
            if tuning_inferred and isinstance(tuning_inferred, dict):
                inferred_error = tuning_inferred.get('avg_cents_error', float('inf'))
                if isinstance(inferred_error, (int, float)) and inferred_error < 2.0:
                    scale_steps = analysis_data.get('scale_steps')
                    if scale_steps and len(scale_steps) > 0:
                        inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scale_steps]
                        use_inferred_system = True
            if not use_inferred_system:
                scala_match = analysis_data.get('scala_match_info')
                if scala_match and isinstance(scala_match, dict):
                    scala_error = scala_match.get('avg_cents_error', float('inf'))
                    if isinstance(scala_error, (int, float)) and scala_error < 2.0:
                        scala_steps = analysis_data.get('scala_match_steps')
                        if scala_steps and len(scala_steps) > 0:
                            inferred_ratios = [float(ratio) for (_idx, ratio, _count) in scala_steps]
                            use_inferred_system = True

        if use_inferred_system and inferred_ratios:
            csd_ratios = inferred_ratios
        else:
            # Fallback to raw analysis ratios
            diapason_ratios = analysis_data.get('ratios_reduced')
            if diapason_ratios and len(diapason_ratios) > 0:
                csd_ratios = diapason_ratios

        # Prefer estimated basenote/basekey if available
        est_basenote_hz = analysis_data.get('basenote_est_hz')
        if est_basenote_hz and isinstance(est_basenote_hz, (int, float)) and est_basenote_hz > 0:
            csd_basenote_out = float(est_basenote_hz)
        est_basenote_midi = analysis_data.get('basenote_est_midi')
        if est_basenote_midi and isinstance(est_basenote_midi, int) and 0 <= est_basenote_midi <= 127:
            csd_basekey_out = est_basenote_midi

        # Append a new cpstun table reflecting the analysis-based system
        tun_csd.write_cpstun_table(export_base, csd_ratios, csd_basekey_out, csd_basenote_out, interval_value=None)


if __name__ == "__main__":
    main()
