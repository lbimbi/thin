"""Audio analysis utilities for musical intonation systems analysis.

This module provides comprehensive audio analysis capabilities for THIN, supporting:

Core Analysis Features:
- F0 (fundamental frequency) tracking using pYIN/YIN algorithms or CREPE (high quality)
- Formant analysis via multiple methods: LPC, spectral envelope, SFFT, or CQT
- Diapason (A4) estimation from audio signals with uncertainty quantification
- Pitch ratio calculation and clustering for scale inference

Scale Matching and Inference:
- Automatic tuning system detection from audio-derived ratios
- Scala (.scl) scale database matching with Top-5 ranking system
- Built-in pattern matching for: Equal Temperament, Pythagorean, Just Intonation
- Circular distance metrics in cents space for robust matching
- Refined step matching and threshold-based filtering

Audio Processing Pipeline:
- Multi-frame analysis with configurable overlap and windowing
- Robust frequency clustering with 2% relative tolerance
- Cross-validation between F0 tracking and formant analysis
- Statistical analysis of pitch distributions and stability

Export and Visualization:
- PNG plot generation for diapason analysis and CQT spectrograms
- Audio rendering of analysis results (both analysis-following and constant tones)
- Integration with Excel export for detailed analysis reports
- Diagnostic plots showing F0 tracking over time

Dependencies:
- librosa and scipy for core audio processing
- matplotlib for visualization (optional)
- crepe and tensorflow for high-quality F0 extraction (optional)

Note: Requires manual download of Scala database from Huygens-Fokker website.
Historical DaMuSc integrations have been removed; scale matching now relies on Scala files only.
"""
from typing import List, Tuple, Optional
import logging
import math
import utils
import os
import consts
import tun_csd
import functools

# Pre-import commonly used modules to avoid repeated utils.ensure_* calls
_np = utils.ensure_numpy_import()
_lb = utils.ensure_librosa_import()
_warnings = utils.ensure_warnings_import()

# Caching for performance optimization
@functools.lru_cache(maxsize=128)

# Global cache for Scala scales to avoid repeated file I/O

# Cache for scale matching results to avoid recomputation
@functools.lru_cache(maxsize=64)
def _cached_scala_match(cluster_centers_hash: str, dir_path: str) -> Tuple[Optional[dict], List[Tuple[int, float, int]]]:
    """Cached version of scala scale matching."""
    # Convert hash back to cluster centers for processing
    import json
    cluster_centers = json.loads(cluster_centers_hash)
    return _match_scala_scales_uncached(cluster_centers, dir_path)

@functools.lru_cache(maxsize=32)
def _cached_scala_topk_match(cluster_centers_hash: str, dir_path: str, k: int) -> List[Tuple[dict, List[Tuple[int, float, int]]]]:
    """Cached version of top-k scala scale matching."""
    import json
    cluster_centers = json.loads(cluster_centers_hash)
    return _match_scala_scales_topk_uncached(cluster_centers, dir_path, k)



def _create_candidate_dict(name: str, family: str, err: float, mp: list, params: dict) -> dict:
    """Create standardized candidate dictionary."""
    return {"name": name, "family": family, "err": err, "map": mp, "params": params}


def _process_spectrum_for_formants(spectrum_data, freqs_axis, f0_hz):
    """Process spectrum data and extract formants, with common validation."""
    if spectrum_data.size == 0:
        return {"f0_hz": f0_hz, "formants": []}
    mag = _np.mean(spectrum_data, axis=1)
    if mag.size == 0:
        return {"f0_hz": f0_hz, "formants": []}
    return mag, freqs_axis


def _levinson_durbin(autocorr):
    """Levinson-Durbin algorithm for LPC coefficient computation."""
    if not _np:
        return None

    try:
        n = len(autocorr) - 1
        if n <= 0 or autocorr[0] == 0:
            return None

        # Initialize
        lpc_coeffs = _np.zeros(n)
        error = autocorr[0]

        for i in range(n):
            # Reflection coefficient
            if error == 0:
                break

            reflection = autocorr[i + 1]
            for j in range(i):
                reflection -= lpc_coeffs[j] * autocorr[i - j]
            reflection /= error

            # Update LPC coefficients
            lpc_coeffs[i] = reflection
            for j in range(i):
                lpc_coeffs[j] -= reflection * lpc_coeffs[i - 1 - j]

            # Update error
            error *= (1 - reflection * reflection)
            if error <= 0:
                break

        return lpc_coeffs
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _lpc_to_formants(lpc_coeffs, sr):
    """Extract formant frequencies from LPC coefficients."""
    if not _np:
        return []

    try:
        # Create polynomial from LPC coefficients
        # LPC polynomial: A(z) = 1 - a1*z^(-1) - a2*z^(-2) - ... - ap*z^(-p)
        poly_coeffs = _np.concatenate(([1], -lpc_coeffs))

        # Find roots of the polynomial
        roots = _np.roots(poly_coeffs)

        # Keep only complex roots inside unit circle
        valid_roots = []
        for root in roots:
            if _np.abs(root) < 1.0 and _np.imag(root) > 0:  # Upper half of unit circle
                valid_roots.append(root)

        # Convert roots to formant frequencies
        formants = []
        for root in valid_roots:
            # Angle of the complex root gives frequency
            angle = _np.angle(root)
            freq = angle * sr / (2 * _np.pi)
            if 50.0 <= freq <= sr / 2.0:  # Reasonable frequency range
                formants.append(freq)

        return sorted(formants)
    except (ValueError, RuntimeError, TypeError):
        return []


def _extract_f0_values(y, sr):
    """Extract F0 values using CREPE only.
    Returns (f0_vals, time_crepe, confidence_values) tuple.
    If CREPE is unavailable, returns (empty_array, None, None).
    """
    if not _np or not _lb:
        return utils.empty_float_array(), None, None
    try:
        import crepe as _cr
        import warnings as crepe_warnings
        # Suppress TensorFlow and CREPE warnings
        utils.suppress_tensorflow_warnings()
        crepe_warnings.filterwarnings('ignore', category=UserWarning, module='crepe.*')
    except ImportError:
        return utils.empty_float_array(), None, None

    try:
        # CREPE expects 16kHz
        y16 = _lb.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        time_c, freq_c, conf_c, _act = _cr.predict(y16, 16000, viterbi=True, step_size=10,
                                                   model_capacity='full', verbose=0)
        # Filter by confidence (keep both frequencies and confidences)
        mask = conf_c > 0.5
        f0_vals = _np.asarray(freq_c[mask], dtype=float)
        conf_vals = _np.asarray(conf_c[mask], dtype=float)
        return f0_vals, time_c, conf_vals
    except (ValueError, RuntimeError, TypeError) as e:
        logging.getLogger('warnings').warning(f"F0 extraction failed: {e}")
        return utils.empty_float_array(), None, None
    except ImportError as e:
        logging.getLogger('warnings').warning(f"CREPE module not available: {e}")
        return utils.empty_float_array(), None, None


def _process_f0_values_with_outlier_filtering(f0_vals):
    """Process F0 values with outlier filtering and return median value.
    Returns processed f0_hz or None if no valid values.
    """
    if not _np or f0_vals.size == 0:
        return None

    arr = f0_vals[_np.isfinite(f0_vals)]
    if arr.size == 0:
        return None

    med = _np.median(arr)
    mad = _np.median(_np.abs(arr - med))
    if mad > 0:
        sel = _np.abs(arr - med) <= (3.0 * mad)
        arr_f = arr[sel]
        if arr_f.size == 0:
            arr_f = arr
    else:
        lo = _np.percentile(arr, 5.0)
        hi = _np.percentile(arr, 95.0)
        arr_f = arr[(arr >= lo) & (arr <= hi)] if hi > lo else arr
    return float(_np.median(arr_f))


# Moved to utils.fold_frequency_to_octave and utils.circular_difference


def _score_tuning_candidate(steps_cents: List[float], centers: List[Tuple[float, int]]) -> Tuple[float, List[Tuple[int, float, int]]]:
    """Score a tuning system candidate against cluster centers.
    Returns (weighted_error_average, mapping) where mapping = [(index, ratio, count)].
    """
    if not steps_cents or not centers:
        return float('inf'), []

    total_werr = 0.0
    total_w = 0
    mapping: List[Tuple[int, float, int]] = []

    for (ci, cnt) in centers:
        try:
            # Find closest step using helper function
            best_idx, best_err = utils.find_closest_circular_index(ci, steps_cents, 1200.0)

            if best_idx is not None and best_err < float('inf'):
                w = int(cnt)
                total_werr += best_err * w
                total_w += w
                ratio = 2.0 ** (steps_cents[best_idx] / 1200.0)
                mapping.append((best_idx, ratio, int(cnt)))
        except (ValueError, TypeError, ArithmeticError):
            continue

    avg = (total_werr / total_w) if total_w > 0 else float('inf')

    # Debug: check for suspiciously low errors
    if 0 < avg < 0.001:
        print(f"DEBUG: Suspiciously low error {avg:.6f} for scale with {len(mapping)} mappings")
        print(f"DEBUG: total_werr={total_werr:.6f}, total_w={total_w}")

    return avg, mapping


def analyze_audio(audio_path: str,
                  method: str = "cqt",
                  frame_size: int = 1024,
                  hop_size: int = 512,
                  window_type: str = "hamming",
                  cqt_n_bins: Optional[int] = None,
                  cqt_bins_per_octave: Optional[int] = None,
                  cqt_fmin: Optional[float] = None,
                  f0_only: bool = False,
                  lpc_order: Optional[int] = None,
                  lpc_preemph: float = 0.97,
                  lpc_window_ms: Optional[float] = None) -> Optional[dict]:
    """Analizza un file WAV per F0 e formanti usando librosa.
    Ritorna un dict: { 'f0_hz': float|None, 'formants': [(freq_hz, rel_amp_0_1), ...] }
    If librosa is unavailable or analysis fails, returns None.
    """
    try:
        # Using pre-imported modules
        try:
            from scipy.signal import find_peaks as _find_peaks  # optional
        except (ImportError, ModuleNotFoundError):
            _find_peaks = None
    except (ImportError, ModuleNotFoundError) as e:
        print(f"librosa unavailable ({e}): install 'librosa' for audio analysis.",
              f"librosa not available ({e}): install 'librosa' for audio analysis.")
        return None

    result = utils.load_mono_audio(audio_path, verbose=True)
    if result is None:
        return None
    y, sr = result

    # Prepare window function
    win_type_in = (window_type or "hamming").lower()
    if win_type_in not in ("hamming", "hanning"):
        win_type_in = "hamming"
    win = _lb.filters.get_window("hann" if win_type_in == "hanning" else win_type_in, frame_size, fftbins=True)
    # Pitch estimation: CREPE only
    f0_hz: Optional[float] = None
    f0_vals = utils.empty_float_array()
    time_crepe = None
    confidence_mask = None
    try:
        fmin = 50.0
        fmax = max(2000.0, sr / 4.0)
        # Auto-adapt fmin so at least two periods of fmin fit into the frame
        fmin_eff, fmax_eff = utils.calculate_f0_params(sr, frame_size, fmin, fmax)
        if fmin_eff > fmin:
            pass
            # print(L(
            #    f"Adatto automaticamente fmin da {fmin:.3f} a {fmin_eff:.3f} Hz per frame_length={frame_size}",
            #    f"Auto-adapting fmin from {fmin:.3f} to {fmin_eff:.3f} Hz for frame_length={frame_size}"
            # ))
        # Use the effective fmin
        fmin = fmin_eff
        
        # Extract F0 values using CREPE-only helper
        f0_vals, time_crepe, confidence_mask = _extract_f0_values(y, sr)
        f0_hz = _process_f0_values_with_outlier_filtering(f0_vals)
    except ValueError:
        f0_hz = None
        time_crepe = None
        confidence_mask = None

    formant_freqs: list = []
    formant_amps: list = []

    def extract_formants_from_spectrum(mag_spectrum, freqs_axis_vals, max_peaks=10):
        """Helper to extract formants from magnitude spectrum."""
        nonlocal formant_freqs, formant_amps
        formant_freqs, formant_amps = utils.extract_peaks_from_spectrum(
            mag_spectrum, freqs_axis_vals, max_peaks=max_peaks, sr=sr)

    try:
        # Formant analysis based on method
        cqt_data = None  # Store CQT data for potential plot generation
        cqt_freqs = None
        cqt_times = None
        cqt_params = None

        if method == "lpc":
            # Linear Predictive Coding analysis
            try:
                # Auto-calculate LPC parameters if not provided
                if lpc_order is None:
                    # Rule of thumb: order = 2 + sr/1000
                    lpc_order = max(8, min(50, int(2 + sr / 1000)))

                # Calculate window length in samples
                if lpc_window_ms is None:
                    # Default: use frame_size converted to milliseconds
                    window_length_samples = frame_size
                else:
                    window_length_samples = int(lpc_window_ms * sr / 1000.0)
                    window_length_samples = max(frame_size, window_length_samples)  # Ensure minimum size

                # Apply pre-emphasis filter
                y_preemph = _np.append(y[0], y[1:] - lpc_preemph * y[:-1])

                # Frame the signal
                n_frames = max(1, (len(y_preemph) - window_length_samples) // hop_size + 1)

                all_formants = []
                for frame_idx in range(n_frames):
                    start = frame_idx * hop_size
                    end = start + window_length_samples
                    if end > len(y_preemph):
                        break

                    frame = y_preemph[start:end]
                    if len(frame) < lpc_order + 1:
                        continue

                    # Apply window
                    window = _np.hanning(len(frame))
                    windowed_frame = frame * window

                    # Compute LPC coefficients using autocorrelation (Levinson-Durbin)
                    try:
                        # Autocorrelation
                        autocorr = _np.correlate(windowed_frame, windowed_frame, mode='full')
                        autocorr = autocorr[len(autocorr)//2:]

                        if len(autocorr) < lpc_order + 1:
                            continue

                        # Levinson-Durbin algorithm
                        lpc_coeffs = _levinson_durbin(autocorr[:lpc_order + 1])

                        if lpc_coeffs is None or len(lpc_coeffs) < lpc_order:
                            continue

                        # Find formants from LPC coefficients
                        formant_freqs_frame = _lpc_to_formants(lpc_coeffs, sr)
                        all_formants.extend(formant_freqs_frame)

                    except (ValueError, RuntimeError, ZeroDivisionError):
                        continue

                # Process collected formants
                if all_formants:
                    # Remove duplicates and sort
                    unique_formants = list(set(all_formants))
                    unique_formants.sort()

                    # Keep reasonable formant frequencies (50 Hz to Nyquist/2)
                    valid_formants = [f for f in unique_formants if 50.0 <= f <= sr/4.0]

                    # Assign equal amplitudes (LPC doesn't provide amplitude info directly)
                    if valid_formants:
                        formant_freqs = valid_formants[:12]  # Limit to 12 formants
                        formant_amps = [1.0] * len(formant_freqs)

            except (ValueError, RuntimeError, ImportError) as e:
                print(f"LPC analysis failed ({e}), falling back to STFT")
                # Fallback to STFT
                stft_result = _lb.stft(y, n_fft=frame_size, hop_length=hop_size, window=win)
                s = _np.abs(stft_result)
                result = _process_spectrum_for_formants(s, None, f0_hz)
                if isinstance(result, dict):
                    return result
                mag, _ = result
                freqs_axis = _np.linspace(0.0, sr / 2.0, num=mag.shape[0])
                extract_formants_from_spectrum(mag, freqs_axis, max_peaks=10)

        elif method == "cqt":
            # Constant-Q analysis
            try:
                fmin = float(cqt_fmin) if (cqt_fmin is not None and float(cqt_fmin) > 0) else 55.0
                bins_per_octave = int(cqt_bins_per_octave) if (cqt_bins_per_octave is not None and int(cqt_bins_per_octave) > 0) else 48
                n_bins = int(cqt_n_bins) if (cqt_n_bins is not None and int(cqt_n_bins) > 0) else int(_np.ceil(bins_per_octave * _np.log2((sr / 2.0) / fmin)))

                # Validate CQT parameters to avoid Nyquist frequency errors
                max_freq = fmin * (2.0 ** (n_bins / bins_per_octave))
                nyquist_freq = sr / 2.0
                if max_freq >= nyquist_freq:
                    # Adjust parameters to avoid exceeding Nyquist
                    n_bins = int(_np.floor(bins_per_octave * _np.log2(nyquist_freq * 0.95 / fmin)))
                    if n_bins <= 0:
                        raise ValueError("Invalid CQT parameters")
                    print(f"\nCQT parameters adjusted: n_bins={n_bins} to avoid Nyquist frequency limit")

                if n_bins <= 0:
                    return {"f0_hz": f0_hz, "formants": []}
                cqt_result = _lb.cqt(y, sr=sr, hop_length=hop_size, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
                c = _np.abs(cqt_result)

                # Store CQT data for plot generation
                cqt_data = c
                cqt_freqs = _lb.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
                cqt_times = _lb.times_like(c, sr=sr, hop_length=hop_size)
                cqt_params = {
                    'fmin': float(fmin),
                    'bins_per_octave': int(bins_per_octave),
                    'n_bins': int(n_bins),
                    'hop_length': int(hop_size)
                }

                result = _process_spectrum_for_formants(c, None, f0_hz)
                if isinstance(result, dict):
                    return result
                mag, _ = result
                freqs_axis = _lb.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
                if _find_peaks is not None:
                    peaks, _ = _find_peaks(mag, distance=max(1, int(bins_per_octave / 2)))
                    if peaks.size > 12:
                        idx = _np.argsort(mag[peaks])[-12:]
                        peaks = peaks[idx]
                else:
                    peaks = _np.argpartition(mag, -12)[-12:]
                extract_formants_from_spectrum(mag, _np.array(freqs_axis), max_peaks=12)
            except ValueError as e:
                print(f"CQT analysis failed ({e}), falling back to STFT")
            except (RuntimeError, ImportError) as e:
                print(f"CQT analysis failed ({e}), falling back to STFT")
                # Fallback to STFT if CQT fails
                stft_result = _lb.stft(y, n_fft=frame_size, hop_length=hop_size, window=win)
                s = _np.abs(stft_result)
                result = _process_spectrum_for_formants(s, None, f0_hz)
                if isinstance(result, dict):
                    return result
                mag, _ = result
                freqs_axis = _np.linspace(0.0, sr / 2.0, num=mag.shape[0])
                extract_formants_from_spectrum(mag, freqs_axis, max_peaks=10)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Audio analysis error: {e}")
        return None

    # Normalize amplitudes 0..1
    pairs = []
    if formant_freqs:
        maxamp = max(formant_amps) if formant_amps else 1.0
        if maxamp <= 0:
            maxamp = 1.0
        for f, a in zip(formant_freqs, formant_amps):
            pairs.append((float(f), max(0.0, min(1.0, float(a) / maxamp))))
    # Build f0_times if possible
    f0_times = []
    if time_crepe is not None and isinstance(time_crepe, (_np.ndarray, list)):
        # CREPE path with optional confidence mask
        time_array = _np.asarray(time_crepe)
        if confidence_mask is not None and isinstance(confidence_mask, _np.ndarray) and hasattr(confidence_mask, 'shape') and confidence_mask.shape[0] == time_array.shape[0]:
            time_masked = time_array[confidence_mask]
        else:
            time_masked = time_array
        f0_times = [float(t) for t in time_masked.tolist()[:len(f0_vals)]]
    else:
        # Estimate frame times from hop_size and sr using librosa
        if isinstance(sr, (int, float)) and sr > 0:
            n_frames = int(_np.ceil(len(y) / float(hop_size))) if hop_size and hop_size > 0 else 0
            if n_frames > 0:
                t = _lb.times_like(_np.zeros(n_frames), sr=sr, hop_length=hop_size)
                if len(t) > 0:
                    # simple resample to match f0_vals length
                    idx = _np.linspace(0, max(1, len(t) - 1), num=len(f0_vals)) if len(f0_vals) > 0 else []
                    f0_times = [float(t[int(round(i))]) for i in idx] if len(f0_vals) > 0 else []

    # Heuristic uncertainty estimates for sfft/specenv/cqt methods
    freq_resolution_hz = None
    freq_uncert_hz = None  # Initialize with default value
    frames_used_count = 1
    frames_used = 1  # Initialize with default value
    ratio_uncert_typical = None
    suggested_ratio_decimals = None
    try:
        m_eff = (method or "cqt").lower()
        # Estimate frames used from hop_size and signal length
        if isinstance(hop_size, int) and hop_size > 0 and isinstance(sr, (int, float)) and sr > 0:
            frames_used = int(_np.ceil(len(y) / float(hop_size)))
            if frames_used <= 0:
                frames_used = 1
        else:
            frames_used = 1
        if m_eff == "sfft":
            # Recompute n_fft heuristic like above branch
            try:
                nfft_est = int(2 ** _np.ceil(_np.log2(max(frame_size * 4, len(y))))) if len(y) > 0 else frame_size
            except ValueError:
                nfft_est = frame_size
            nfft_est = max(2048, min(int(nfft_est), 1 << 18))
            if isinstance(sr, (int, float)) and sr > 0 and nfft_est > 0:
                freq_resolution_hz = float(sr) / float(nfft_est)
        elif m_eff in ("specenv",):
            if isinstance(sr, (int, float)) and isinstance(frame_size, int) and frame_size > 0:
                freq_resolution_hz = float(sr) / float(frame_size)
        elif m_eff == "cqt":
            # Constant-Q: Δf ≈ f/Q, Q = 1/(2^(1/B) - 1), B ~ 48 here
            try:
                bpo = 48.0
                q_factor = 1.0 / (float(2.0 ** (1.0 / bpo)) - 1.0)
                # Representative frequency: median of selected peaks or 440 Hz
                if pairs:
                    f_med = utils.safe_median_from_floats([p[0] for p in pairs])
                elif isinstance(f0_hz, (int, float)) and f0_hz:
                    f_med = float(f0_hz)
                else:
                    f_med = 440.0
                freq_resolution_hz = float(f_med) / float(q_factor)
            except (TypeError, ValueError, ZeroDivisionError):
                freq_resolution_hz = None
        # Compute effective frequency uncertainty with sub-bin interpolation heuristic
        if isinstance(freq_resolution_hz, (int, float)) and freq_resolution_hz > 0:
            subbin = float(freq_resolution_hz) / 10.0
            denom = math.sqrt(max(1.0, float(frames_used)))
            freq_uncert_hz = subbin / denom
        else:
            freq_uncert_hz = None
        # Typical ratio uncertainty using representative frequencies
        try:
            if isinstance(freq_uncert_hz, (int, float)) and freq_uncert_hz > 0:
                if pairs:
                    f_rep = utils.safe_median_from_floats([p[0] for p in pairs])
                elif isinstance(f0_hz, (int, float)) and f0_hz > 0:
                    f_rep = float(f0_hz)
                else:
                    f_rep = None
                if isinstance(f_rep, float) and f_rep > 0:
                    f_ref = float(f0_hz) if isinstance(f0_hz, (int, float)) and f0_hz > 0 else f_rep
                    ratio_uncert_typical = math.sqrt((freq_uncert_hz / f_rep) ** 2 + (freq_uncert_hz / f_ref) ** 2)
                    # Suggest decimals
                    if ratio_uncert_typical > 0 and math.isfinite(ratio_uncert_typical):
                        d_raw = math.ceil(max(0.0, -math.log10(float(ratio_uncert_typical) * 2.0)))
                        suggested_ratio_decimals = int(max(2, min(8, d_raw)))
        except (TypeError, ValueError):
            ratio_uncert_typical = None
            suggested_ratio_decimals = None
    except (TypeError, ValueError):
        freq_uncert_hz = None
        ratio_uncert_typical = None
        suggested_ratio_decimals = None

    return {
        "f0_hz": f0_hz,
        "formants": pairs,
        "f0_series": [float(x) for x in (f0_vals.tolist() if 'f0_vals' in locals() else [])],
        "f0_times": f0_times,
        # Provenance
        "audio_path": str(audio_path) if isinstance(audio_path, str) else None,
        "sample_rate": int(sr) if isinstance(sr, (int, float)) else None,
        # CQT data for plotting
        "cqt_data": cqt_data,
        "cqt_freqs": cqt_freqs,
        "cqt_times": cqt_times,
        "cqt_params": cqt_params,
        # Uncertainty additions (heuristic)
        "freq_resolution_hz": (float(freq_resolution_hz) if isinstance(freq_resolution_hz, (int, float)) else None),
        "freq_uncert_hz": (float(freq_uncert_hz) if isinstance(freq_uncert_hz, (int, float)) else None),
        "frames_used": int(frames_used) if isinstance(frames_used, int) else (int(frames_used) if frames_used else 0),
        "ratio_uncert_typical": (
            float(ratio_uncert_typical) if isinstance(ratio_uncert_typical, (int, float)) else None),
        "suggested_ratio_decimals": int(suggested_ratio_decimals) if isinstance(suggested_ratio_decimals,
                                                                                int) else None,
    }


def _analyze_cluster_bounds(centers: List[Tuple[float, int]]) -> Tuple[float, float]:
    """Analyze cluster data to determine smart generator bounds.
    Returns (suggested_min_generator, suggested_max_generator) in cents.
    """
    if not centers:
        return 20.0, 800.0  # fallback to defaults

    # Extract just the cents values
    cents_values = [c for c, _ in centers]
    if len(cents_values) < 2:
        return 20.0, 800.0

    # Calculate the average interval between adjacent clusters
    sorted_cents = sorted(cents_values)
    intervals = []
    for i in range(len(sorted_cents) - 1):
        interval = sorted_cents[i + 1] - sorted_cents[i]
        intervals.append(interval)

    if intervals:
        # Use the median interval as a guide for generator size
        import statistics
        median_interval = statistics.median(intervals)

        # Smart bounds based on data:
        # Min generator: half the median interval (but at least 15 cents)
        # Max generator: 3x the median interval (but no more than 900 cents)
        min_gen = max(15.0, median_interval * 0.5)
        max_gen = min(900.0, median_interval * 3.0)

        # Ensure reasonable bounds
        if max_gen <= min_gen:
            max_gen = min_gen + 50.0

        return min_gen, max_gen

    return 20.0, 800.0  # fallback




def infer_tuning_system_general(cluster_centers: List[Tuple[float, int]]) -> Tuple[Optional[dict], List[Tuple[int, float, int]]]:
    """
    Inferenza generale del sistema di intonazione a partire dai centroidi dei cluster (ratio, count).
    Restituisce (tuning_info: Optional[dict], scale_steps: List[Tuple[int, float, int]]).
    tuning_info: {'name': str, 'avg_cents_error': float, 'params': {...}}
    scale_steps: lista di tuple (indice_step, ratio in [1,2), count)
    Comments included.
    """
    try:
        import math as _math
    except ImportError:  # pragma: no cover
        _math = math

    # Converti cluster in cents [0,1200)
    centers = utils.process_cluster_centers_to_cents(cluster_centers)
    if not centers:
        return None, []

    # Using helper function for circular difference

    def score_candidate(steps_cents: List[float]) -> Tuple[float, List[Tuple[int, float, int]]]:
        """Evaluates a candidate list of steps (in cents, reduced to [0,1200)).
        Returns (weighted_average_error, mapping) where mapping = [(idx, ratio, count)]."""
        if not steps_cents:
            return float('inf'), []
        sc = sorted([(s % 1200.0) for s in steps_cents])
        return _score_tuning_candidate(sc, centers)

    best_name = None
    best_err = float('inf')
    best_map: List[Tuple[int, float, int]] = []
    best_params = {}

    # Known families (for continuity)
    tet12 = utils.get_tet_scale(12)
    err, mp = score_candidate(tet12)
    if err < best_err:
        best_name, best_err, best_map = "12-TET", err, mp
        best_params = {"n": 12}

    # Pitagorico 12 / Pyth-12
    p12_cents, p7_cents = utils.get_pythagorean_scales()
    err, mp = score_candidate(p12_cents)
    if err < best_err:
        best_name, best_err, best_map = "Pythagorean-12", err, mp
        best_params = {}

    # Pitagorico 7 / Pyth-7
    err, mp = score_candidate(p7_cents)
    if err < best_err:
        best_name, best_err, best_map = "Pythagorean-7", err, mp
        best_params = {}

    # n-TET generico / General n-TET (5..72)
    for n in range(5, 73):
        steps = utils.get_tet_scale(n)
        err, mp = score_candidate(steps)
        if err < best_err:
            best_name, best_err, best_map = f"{n}-TET", err, mp
            best_params = {"n": n}

    # Rank-1 con generatore / Rank-1 generated with free generator
    periods = [1200.0, 1901.955000865, 2400.0]  # octave, tritave, double octave
    for P in periods:
        # Limit generator search with optimized bounds
        g_min = max(20.0, P / 50.0)  # Reduced from 72 to 50 for tighter bounds
        g_max = max(g_min + 10.0, min(800.0, P / 2.5))  # Reduced upper bound for faster search
        step_cents = 5.0  # 5 cent resolution for faster search
        g = g_min
        while g <= g_max + 1e-9:
            max_k = int(_math.floor(P / g)) + 1
            steps = [(k * g) % 1200.0 for k in range(max(1, max_k))]
            # Dedup near-equal cents
            steps_u = utils.deduplicate_near_equal_cents(steps)
            err, mp = score_candidate(steps_u)
            if err < best_err:
                best_name, best_err, best_map = (
                    f"Rank-1 gen={g:.2f}c period={P:.2f}c" if P != 1200.0 else f"Rank-1 gen={g:.2f}c"), err, mp
                best_params = {"generator_cents": round(g, 4), "period_cents": round(P, 4)}
            g += step_cents

    # Rank-2 (due generatori) con periodo ottava: insieme limitato di coppie comuni
    common_gens = [701.955000865, 386.313713865, 203.910001, 294.135, 968.826]
    pairs = []
    for i, gen_i in enumerate(common_gens):
        for j, gen_j in enumerate(common_gens):
            if i == j:
                continue
            g1 = float(gen_i) % 1200.0
            g2 = float(gen_j) % 1200.0
            # Evita coppie troppo simili (< 5c)
            if utils.circular_difference(g1, g2, 1200.0) < 5.0:
                continue
            pairs.append((g1, g2))

    # Limita il numero di coppie per economia
    pairs = pairs[:12]
    for (g1, g2) in pairs:
        # Costruisci un piccolo reticolo con indici limitati
        lattice_size = 6
        pts = []
        for coef_a in range(-lattice_size, lattice_size + 1):
            for coef_b in range(-lattice_size, lattice_size + 1):
                val = (coef_a * g1 + coef_b * g2) % 1200.0
                pts.append(val)
        # Dedup e ordina
        steps_u = utils.deduplicate_near_equal_cents(pts)
        # Se troppi gradi, prendi i primi 72 per coerenza di scoring
        if len(steps_u) > 72:
            steps_u = steps_u[:72]
        err, mp = score_candidate(steps_u)
        if err < best_err:
            best_name, best_err, best_map = f"Rank-2 gen1={g1:.2f}c gen2={g2:.2f}c", err, mp
            best_params = {"generator1_cents": round(g1, 4), "generator2_cents": round(g2, 4),
                           "period_cents": 1200.0}

    # Prepara output / Prepare output
    if best_name is None:
        return None, []

    # Normalizza: assicurati che esista step indice 0 a ratio 1.0
    steps_out = utils.normalize_mapping_with_zero_step(best_map)
    info = {"name": best_name, "avg_cents_error": float(best_err), "params": best_params}
    return info, steps_out


def infer_tuning_system_with_comparative(cluster_centers: List[Tuple[float, int]]) -> Tuple[Optional[dict], List[Tuple[int, float, int]], Optional[dict], List[Tuple[int, float, int]]]:
    """
    Come infer_tuning_system_general ma ritorna anche una soluzione comparativa
    di famiglia diversa: (primary_info, primary_steps, comp_info, comp_steps).
    Famiglie: 'TET', 'Pyth', 'Rank1', 'Rank2'.
    """
    try:
        import math as _math
    except ImportError:
        _math = math

    # Prepara centers in cents [0,1200)
    centers = utils.process_cluster_centers_to_cents(cluster_centers)
    if not centers:
        return None, [], None, []

    # Using helper function for circular difference

    def score_candidate(steps_cents: List[float]):
        if not steps_cents:
            return float('inf'), []
        sc = sorted([(st % 1200.0) for st in steps_cents])
        return _score_tuning_candidate(sc, centers)

    def norm_steps_out(map_in: List[Tuple[int, float, int]]):
        return utils.normalize_mapping_with_zero_step(map_in)

    cands = []  # list of dicts with name,family,err,map,params

    # TETs (including 12-TET)
    for n in range(5, 73):
        try:
            steps = utils.get_tet_scale(n)
            err, mp = score_candidate(steps)
            cands.append({"name": f"{n}-TET", "family": "TET", "err": err, "map": mp, "params": {"n": n}})
        except (ValueError, TypeError) as e:
            print(f"Error generating TET {n}: {e}")
            pass

    # Pythagorean families
    p12_cents, p7_cents = utils.get_pythagorean_scales()
    err, mp = score_candidate(p12_cents)
    cands.append({"name": "Pythagorean-12", "family": "Pyth", "err": err, "map": mp, "params": {}})
    err, mp = score_candidate(p7_cents)
    cands.append({"name": "Pythagorean-7", "family": "Pyth", "err": err, "map": mp, "params": {}})

    # Rank-1 generated systems (octave, tritave, double octave)
    periods = [1200.0, 1901.955000865, 2400.0]
    for P in periods:
        try:
            g_min = max(20.0, P / 50.0)  # Optimized bounds
            g_max = max(g_min + 10.0, min(800.0, P / 2.5))
            g = g_min
            while g <= g_max + 1e-9:
                try:
                    max_k = int(_math.floor(P / g)) + 1
                    steps = [(k * g) % 1200.0 for k in range(max(1, max_k))]
                    steps_u = utils.deduplicate_steps(steps)
                    err, mp = score_candidate(steps_u)
                    name = (f"Rank-1 gen={g:.2f}c period={P:.2f}c" if P != 1200.0 else f"Rank-1 gen={g:.2f}c")
                    params = {"generator_cents": round(g, 4), "period_cents": round(P, 4)}
                    cands.append(_create_candidate_dict(name, "Rank1", err, mp, params))
                except ValueError:
                    pass
                g += 5.0  # 5 cent resolution for faster search
        except ValueError:
            pass

    # Rank-2 (two generators) with octave period: small curated set
    try:
        common_gens = [701.955000865, 386.313713865, 203.910001, 294.135, 968.826]
        pairs = []
        for i in range(len(common_gens)):
            for j in range(len(common_gens)):
                if i == j:
                    continue
                g1 = float(common_gens[i]) % 1200.0
                g2 = float(common_gens[j]) % 1200.0
                # Skip near-equal generators
                if utils.circular_difference(g1, g2, 1200.0) < 5.0:
                    continue
                pairs.append((g1, g2))
        pairs = pairs[:12]
        for (g1, g2) in pairs:
            try:
                max_a = 6
                max_b = 6
                pts = []
                for a in range(-max_a, max_a + 1):
                    for b in range(-max_b, max_b + 1):
                        pts.append((a * g1 + b * g2) % 1200.0)
                steps_u = utils.deduplicate_steps(pts)
                if len(steps_u) > 72:
                    steps_u = steps_u[:72]
                err, mp = score_candidate(steps_u)
                name = f"Rank-2 gen1={g1:.2f}c gen2={g2:.2f}c"
                params = {"generator1_cents": round(g1, 4), "generator2_cents": round(g2, 4), "period_cents": 1200.0}
                cands.append(_create_candidate_dict(name, "Rank2", err, mp, params))
            except ValueError:
                pass
    except ValueError:
        pass

    if not cands:
        return None, [], None, []

    # Pick primary and complementary comparative
    primary = min(cands, key=lambda d: float(d.get("err", float('inf'))))
    comp = None
    fam = primary.get("family")
    # Complementary rule: TET <-> (Rank1|Rank2); Pyth -> TET (fallback: any different family)
    try:
        if fam == "TET":
            pool = [d for d in cands if d.get("family") in ("Rank1", "Rank2")]
        elif fam in ("Rank1", "Rank2"):
            pool = [d for d in cands if d.get("family") == "TET"]
        else:
            pool = [d for d in cands if d.get("family") == "TET"]
        if not pool:
            pool = [d for d in cands if d.get("family") != fam]
        comp = min(pool, key=lambda d: float(d.get("err", float('inf')))) if pool else None
    except ValueError:
        comp = None

    def _create_tuning_info(tuning_dict):
        return {"name": tuning_dict["name"], "avg_cents_error": float(tuning_dict["err"]), "params": tuning_dict.get("params", {})}

    prim_info = _create_tuning_info(primary)
    prim_steps = norm_steps_out(primary["map"])

    if comp is not None:
        comp_info = _create_tuning_info(comp)
        comp_steps = norm_steps_out(comp["map"])
    else:
        comp_info, comp_steps = None, []

    return prim_info, prim_steps, comp_info, comp_steps


# --- Scala (.scl) parsing and matching utilities ---
# IT: Parser minimale per file .scl (Scala) e matching con i centroidi.
# EN: Minimal parser for .scl (Scala) files and matching with cluster centers.


def _match_scala_scales_uncached(cluster_centers: List[Tuple[float, int]], dir_path: str = 'scl') -> Tuple[
    Optional[dict], List[Tuple[int, float, int]]]:
    """Trova la scala .scl più vicina ai centroidi (ratio,count).
    - Converte i centroidi in cents [0,1200) e per ciascuno trova il grado più vicino della scala
      usando distanza circolare modulo 1200.
    - Ritorna (best_info, steps_map) dove best_info include 'name','file','avg_cents_error'.
    - steps_map: [(index, ratio_from_degree, count)].
    Se nessuna scala è disponibile, ritorna (None, []).
    """
    scales = tun_csd.load_scales_from_dir(dir_path)
    if not scales:
        return None, []

    # Prepare centers in cents
    centers_c = utils.process_cluster_centers_to_cents(cluster_centers)
    if not centers_c:
        return None, []

    # Using helper function for circular difference

    best = None  # tuple(err, scale_info, mapping)
    for sc in scales:
        degs = sc.get('degrees_cents') or []
        if not degs:
            continue
        avg, mapping = utils.score_scale_against_centers(degs, centers_c)
        if best is None or avg < best[0]:
            best = (avg, sc, mapping)

    if best is None:
        return None, []

    err, sc, mapping = best
    info = {'name': sc.get('name', ''), 'file': sc.get('file', ''), 'avg_cents_error': float(err)}
    # Normalize mapping to ensure index 0 & ratio 1.0 present
    steps = utils.normalize_mapping_with_zero_step(mapping)
    return info, steps


def match_scala_scales(cluster_centers: List[Tuple[float, int]], dir_path: str = 'scl') -> Tuple[
    Optional[dict], List[Tuple[int, float, int]]]:
    """Cached wrapper for Scala scale matching."""
    import json
    cluster_centers_hash = json.dumps(cluster_centers, sort_keys=True)
    return _cached_scala_match(cluster_centers_hash, dir_path)


def match_scala_scales_topk(cluster_centers: List[Tuple[float, int]], dir_path: str = 'scl', k: int = 5) -> List[
    Tuple[dict, List[Tuple[int, float, int]]]]:
    """Cached wrapper for top-k Scala scale matching."""
    import json
    cluster_centers_hash = json.dumps(cluster_centers, sort_keys=True)
    return _cached_scala_topk_match(cluster_centers_hash, dir_path, k)


def _match_scala_scales_topk_uncached(cluster_centers: List[Tuple[float, int]], dir_path: str = 'scl', k: int = 5) -> List[
    Tuple[dict, List[Tuple[int, float, int]]]]:
    """Ritorna le migliori k scale .scl per i centroidi dati.
    Output: list of (info_dict, steps_map) sorted by ascending error.
    Mantiene la stessa logica di punteggio di match_scala_scales.
    """
    scales = tun_csd.load_scales_from_dir(dir_path)
    if not scales:
        return []
    # Prepara centers in cents
    centers_c = utils.process_cluster_centers_to_cents(cluster_centers)
    if not centers_c:
        return []

    # Using helper function for circular difference

    scored: List[Tuple[float, dict, List[Tuple[int, float, int]]]] = []
    for sc in scales:
        degs = sc.get('degrees_cents') or []
        if not degs:
            continue
        avg, mapping = utils.score_scale_against_centers(degs, centers_c)
        if avg == float('inf'):
            continue
        info = {'name': sc.get('name', ''), 'file': sc.get('file', ''), 'avg_cents_error': float(avg)}
        # Normalize mapping to ensure index 0 & ratio 1.0 present
        steps = utils.normalize_mapping_with_zero_step(mapping)
        scored.append((float(avg), info, steps))

    scored.sort(key=lambda x: float(x[0]))
    top = scored[:max(0, int(k))] if scored else []
    return [(info, steps) for (_err, info, steps) in top]


def match_scala_scales_within_threshold(cluster_centers: List[Tuple[float, int]], dir_path: str = 'scl',
                                        threshold_cents: float = 0.0) -> List[dict]:
    """Returns all .scl scales with average error <= threshold_cents.
    Output: list of dict {'name','file','avg_cents_error'} sorted by ascending error.
    """
    try:
        thr = float(threshold_cents)
    except ValueError:
        return []
    if not math.isfinite(thr) or thr < 0:
        return []
    # Riutilizza il punteggio come in topk
    scales = tun_csd.load_scales_from_dir(dir_path)
    if not scales:
        return []
    centers_c = utils.process_cluster_centers_to_cents(cluster_centers)
    if not centers_c:
        return []

    # Using helper function for circular difference

    scored: List[Tuple[float, dict]] = []
    for sc in scales:
        degs = sc.get('degrees_cents') or []
        if not degs:
            continue
        avg, _ = utils.score_scale_against_centers(degs, centers_c)
        if avg <= thr:
            scored.append(
                (float(avg), {'name': sc.get('name', ''), 'file': sc.get('file', ''), 'avg_cents_error': float(avg)}))
    scored.sort(key=lambda x: float(x[0]))
    return [info for (_err, info) in scored]


def match_refined_scale_steps(scale_steps: List[Tuple[int, float, int]], dir_path: str = 'scl') -> Tuple[
    Optional[dict], List[Tuple[int, float, int]], List[dict]]:
    """Match raffinato per scale_steps contro scale note.

    Usa SOLO i rapporti dai scale_steps (sottoinsieme inferito) per trovare
    la scala Scala che meglio rappresenta questi specifici gradi.

    Args:
        scale_steps: Lista di tuple (indice_step, ratio in [1,2), count) dai sistemi inferiti
        dir_path: Directory contenente i file .scl

    Returns:
        Tuple (best_match_info, matched_steps, top3_matches) dove:
        - best_match_info: dict con 'name', 'file', 'avg_cents_error', 'refined_match': True (best match)
        - matched_steps: lista dei scale_steps originali
        - top3_matches: lista di max 3 migliori match ordinati per score
    """
    if not scale_steps:
        return None, [], []

    # Estrai solo i ratio dai scale_steps e convertili in cents
    step_ratios = [ratio for (_idx, ratio, _count) in scale_steps]
    step_cents = []
    for ratio in step_ratios:
        if ratio > 0:
            cents = (1200.0 * math.log(ratio, 2.0)) % 1200.0
            step_cents.append(cents)

    if not step_cents:
        return None, [], []

    # Aggiungi sempre 0 (unisono) se non presente
    if 0.0 not in step_cents:
        step_cents.append(0.0)

    step_cents = sorted(set(step_cents))  # Rimuovi duplicati e ordina
    # Format step cents for better readability
    if len(step_cents) <= 18:
        cents_str = ' '.join([f'{c:.1f}' for c in step_cents])
    else:
        # For long lists, show first 15 and last 3
        first_15 = ' '.join([f'{c:.1f}' for c in step_cents[:15]])
        last_3 = ' '.join([f'{c:.1f}' for c in step_cents[-3:]])
        cents_str = f"{first_15} ... {last_3}"

    print(f"\n" + "="*60)
    print(f"Refined Matching with {len(step_cents)}-step scale")
    print("="*60)
    print(f"Step cents: {cents_str}")
    print("="*60)


    # Find the result dict in the calling context
    # This will be stored in the estimate_diapason_and_ratios result dict

    # Carica scale Scala
    scales = tun_csd.load_scales_from_dir(dir_path)
    if not scales:
        return None, scale_steps, []

    # Raccogli tutti i match validi
    valid_matches = []

    for scale in scales:
        scale_degrees = scale.get('degrees_cents', [])
        if not scale_degrees or len(scale_degrees) == 0:
            continue

        # Assicurati che la scala includa 0 (unisono)
        if 0.0 not in scale_degrees:
            scale_degrees = [0.0] + list(scale_degrees)

        scale_degrees = sorted(set(scale_degrees))

        # Per ogni step dei nostri scale_steps, trova il grado più vicino nella scala
        matched_pairs = []
        total_error = 0.0

        for step_c in step_cents:
            # Trova il grado più vicino nella scala
            min_dist = float('inf')
            closest_degree = None

            for deg_c in scale_degrees:
                # Distanza circolare modulo 1200
                dist = min(abs(step_c - deg_c), 1200.0 - abs(step_c - deg_c))
                if dist < min_dist:
                    min_dist = dist
                    closest_degree = deg_c

            # Accetta match solo se ragionevolmente vicino (< 30 cents)
            if min_dist < 30.0:
                matched_pairs.append((step_c, closest_degree, min_dist))
                total_error += min_dist

        if len(matched_pairs) > 0:
            avg_error = total_error / len(matched_pairs)
            coverage_ratio = len(matched_pairs) / len(step_cents)

            # Score che favorisce alto coverage e basso errore
            # Penalizza fortemente scale con coverage basso
            if coverage_ratio >= 0.5:  # Almeno 50% di coverage richiesto
                score = avg_error / (coverage_ratio ** 2)  # Favorisce coverage alto

                match_info = {
                    'name': scale.get('name', ''),
                    'file': scale.get('file', ''),
                    'avg_cents_error': avg_error,
                    'coverage_ratio': coverage_ratio,
                    'matched_steps': len(matched_pairs),
                    'total_steps': len(step_cents),
                    'score': score,
                    'refined_match': True
                }

                # Aggiungi alla lista se l'errore è ragionevole
                if avg_error < 25.0:
                    valid_matches.append(match_info)

    # Ordina per score e prendi i top 3
    valid_matches.sort(key=lambda x: x['score'])
    top3_matches = valid_matches[:3]

    # Il migliore è il primo della lista (se esiste)
    best_match = top3_matches[0] if top3_matches else None

    return best_match, scale_steps, top3_matches



def estimate_diapason_and_ratios(audio_path: str,
                                 base_hint_hz: float,
                                 initial_a4_hz: float = 440.0,
                                 frame_size: int = 1024,
                                 hop_size: int = 512,
                                 scala_cent: Optional[float] = None,
                                 digits: Optional[int] = None,
                                 f0_only: bool = False,
                                 cqt_analysis: Optional[dict] = None,
                                 formant_method: str = "cqt",
                                 lpc_order: Optional[int] = None,
                                 lpc_preemph: float = 0.97,
                                 lpc_window_ms: Optional[float] = None,
                                 tolerance_percent: float = 2.0,
                                 min_ratio_count: int = 1,
                                 confidence_threshold: float = 0.0,
                                 clustering_method: str = "simple",
                                 scale_size_hint: Optional[int] = None,
                                 duration_weighting: bool = False,
                                 harmonic_bias: bool = False) -> Optional[dict]:
    """Stima il diapason (A4) e raggruppa i rapporti dai tracciati F0 e CQT.
    - Non assume 12-TET o dizionari storici.
    - DEFAULT: Usa F0 (pYIN/YIN) + CQT per stima diapason su segnali complessi
    - Con f0_only=True: Usa solo F0 (pYIN/YIN) per compatibilità e velocità
    - Calcola i rapporti rispetto a base_hint_hz (basenote) e li riduce in [1,2).
    - Se digits è specificato, arrotonda i valori numerici al numero di decimali dato.
    - Raggruppa i rapporti per prossimità relativa (~2%.

    Args:
        audio_path: Path to audio file to analyze
        base_hint_hz: Base frequency for ratio calculation
        initial_a4_hz: Initial A4 frequency guess (default 440.0)
        frame_size: Frame size for analysis (default 1024)
        hop_size: Hop size for analysis (default 512)
        scala_cent: Scala cent parameter (optional)
        digits: Number of decimal digits for rounding (optional)
        f0_only: Se True, esclude analisi CQT dalla stima del diapason (solo F0)
        cqt_analysis: Dati CQT pre-calcolati (opzionale, per evitare ricalcolo)
        formant_method: Method for formant analysis ("cqt", "lpc", etc.)
        lpc_order: LPC order for analysis (optional)
        lpc_preemph: LPC pre-emphasis factor (default 0.97)
        lpc_window_ms: LPC window size in milliseconds (optional)
        tolerance_percent: Clustering tolerance percentage (default 2.0)
        min_ratio_count: Minimum occurrences for cluster (default 1)
        confidence_threshold: CREPE F0 confidence filter (default 0.0)
        clustering_method: Clustering method ("simple", "weighted", "adaptive")
        scale_size_hint: Expected number of scale degrees (optional)
        duration_weighting: Weight clusters by note duration (default False)
        harmonic_bias: Favor simple integer ratios (default False)

    Ritorna dict: { 'diapason_est': float, 'f0_list': [..], 'ratios_reduced': [..], 'ratio_clusters': [(center, count)] }
    """
    try:
        # Using pre-imported modules
        if not _np or not _lb:
            return None
    except (ImportError, ModuleNotFoundError):
        return None

    result = utils.load_mono_audio(audio_path)
    if result is None:
        return None
    y, sr = result

    # Estrai F0: CREPE (unico metodo)
    try:
        fmin = 50.0
        fmax = max(2000.0, sr / 4.0)
        # Auto-adapt fmin so at least two periods of fmin fit into the frame
        fmin_eff, fmax_eff = utils.calculate_f0_params(sr, frame_size, fmin, fmax)
        if fmin_eff > fmin:
            # Note: fmin is auto-adapted to ensure at least two periods fit into the frame.
            # Suppressing verbose print to avoid confusing output when using CQT for formants.
            pass
        fmin = fmin_eff
        
        # Use CREPE for detailed F0 analysis (scale degrees and base note detection)
        f0_vals, time_crepe, conf_vals = _extract_f0_values(y, sr)

        # For A4 diapason estimation only, use librosa pYIN/YIN (more robust)
        f0_vals_diapason = utils.extract_f0_with_pyin_yin_fallback(y, sr, fmin, fmax, frame_size, hop_size)
    except ValueError:
        f0_vals = utils.empty_float_array()
        conf_vals = utils.empty_float_array()
        f0_vals_diapason = utils.empty_float_array()

    # Deduplicate F0 list (preserve order, round to 0.1 Hz) and keep confidences
    f0_list: list[float] = []
    conf_list: list[float] = []
    if f0_vals.size > 0 and conf_vals.size > 0:
        seen_keys = set()
        for v, c in zip(map(float, f0_vals.tolist()), map(float, conf_vals.tolist())):
            key = round(v, 1)
            # Apply confidence threshold filtering
            if key not in seen_keys and v > 0 and c >= confidence_threshold:
                seen_keys.add(key)
                f0_list.append(v)
                conf_list.append(c)
    if not f0_list:
        return None

    # Initialize confidence tracking variables
    a4_confidence_interval = None
    a4_confidence_std = None

    # Initialize statistics variables
    a4_sample_count = 0
    a4_delta_440_hz = 0.0
    a4_delta_440_percent = 0.0

    # Stima del diapason: F0 + CQT (default) o solo F0 (con --f0-only)
    # Nota: il valore initial_a4_hz (eventuale diapason utente) viene usato solo per i riferimenti
    # nelle fasi di esportazione; qui manteniamo la stima indipendente per documentare il diapason originario.
    band_low = 220.0
    band_high = 880.0

    # Convert librosa F0 values to list for diapason estimation
    f0_diapason_list = []
    if f0_vals_diapason.size > 0:
        seen_keys = set()
        for v in map(float, f0_vals_diapason.tolist()):
            key = round(v, 1)
            if key not in seen_keys and v > 0:
                seen_keys.add(key)
                f0_diapason_list.append(v)

    # Add CQT-derived frequencies for enhanced diapason estimation (unless f0_only)
    cqt_diapason_list = []
    if not f0_only and cqt_analysis and isinstance(cqt_analysis, dict):
        # Extract prominent frequencies from existing CQT data
        cqt_data = cqt_analysis.get('cqt_data')
        cqt_freqs = cqt_analysis.get('cqt_freqs')

        if cqt_data is not None and cqt_freqs is not None:
            try:
                # Compute time-averaged spectrum
                cqt_mean = _np.mean(cqt_data, axis=1)

                # Find peaks in A4 range (220-880 Hz with some margin)
                a4_band_mask = (cqt_freqs >= 180.0) & (cqt_freqs <= 1000.0)
                if _np.any(a4_band_mask):
                    cqt_a4_region = cqt_mean[a4_band_mask]
                    freqs_a4_region = cqt_freqs[a4_band_mask]

                    # Find local maxima
                    if len(cqt_a4_region) > 2:
                        try:
                            from scipy.signal import find_peaks
                            # Use relative threshold for peak detection
                            threshold = _np.max(cqt_a4_region) * 0.1  # 10% of max
                            peaks, _ = find_peaks(cqt_a4_region, height=threshold, distance=5)

                            # Extract peak frequencies and weights
                            for peak_idx in peaks:
                                freq = freqs_a4_region[peak_idx]
                                amplitude = cqt_a4_region[peak_idx]

                                # Weight by amplitude (stronger peaks contribute more)
                                weight = max(1, int(amplitude * 10))  # Scale amplitude to weight
                                for _ in range(weight):
                                    cqt_diapason_list.append(float(freq))

                        except ImportError:
                            find_peaks = None  # Define fallback for linter
                            # Fallback without scipy
                            # Simple approach: take strongest frequencies
                            top_indices = _np.argsort(cqt_a4_region)[-5:]  # Top 5 frequencies
                            for idx in top_indices:
                                freq = freqs_a4_region[idx]
                                cqt_diapason_list.append(float(freq))

            except (ValueError, TypeError) as e:
                print(f"CQT diapason extraction failed: {e}")

    # Add LPC-derived frequencies for enhanced diapason estimation (alternative to CQT)
    lpc_diapason_list = []
    if not f0_only and formant_method == "lpc":
        try:
            # Run LPC analysis on the audio
            lpc_analysis_result = analyze_audio(
                audio_path=audio_path,
                method="lpc",
                frame_size=frame_size,
                hop_size=hop_size,
                lpc_order=lpc_order,
                lpc_preemph=lpc_preemph,
                lpc_window_ms=lpc_window_ms
            )

            if lpc_analysis_result and 'formants' in lpc_analysis_result:
                formants = lpc_analysis_result['formants']

                # Extract formant frequencies in A4 range (with margin)
                for freq, amp in formants:
                    if 180.0 <= freq <= 1000.0:  # A4 range with margin
                        # Weight by amplitude (if available)
                        weight = max(1, int(amp * 10)) if amp > 0 else 1
                        for _ in range(weight):
                            lpc_diapason_list.append(float(freq))

        except Exception as e:
            print(f"LPC diapason extraction failed: {e}")

    # Combine F0 and formant analysis frequency candidates
    formant_candidates = cqt_diapason_list if formant_method == "cqt" else lpc_diapason_list
    all_diapason_candidates = f0_diapason_list + formant_candidates

    if not f0_only and formant_candidates:
        method_name = "CQT" if formant_method == "cqt" else "LPC"
        print(f"Diapason estimation: {len(f0_diapason_list)} F0 + {len(formant_candidates)} {method_name} candidates")

    # Simplified approach: fold frequencies to A4 range with octave correction
    folded_candidates = []
    for f in all_diapason_candidates:
        # Apply conservative octave correction
        candidates = [f, f/2, f*2]
        valid_candidates = [cand for cand in candidates if band_low <= cand <= band_high]

        if valid_candidates:
            # Choose the candidate closest to 440 Hz (less aggressive than before)
            best_candidate = min(valid_candidates, key=lambda x: abs(x - 440.0))
            folded_candidates.append(best_candidate)

    if not folded_candidates:
        a4_est = 440.0
    else:
        try:
            import statistics

            # Focus on the center of the distribution (less extreme outliers)
            folded_candidates.sort()
            # Remove extreme outliers (bottom/top 10% if enough samples)
            if len(folded_candidates) >= 10:
                trim_count = max(1, len(folded_candidates) // 10)
                folded_candidates = folded_candidates[trim_count:-trim_count]

            # Use simple median (more robust than weighted average for librosa F0)
            a4_est_candidate = float(statistics.median(folded_candidates))

            # Apply conservative sanity check
            if 380.0 <= a4_est_candidate <= 465.0:
                a4_est = a4_est_candidate
            else:
                # If unrealistic, try trimming more aggressively
                if len(folded_candidates) >= 6:
                    # Remove more outliers and try again
                    trim_more = max(1, len(folded_candidates) // 4)
                    trimmed_more = folded_candidates[trim_more:-trim_more]
                    a4_est_trimmed = float(statistics.median(trimmed_more))

                    if 380.0 <= a4_est_trimmed <= 465.0:
                        a4_est = a4_est_trimmed
                        print(f"Warning: A4 estimate required aggressive outlier removal ({a4_est_candidate:.1f} → {a4_est:.1f} Hz)")
                    else:
                        print(f"Warning: A4 estimate ({a4_est_candidate:.1f} Hz) unrealistic, using 440 Hz default")
                        a4_est = 440.0
                else:
                    print(f"Warning: A4 estimate ({a4_est_candidate:.1f} Hz) unrealistic, using 440 Hz default")
                    a4_est = 440.0

        except (ValueError, TypeError, ImportError, AttributeError):
            a4_est = 440.0

    # Calculate statistical measures for A4 estimation reliability
    a4_confidence_interval = None
    a4_confidence_std = None
    a4_sample_count = 0
    a4_delta_440_hz = 0.0
    a4_delta_440_percent = 0.0

    if folded_candidates and len(folded_candidates) >= 3:  # Need sufficient samples
        try:
            import statistics
            a4_sample_count = len(folded_candidates)

            # Calculate basic statistics from the folded candidates
            a4_confidence_std = float(statistics.stdev(folded_candidates))

            # Calculate 95% confidence interval using standard error
            stderr = a4_confidence_std / (len(folded_candidates) ** 0.5)
            ci_margin = 1.96 * stderr  # 95% CI
            ci_lower = a4_est - ci_margin
            ci_upper = a4_est + ci_margin
            a4_confidence_interval = (ci_lower, ci_upper)

            # Calculate delta from standard A4 = 440 Hz
            a4_delta_440_hz = a4_est - 440.0
            a4_delta_440_percent = (a4_delta_440_hz / 440.0) * 100.0

        except (ImportError, ValueError, TypeError):
            pass  # Bootstrap failed, continue without CI

    # Calcola rapporti rispetto alla basenote indicata
    base = float(base_hint_hz) if base_hint_hz and base_hint_hz > 0 else a4_est
    ratios = [f / base for f in f0_list if f > 0 and base > 0]

    # Riduci in [1,2) using helper function
    ratios_reduced = [utils.fold_frequency_to_octave(r, 1.0, 2.0) for r in ratios]

    # Cluster per prossimità relativa con tolleranza personalizzabile
    tolerance_fraction = tolerance_percent / 100.0  # Converte percentuale in frazione

    # Calculate tolerance impact in cents for user information
    # For a typical reference frequency around 440Hz, tolerance_percent gives this cents tolerance:
    tolerance_cents = tolerance_percent / 100.0 * 1200 / math.log(2) * math.log(1 + tolerance_percent/100.0)
    # Simplified approximation: tolerance_percent ≈ cents/17.3 (for small percentages)
    tolerance_cents_approx = tolerance_percent * 17.3  # More accurate for small tolerances

    print(f"Clustering tolerance: {tolerance_percent}% relative ≈ {tolerance_cents_approx:.1f} cents")
    print(f"Clustering method: {clustering_method}")
    if scale_size_hint:
        print(f"Scale size hint: {scale_size_hint} degrees")

    # Prepare weights for weighted clustering if enabled
    weights = None
    if duration_weighting and clustering_method == "weighted":
        # Use confidence values as proxy for duration/strength
        weights = [conf_list[i] for i in range(len(ratios_reduced)) if i < len(conf_list)]
        if len(weights) < len(ratios_reduced):
            # Pad with equal weights if conf_list is shorter
            weights.extend([1.0] * (len(ratios_reduced) - len(weights)))

    # Apply clustering method
    if clustering_method == "weighted" and weights:
        centers = utils.cluster_ratios_weighted(ratios_reduced, weights, tolerance=tolerance_fraction)
    elif clustering_method == "adaptive":
        centers = utils.cluster_ratios_adaptive(ratios_reduced, base_tolerance=tolerance_fraction, scale_size_hint=scale_size_hint)
    else:
        # Default simple clustering
        centers = utils.cluster_ratios_by_proximity(ratios_reduced, tolerance=tolerance_fraction)

    # Apply harmonic bias if enabled
    if harmonic_bias:
        centers = utils.apply_harmonic_bias(centers, bias_strength=0.1)

    # Filter clusters by minimum count
    if min_ratio_count > 1:
        centers = [(center, count) for center, count in centers if count >= min_ratio_count]

    # Infer tuning system using comparative fitter (primary + alternative)
    prim_info, prim_steps, comp_info, comp_steps = infer_tuning_system_with_comparative(centers)
    tuning_info, scale_steps = prim_info, prim_steps
    tuning_comp, scale_steps_comp = comp_info, comp_steps
    if tuning_info is None:
        # Fallback: no inference possible
        tuning_info, scale_steps = None, []
    # Comparative may be None/[]

    # Estimate basenote: choose from actually detected scale degrees (not raw F0s)
    bn_midi = None
    bn_name = None
    bn_hz = None

    # Check if user specified a custom basenote (different from default C4 ~261.626 Hz)
    default_c4_hz = 261.625565  # C4 at A4=440Hz
    user_specified_basenote = abs(base_hint_hz - default_c4_hz) > 0.1  # Allow small tolerance

    if user_specified_basenote:
        # User specified a custom basenote - use it and convert to MIDI/name
        try:
            if a4_est and a4_est > 0:
                # Convert user's base frequency to MIDI note relative to estimated A4
                bn_hz = float(base_hint_hz)
                bn_midi = int(round(consts.MIDI_A4 + consts.SEMITONES_PER_OCTAVE * math.log2(bn_hz / float(a4_est))))
                bn_midi = max(consts.MIDI_MIN, min(consts.MIDI_MAX, bn_midi))
                try:
                    bn_name = utils.midi_to_note_name_12tet(bn_midi)
                except ValueError:
                    bn_name = None
                print(f"Using user-specified basenote: {bn_hz:.2f} Hz -> {bn_name}")
        except (ValueError, TypeError):
            pass

    # If no user basenote specified, try to infer from detected scale steps
    if bn_midi is None and scale_steps and a4_est and a4_est > 0:
        try:
            # Find the scale step with highest count (most detected)
            detected_steps = [(idx, ratio, count) for (idx, ratio, count) in scale_steps if count > 0]

            if detected_steps:
                # Sort by count (descending), then by ratio (ascending) for ties
                detected_steps.sort(key=lambda x: (-x[2], x[1]))
                best_idx, best_ratio, best_count = detected_steps[0]

                # Calculate base note frequency from the most frequent detected ratio
                if isinstance(best_ratio, (int, float)) and best_ratio > 0:
                    # base_hint_hz * best_ratio gives the actual frequency of this note
                    detected_note_freq = float(base_hint_hz) * float(best_ratio)

                    # Convert to MIDI note relative to estimated A4
                    bn_midi = int(round(consts.MIDI_A4 + consts.SEMITONES_PER_OCTAVE * math.log2(detected_note_freq / float(a4_est))))
                    bn_midi = max(consts.MIDI_MIN, min(consts.MIDI_MAX, bn_midi))
                    bn_hz = float(a4_est) * (2.0 ** ((bn_midi - consts.MIDI_A4) / 12.0))

                    try:
                        bn_name = utils.midi_to_note_name_12tet(bn_midi)
                    except ValueError:
                        bn_name = None

                    # Format base note selection message
                    print(f"\n" + "-"*60)
                    print("BASE NOTE SELECTION")
                    print("-"*60)
                    print(f"Selected from detected scale:")
                    print(f"  Ratio:     {best_ratio:.6f}")
                    print(f"  Count:     {best_count} occurrences")
                    print(f"  Note:      {bn_name}")
                    print(f"  Frequency: {bn_hz:.3f} Hz" if bn_hz else "")
                    print("-"*60)

                    # Store formatted info for diapason text file
                    base_note_info = (
                        f"Base Note Selection:\n"
                        f"  Source: Detected scale\n"
                        f"  Ratio: {best_ratio:.6f} (count: {best_count})\n"
                        f"  Note: {bn_name}\n"
                        f"  Frequency: {bn_hz:.3f} Hz" if bn_hz else "Base Note Selection: N/A"
                    )
                    if isinstance(result, dict):
                        result['base_note_selection_info'] = base_note_info
        except (ValueError, TypeError, IndexError):
            pass

    # Fallback: use original F0-based method if scale inference failed
    if bn_midi is None:
        try:
            mids = []
            if a4_est and a4_est > 0:
                for f in f0_list:
                    if f and f > 0:
                        m = int(round(consts.MIDI_A4 + consts.SEMITONES_PER_OCTAVE * math.log2(float(f) / float(a4_est))))
                        m = max(consts.MIDI_MIN, min(consts.MIDI_MAX, m))
                        mids.append(m)
            if mids:
                # istogramma classi mod 12
                counts = [0] * 12
                for m in mids:
                    counts[m % 12] += 1
                best_class = int(max(range(12), key=lambda k: counts[k])) if any(counts) else (mids[0] % 12)
                # mediana delle note appartenenti alla classe
                in_class = [m for m in mids if (m % 12) == best_class]
                try:
                    import statistics
                    m_med = int(round(statistics.median(in_class if in_class else mids)))
                except (ValueError, ImportError):
                    m_med = int(round(sum(in_class if in_class else mids) / float(len(in_class if in_class else mids))))
                m_med = max(consts.MIDI_MIN, min(consts.MIDI_MAX, m_med))
                bn_midi = m_med
                bn_hz = float(a4_est) * (2.0 ** ((bn_midi - consts.MIDI_A4) / 12.0))
                try:
                    bn_name = utils.midi_to_note_name_12tet(bn_midi)
                except ValueError:
                    bn_name = None
                print(f"\n" + "-"*60)
                print("BASE NOTE SELECTION")
                print("-"*60)
                print(f"Selected from F0 analysis (fallback):")
                print(f"  Note:      {bn_name}")
                print(f"  MIDI:      {bn_midi}" if bn_midi else "")
                print(f"  Frequency: {bn_hz:.3f} Hz" if bn_hz else "")
                print("-"*60)

                # Store formatted info for diapason text file
                base_note_info_parts = [
                    f"Base Note Selection:",
                    f"  Source: F0 analysis (fallback)",
                    f"  Note: {bn_name}"
                ]
                if bn_midi:
                    base_note_info_parts.append(f"  MIDI: {bn_midi}")
                if bn_hz:
                    base_note_info_parts.append(f"  Frequency: {bn_hz:.3f} Hz")
                base_note_info = "\n".join(base_note_info_parts) if bn_name else "Base Note Selection: N/A"
                if isinstance(result, dict):
                    result['base_note_selection_info'] = base_note_info
        except ValueError:
            pass

    # Scala (.scl) best-match from 'scl' directory based on ratio cluster centers
    # Filter out clusters with count = 0 (theoretical ratios not actually detected)
    detected_centers = [(ratio, count) for (ratio, count) in centers if count > 0]
    if not detected_centers:
        # Fallback: use all centers if no detected ones (shouldn't happen normally)
        detected_centers = centers

    try:
        scala_info, scala_steps = match_scala_scales(detected_centers, dir_path='scl')
    except ValueError:
        scala_info, scala_steps = (None, [])
    # Also compute top-5 Scala matches
    try:
        _topk = match_scala_scales_topk(detected_centers, dir_path='scl', k=5)
        scala_top_matches = [{'name': info.get('name', ''), 'file': info.get('file', ''),
                              'avg_cents_error': float(info.get('avg_cents_error', 0.0))} for (info, _steps) in _topk]
    except ValueError:
        scala_top_matches: list = []
    # Optionally compute all matches within threshold cents
    scala_within_matches: list = []
    scala_within_threshold = None
    try:
        if isinstance(scala_cent, (int, float)) and math.isfinite(scala_cent) and float(scala_cent) >= 0:
            scala_within_matches = match_scala_scales_within_threshold(centers, dir_path='scl',
                                                                       threshold_cents=float(scala_cent))
            scala_within_threshold = float(scala_cent)
    except (ValueError, TypeError, ImportError):
        scala_within_matches: list = []
        scala_within_threshold = None

    # --- NEW: Match raffinato sui scale_steps ---
    refined_scala_info = None
    refined_scala_steps = []
    refined_scala_top3 = []
    try:
        if scale_steps:  # Solo se abbiamo scale_steps dall'inferenza
            refined_scala_info, refined_scala_steps, refined_scala_top3 = match_refined_scale_steps(scale_steps, dir_path='scl')
            # refined_scala_info will be displayed in main program output
    except (ValueError, TypeError, ImportError):
        refined_scala_info = None
        refined_scala_steps = []
        refined_scala_top3 = []

    # --- Estimate frequency and ratio uncertainty (heuristic) ---
    freq_resolution_hz = None
    freq_uncert_hz = None
    frames_used_est = 0
    ratio_uncert_typical = None
    suggested_ratio_decimals = None
    try:
        # Nominal bin resolution (as in STFT): Δf = sr / N
        if 'sr' in locals() and isinstance(sr, (int, float)) and sr > 0 and isinstance(frame_size,
                                                                                       int) and frame_size > 0:
            freq_resolution_hz = float(sr) / float(frame_size)
            # Sub-bin interpolation heuristic (~1/10 of bin)
            subbin = freq_resolution_hz / 10.0
            # Multi-frame averaging reduces variance ~ 1/sqrt(N)
            try:
                # Using pre-imported numpy  # reuse numpy
                frames_used_est = int(_np.size(f0_vals)) if 'f0_vals' in locals() and _np is not None else 0
            except (ImportError, AttributeError):
                frames_used_est = (len(f0_list) if isinstance(f0_list, list) else 0)
            denom = math.sqrt(max(1.0, float(frames_used_est)))
            freq_uncert_hz = subbin / denom
            # Typical ratio uncertainty around r = f/base
            f_ref = float(a4_est) if isinstance(a4_est, (int, float)) and a4_est > 0 else None
            b_ref = float(base) if isinstance(base, (int, float)) and base > 0 else f_ref
            if f_ref and b_ref and f_ref > 0 and b_ref > 0 and isinstance(freq_uncert_hz, (int, float)):
                # Propagate relative errors: Δr/r = sqrt((Δf/f)^2 + (Δb/b)^2)
                rel = math.sqrt((freq_uncert_hz / f_ref) ** 2 + (freq_uncert_hz / b_ref) ** 2)
                ratio_uncert_typical = rel * (f_ref / b_ref)
                # Suggest decimals d such that 0.5*10^-d < Δr
                if ratio_uncert_typical > 0 and math.isfinite(ratio_uncert_typical):
                    d_raw = math.ceil(max(0.0, -math.log10(float(ratio_uncert_typical) * 2.0)))
                    suggested_ratio_decimals = int(max(2, min(8, d_raw)))
    except (ValueError, TypeError, ArithmeticError):
        pass

    # Apply digits rounding if specified
    if digits is not None:
        a4_est = utils.apply_digits_rounding(a4_est, digits)
        base = utils.apply_digits_rounding(base, digits)
        bn_hz = utils.apply_digits_rounding(bn_hz, digits)

        # Round ratios_reduced
        if ratios_reduced:
            ratios_reduced = [utils.apply_digits_rounding(r, digits) for r in ratios_reduced]

        # Round f0_list
        if f0_list:
            f0_list = [utils.apply_digits_rounding(f, digits) for f in f0_list]

        # Round cluster centers
        if centers:
            centers = [(utils.apply_digits_rounding(ratio, digits), count) for (ratio, count) in centers]

        # Round scale_steps ratios
        if scale_steps:
            scale_steps = [(idx, utils.apply_digits_rounding(ratio, digits), count) for (idx, ratio, count) in scale_steps]

        # Round scale_steps_comp ratios
        if scale_steps_comp:
            scale_steps_comp = [(idx, utils.apply_digits_rounding(ratio, digits), count) for (idx, ratio, count) in scale_steps_comp]

        # Round refined scale match results
        if refined_scala_top3:
            for match in refined_scala_top3:
                if isinstance(match, dict):
                    match['avg_cents_error'] = utils.apply_digits_rounding(match.get('avg_cents_error'), digits)
                    match['coverage_ratio'] = utils.apply_digits_rounding(match.get('coverage_ratio'), digits)

    result = {
        'diapason_est': a4_est,
        'f0_list': f0_list,
        'ratios_reduced': ratios_reduced,
        'ratio_clusters': centers,
        'base_hint_hz': base,
        'tuning_inferred': tuning_info,
        'scale_steps': scale_steps,
        'tuning_comparative': tuning_comp,
        'scale_steps_comp': scale_steps_comp,
        'basenote_est_hz': bn_hz,
        'basenote_est_midi': bn_midi,
        'basenote_est_name_12tet': bn_name,
        # Scala match results
        'scala_match_info': scala_info,
        'scala_match_steps': scala_steps,
        'scala_top_matches': scala_top_matches,
        'scala_within_matches': scala_within_matches,
        'scala_within_threshold_cents': scala_within_threshold,
        # Refined scale match results (based on scale_steps)
        'refined_scala_match_info': refined_scala_info,
        'refined_scala_match_steps': refined_scala_steps,
        'refined_scala_top3_matches': refined_scala_top3,
        # Uncertainty (heuristic)
        'freq_resolution_hz': freq_resolution_hz,
        'freq_uncert_hz': freq_uncert_hz,
        'frames_used': frames_used_est,
        'ratio_uncert_typical': ratio_uncert_typical,
        'suggested_ratio_decimals': suggested_ratio_decimals,
        # A4 confidence information
        'diapason_confidence_interval': a4_confidence_interval,
        'diapason_confidence_std': a4_confidence_std,
        'diapason_sample_count': a4_sample_count,
        'diapason_delta_440_hz': a4_delta_440_hz,
        'diapason_delta_440_percent': a4_delta_440_percent,
        # Clustering tolerance information
        'clustering_tolerance_percent': tolerance_percent,
        'clustering_tolerance_cents_approx': tolerance_cents_approx,
        'clustering_method': clustering_method,
        'scale_size_hint': scale_size_hint,
    }

    # NEW: Analisi frammenti di scala
    try:
        fragment_analysis = analyze_scale_fragments(
            cluster_centers=centers,
            tuning_info=tuning_info,
            scala_info=scala_info
        )
        # Aggiungi i risultati dell'analisi frammenti al risultato principale
        if fragment_analysis and fragment_analysis.get('fragment_analysis'):
            result.update(fragment_analysis)
    except (ValueError, TypeError, AttributeError):
        # Se l'analisi frammenti fallisce, continua senza
        pass

    return result


def export_diapason_plot_png(output_base: str, analysis_result: dict, basenote_hz: float, diapason_hz: float) -> \
Optional[str]:
    """Esporta un grafico PNG dell'andamento della F0 nel tempo con quattro riferimenti:
    1) 12-TET (Hz), 2) Midicents, 3) Pitagorico 12, 4) Pitagorico 7.
    Saves as f"{output_base}_f0.png". Returns the path or None in case of error.
    """
    try:
        # Force a non-interactive backend for CLI/headless environments
        try:
            import matplotlib as _mpl
            _mpl.use('Agg')
        except (ImportError, AttributeError):
            pass
        import matplotlib.pyplot as _plt
        # Using pre-imported numpy
    except ImportError as e:
        print(f"matplotlib not available for PNG export: {e}")
        return None

    if not isinstance(analysis_result, dict):
        return None
    f0_series = analysis_result.get('f0_series') or analysis_result.get('f0_list') or []
    if not f0_series:
        # Fallback: use single-value series from diapason_est or f0_hz to allow simple plot
        try:
            v = analysis_result.get('diapason_est') or analysis_result.get('f0_hz')
            v = float(v) if v and float(v) > 0 else None
        except (ValueError, TypeError):
            v = None
        if v:
            f0_series = [v] * 1000  # 10 seconds at 10ms step
        else:
            return None
    # Times
    t = analysis_result.get('f0_times') or []
    try:
        t = [float(x) for x in t]
    except ValueError:
        t = []
    n = len(f0_series)
    f = _np.asarray([float(x) for x in f0_series], dtype=float)
    if not t or len(t) != n:
        # fallback: assume 10ms hop
        t = (_np.arange(n, dtype=float) * 0.01).tolist()

    # Helper to build reference frequencies
    def tet12_ratios():
        return [2.0 ** (k / 12.0) for k in range(12)]

    def pyth12_ratios():
        p12_cents, _ = utils.get_pythagorean_scales()
        return [float(utils.cents_to_fraction(c)) for c in p12_cents]

    def pyth7_ratios():
        _, p7_cents = utils.get_pythagorean_scales()
        return [float(utils.cents_to_fraction(c)) for c in p7_cents]

    # Use estimated basenote from analysis if available; otherwise fallback to provided basenote
    try:
        _bn_est = analysis_result.get('basenote_est_hz')
        base_ref_hz = float(_bn_est) if _bn_est and float(_bn_est) > 0 else float(basenote_hz)
    except ValueError:
        base_ref_hz = float(basenote_hz)
    tet_refs = [float(base_ref_hz) * r for r in tet12_ratios()]
    py12_refs = [float(base_ref_hz) * r for r in pyth12_ratios()]
    py7_refs = [float(base_ref_hz) * r for r in pyth7_ratios()]

    # Prepare figure
    fig, axes = _plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # 1) 12-TET (Hz)
    ax = axes[0]
    ax.plot(t, f, label='F0 (Hz)', color='C0', linewidth=1.2)
    for yref in tet_refs:
        ax.axhline(y=yref, color='C1', alpha=0.35, linewidth=0.8)
    ax.set_ylabel('Hz (12-TET refs)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 2) Midicents
    ax = axes[1]
    try:
        a4 = float(diapason_hz) if diapason_hz and diapason_hz > 0 else 440.0
    except ValueError:
        a4 = 440.0
    mc = 1200.0 * _np.log2(_np.clip(f, 1e-6, None) / a4) + 6900.0
    ax.plot(t, mc, label='F0 (midicents)', color='C0', linewidth=1.2)
    # grid every 100 cents
    ymin, ymax = _np.nanmin(mc), _np.nanmax(mc)
    cmin = 100.0 * (int(ymin // 100.0) - 1)
    cmax = 100.0 * (int(ymax // 100.0) + 1)
    for yref in _np.arange(cmin, cmax + 100.0, 100.0):
        ax.axhline(y=yref, color='gray', alpha=0.2, linewidth=0.6)
    ax.set_ylabel('Midicents')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 3) Pitagorico 12 (Hz)
    ax = axes[2]
    ax.plot(t, f, label='F0 (Hz)', color='C0', linewidth=1.2)
    for yref in py12_refs:
        ax.axhline(y=yref, color='C2', alpha=0.35, linewidth=0.8)
    ax.set_ylabel('Hz (Pyth-12 refs)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 4) Pitagorico 7 (Hz)
    ax = axes[3]
    ax.plot(t, f, label='F0 (Hz)', color='C0', linewidth=1.2)
    for yref in py7_refs:
        ax.axhline(y=yref, color='C3', alpha=0.35, linewidth=0.8)
    ax.set_ylabel('Hz (Pyth-7 refs)')
    ax.set_xlabel('Tempo (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    _plt.tight_layout()
    out_path = f"{output_base}_f0.png"
    try:
        # Ensure output directory exists
        try:
            _dir = os.path.dirname(out_path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
        except (OSError, PermissionError):
            pass
        fig.savefig(out_path, dpi=150)
        _plt.close(fig)
        print(f"PNG plot exported: {out_path}")
        return out_path
    except (OSError, IOError, ValueError, ImportError) as e:
        print(f"Error saving PNG: {e}")
        try:
            _plt.close(fig)
        except (AttributeError, ValueError):
            pass
        return None


def render_analysis_to_wav(audio_path: str, analysis_result: dict, output_base: str) -> Optional[str]:
    """Renders a frequency-controlled sinusoidal signal based on analysis results.
    - Uses f0_series if available; otherwise f0_list (diapason); otherwise constant f0_hz.
    - The rendering duration matches that of the source audio.
    Returns the path to the generated WAV file, or None in case of error.
    """
    try:
        # Using pre-imported numpy and librosa
        if _np is None or _lb is None:
            raise ImportError("Required modules not available")

        # Render audio using synthesized frequencies
        result = utils.load_mono_audio(audio_path)
        if result is None:
            print("Audio load error for render")
            return None
        y, sr = result

        n_samples = len(y)
        dur = n_samples / float(sr) if sr and sr > 0 else 0.0
        if n_samples <= 0 or dur <= 0:
            print("Empty audio: render aborted")
            return None

        # Estrai traiettoria di frequenza
        f0_traj = None
        if isinstance(analysis_result, dict):
            seq = analysis_result.get('f0_series')
            if isinstance(seq, (list, tuple)) and len(seq) > 0:
                f0_traj = [float(x) if (x is not None and x == x and x > 0) else _np.nan for x in seq]
            elif isinstance(analysis_result.get('f0_list'), (list, tuple)) and len(analysis_result.get('f0_list')) > 0:
                f0_traj = [float(x) if (x is not None and x == x and x > 0) else _np.nan for x in
                           analysis_result.get('f0_list')]
            else:
                f0_hz = utils.safe_float_positive(analysis_result.get('f0_hz'))
                if f0_hz is not None:
                    f0_traj = [f0_hz]

        if not f0_traj:
            # Fallback: if diapason_est or basenote_est_hz is available, render a constant tone over the duration
            v = analysis_result.get('diapason_est') or analysis_result.get('basenote_est_hz') or analysis_result.get('f0_hz')
            v = utils.safe_float_positive(v)
            if v is not None:
                f0_traj = [v]
            else:
                print("No F0 available for render")
                return None

        f0_arr = _np.array(f0_traj, dtype=float)
        # Sostituisci NaN con interpolazione semplice
        if _np.isnan(f0_arr).any():
            idx = _np.arange(len(f0_arr))
            good = _np.isfinite(f0_arr)
            if good.any():
                f0_arr = _np.interp(idx, idx[good], f0_arr[good])
            else:
                print("Invalid F0 trajectory")
                return None

        # Clamp frequenze a [20, 5000] Hz
        f0_arr = _np.clip(f0_arr, 20.0, 5000.0)

        # Interpola a livello per-sample sull'intervallo [0, dur]
        t_series = _np.linspace(0.0, dur, num=len(f0_arr), endpoint=False)
        t_samples = _np.arange(n_samples, dtype=float) / float(sr)
        f_per_sample = _np.interp(t_samples, t_series, f0_arr)

        # Integrazione di fase: phi[n] = phi[n-1] + 2*pi*f[n]/sr
        phase = _np.cumsum(2.0 * _np.pi * f_per_sample / float(sr))
        out = 0.2 * _np.sin(phase).astype(_np.float32)

        out_path = f"{output_base}_render.wav"

        # Scrittura WAV con fallback
        if utils.write_wav_with_fallback(out_path, out, sr):
            print(f"Rendered audio saved to {out_path}")
            return out_path
        else:
            print(f"WAV write error")
            return None
    except (ImportError, ValueError, TypeError) as e:
        print(f"Audio render error: {e}")
        return None


def export_cqt_plot_png(output_base: str, analysis_result: dict, dpi: int = 300) -> Optional[str]:
    """Esporta un plot PNG del CQT con DPI specificato.
    Saves as f"{output_base}_cqt.png". Returns the path or None in case of error.
    """
    fig = None  # Initialize to avoid linter warning about potential undefined reference
    try:
        # Force a non-interactive backend for CLI/headless environments
        try:
            import matplotlib as _mpl
            _mpl.use('Agg')
        except (ImportError, AttributeError):
            pass
        import matplotlib.pyplot as _plt
        # Using pre-imported numpy
    except ImportError as e:
        print(f"matplotlib not available for CQT PNG export: {e}")
        return None

    if not isinstance(analysis_result, dict):
        return None

    cqt_data = analysis_result.get('cqt_data')
    cqt_freqs = analysis_result.get('cqt_freqs')
    cqt_times = analysis_result.get('cqt_times')

    # Remove debug output

    # Clean CQT plot generation without debug output

    if cqt_data is None or cqt_freqs is None or cqt_times is None:
        return None

    try:
        # Convert to dB scale for better visualization
        cqt_db = 20 * _np.log10(_np.abs(cqt_data) + 1e-8)

        # Use percentile-based scaling for better contrast (critical fix!)
        vmin = _np.percentile(cqt_db, 5)   # 5th percentile
        vmax = _np.percentile(cqt_db, 99)  # 99th percentile

        # Apply percentile-based scaling for better contrast

        # Create figure with high DPI
        fig, ax = _plt.subplots(figsize=(12, 8), dpi=dpi)

        # Create spectrogram plot with percentile-based scaling
        im = ax.imshow(cqt_db, aspect='auto', origin='lower',
                       extent=(cqt_times[0], cqt_times[-1], cqt_freqs[0], cqt_freqs[-1]),
                       cmap='magma', interpolation='nearest', vmin=vmin, vmax=vmax)

        # Set labels and title with provenance info
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        # Compose title including source filename and CQT params if available
        src = analysis_result.get('audio_path') if isinstance(analysis_result, dict) else None
        src_name = os.path.basename(src) if isinstance(src, str) and src else ''
        sr_val = analysis_result.get('sample_rate') if isinstance(analysis_result, dict) else None
        cqp = analysis_result.get('cqt_params') if isinstance(analysis_result, dict) else None
        if isinstance(sr_val, (int, float)) and sr_val:
            sr_txt = f" • sr={int(sr_val)} Hz"
        else:
            sr_txt = ''
        if isinstance(cqp, dict):
            try:
                p_txt = f" • fmin={float(cqp.get('fmin', 0)):.2f} Hz • bpo={int(cqp.get('bins_per_octave', 0))} • n_bins={int(cqp.get('n_bins', 0))} • hop={int(cqp.get('hop_length', 0))}"
            except (ValueError, TypeError):
                p_txt = ''
        else:
            p_txt = ''
        title_main = 'Constant-Q Transform (CQT) Spectrogram'
        title_src = f"source: {src_name}" if src_name else ''
        title = title_main if not title_src else f"{title_main} — {title_src}{sr_txt}{p_txt}"
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = _plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=12)

        # Set y-axis to log scale for better frequency visualization
        # Set frequency axis to log scale for better visualization
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Tight layout
        _plt.tight_layout()

        # Save with specified DPI
        png_path = f"{output_base}_cqt.png"
        _plt.savefig(png_path, dpi=dpi, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        _plt.close(fig)

        return png_path

    except OSError as e:
        print(f"Error generating CQT plot: {e}")
        if fig is not None:
            _plt.close(fig)
        return None


def estimate_a4_from_analysis(analysis_result: dict, default_a4: float = consts.DEFAULT_DIAPASON) -> Optional[float]:
    """Estimates A4 (diapason) from available analysis data.
    Preferisce analysis_result['diapason_est']; altrimenti usa f0_series/f0_list/f0_hz
    ripiegando le frequenze nell'intervallo [220,880) e privilegiando 360–520 Hz.
    """
    v = analysis_result.get('diapason_est') if isinstance(analysis_result, dict) else None
    result = utils.safe_float_positive(v)
    if result is not None:
        return result
    if not _np:
        return utils.safe_float_positive(default_a4, consts.DEFAULT_DIAPASON)
    f0s = []
    try:
        if isinstance(analysis_result, dict):
            if isinstance(analysis_result.get('f0_series'), (list, tuple)):
                f0s = [float(x) for x in analysis_result.get('f0_series') if
                       isinstance(x, (int, float)) and float(x) > 0]
            elif isinstance(analysis_result.get('f0_list'), (list, tuple)):
                f0s = [float(x) for x in analysis_result.get('f0_list') if isinstance(x, (int, float)) and float(x) > 0]
            else:
                f0_hz = utils.safe_float_positive(analysis_result.get('f0_hz'))
                if f0_hz is not None:
                    f0s = [f0_hz]
    except ValueError:
        f0s = []
    if not f0s:
        return utils.safe_float_positive(default_a4, consts.DEFAULT_DIAPASON)

    def _fold(f: float, lo: float, hi: float) -> float:
        if f <= 0:
            return f
        while f < lo:
            f *= 2.0
        while f >= hi:
            f /= 2.0
        return f

    folded = [_fold(float(f), 220.0, 880.0) for f in f0s if f and f > 0]
    focus = [f for f in folded if 360.0 <= f <= 520.0]
    cand = focus if focus else folded
    if not cand:
        return utils.safe_float_positive(default_a4, consts.DEFAULT_DIAPASON)
    try:
        return utils.safe_median_from_floats(cand)
    except (ValueError, TypeError):
        return utils.safe_float_positive(default_a4, consts.DEFAULT_DIAPASON)


def render_constant_tone_wav(output_base: str, freq_hz: float, duration_s: float = 30.0, sr: int = 48000) -> Optional[
    str]:
    """Renderizza una sinusoide costante alla frequenza data per una durata.
    Salva come f"{output_base}_diapason.wav".
    """
    try:
        if not isinstance(freq_hz, (int, float)) or float(freq_hz) <= 0:
            return None
        if not isinstance(duration_s, (int, float)) or float(duration_s) <= 0:
            duration_s = 30.0
        # Using pre-imported numpy
        n_samples = int(max(1, int(sr * float(duration_s))))
        t = _np.arange(n_samples, dtype=float) / float(sr)
        y = (0.2 * _np.sin(2.0 * _np.pi * float(freq_hz) * t)).astype(_np.float32)
        out_path = f"{output_base}_diapason.wav"
        if utils.write_wav_with_fallback(out_path, y, int(sr)):
            return out_path
        else:
            print(f"WAV write error")
            return None
    except (ImportError, ValueError, TypeError) as e:
        print(f"Diapason render error: {e}")
        return None


def analyze_scale_fragments(cluster_centers: List[Tuple[float, int]],
                           tuning_info: Optional[dict] = None,
                           scala_info: Optional[dict] = None) -> dict:
    """Analizza i frammenti di scala identificati dall'audio.

    Questa funzione prende i centroidi dei cluster dell'audio e confronta
    con le scale inferite (TET, Pythagorean, ecc.) e Scala per determinare
    quali gradi/note della scala sono stati effettivamente rilevati nell'audio.

    Args:
        cluster_centers: Lista di tuple (ratio, count) dai cluster dell'audio
        tuning_info: Informazioni sulla scala inferita (da infer_tuning_system)
        scala_info: Informazioni sulla scala Scala (da match_scala_scales)

    Returns:
        dict: {
            'fragment_analysis': {
                'inferred_system': {
                    'detected_degrees': [...],  # gradi rilevati
                    'missing_degrees': [...],   # gradi teorici mancanti
                    'coverage_percent': float   # percentuale di copertura
                },
                'scala_match': {
                    'detected_degrees': [...],
                    'missing_degrees': [...],
                    'coverage_percent': float
                }
            }
        }
    """
    result = {
        'fragment_analysis': {
            'inferred_system': None,
            'scala_match': None
        }
    }

    if not cluster_centers:
        return result

    # Converti i centroidi in cents per il matching
    centers_cents = utils.process_cluster_centers_to_cents(cluster_centers)
    if not centers_cents:
        return result

    # Analizza il sistema inferito
    if tuning_info and isinstance(tuning_info, dict):
        inferred_analysis = _analyze_system_fragment(centers_cents, tuning_info)
        result['fragment_analysis']['inferred_system'] = inferred_analysis

    # Analizza il match Scala
    if scala_info and isinstance(scala_info, dict):
        scala_analysis = _analyze_scala_fragment(centers_cents, scala_info)
        result['fragment_analysis']['scala_match'] = scala_analysis

    return result


def _analyze_fragment_degrees(centers_cents: List[Tuple[float, int]], theoretical_degrees: List[float], threshold_cents: float = 25.0) -> Tuple[List[dict], List[dict]]:
    """Common logic for analyzing detected and missing degrees."""
    detected_degrees = []
    missing_degrees = []

    for i, theory_cent in enumerate(theoretical_degrees):
        found = False
        for center_cent, count in centers_cents:
            # Distanza circolare
            diff = min(abs(center_cent - theory_cent),
                      1200.0 - abs(center_cent - theory_cent))
            if diff <= threshold_cents:
                detected_degrees.append({
                    'degree': i,
                    'theoretical_cents': theory_cent,
                    'detected_cents': center_cent,
                    'cents_error': center_cent - theory_cent,
                    'count': count
                })
                found = True
                break

        if not found:
            missing_degrees.append({
                'degree': i,
                'theoretical_cents': theory_cent
            })

    return detected_degrees, missing_degrees


def _analyze_system_fragment(centers_cents: List[Tuple[float, int]],
                           tuning_info: dict) -> Optional[dict]:
    """Analizza il frammento per un sistema di intonazione inferito."""
    try:
        # Ottieni i parametri del sistema
        params = tuning_info.get('params', {})
        name = tuning_info.get('name', '')

        # Genera la scala teorica completa
        theoretical_degrees = _generate_theoretical_scale(name, params)
        if not theoretical_degrees:
            return None

        # Usa la logica comune per analizzare i gradi
        detected_degrees, missing_degrees = _analyze_fragment_degrees(centers_cents, theoretical_degrees)

        # Calcola percentuale di copertura
        total_degrees = len(theoretical_degrees)
        detected_count = len(detected_degrees)
        coverage_percent = (detected_count / total_degrees * 100.0) if total_degrees > 0 else 0.0

        return {
            'system_name': name,
            'total_degrees': total_degrees,
            'detected_count': detected_count,
            'coverage_percent': coverage_percent,
            'detected_degrees': detected_degrees,
            'missing_degrees': missing_degrees
        }

    except (ValueError, TypeError, KeyError):
        return None


def _analyze_scala_fragment(centers_cents: List[Tuple[float, int]],
                          scala_info: dict) -> Optional[dict]:
    """Analizza il frammento per una scala Scala (.scl)."""
    try:
        # Carica il file .scl per ottenere i gradi teorici
        scala_file = scala_info.get('file', '')
        scala_name = scala_info.get('name', '')

        if not scala_file:
            return None

        # Cerca il file .scl
        import os
        scl_path = os.path.join('scl', scala_file)
        if not os.path.exists(scl_path):
            # Cerca ricorsivamente
            for root, dirs, files in os.walk('scl'):
                if scala_file in files:
                    scl_path = os.path.join(root, scala_file)
                    break

        if not os.path.exists(scl_path):
            return None

        # Parsa il file .scl
        scl_data = tun_csd.parse_scl_file(scl_path)
        if not scl_data:
            return None

        theoretical_degrees = scl_data.get('cents', [])
        if not theoretical_degrees:
            return None

        # Usa la logica comune per analizzare i gradi
        detected_degrees, missing_degrees = _analyze_fragment_degrees(centers_cents, theoretical_degrees)

        # Calcola percentuale di copertura
        total_degrees = len(theoretical_degrees)
        detected_count = len(detected_degrees)
        coverage_percent = (detected_count / total_degrees * 100.0) if total_degrees > 0 else 0.0

        return {
            'scala_name': scala_name,
            'scala_file': scala_file,
            'total_degrees': total_degrees,
            'detected_count': detected_count,
            'coverage_percent': coverage_percent,
            'detected_degrees': detected_degrees,
            'missing_degrees': missing_degrees
        }

    except (ValueError, TypeError, KeyError, OSError):
        return None


def _generate_theoretical_scale(system_name: str, params: dict) -> List[float]:
    """Genera i cents teorici per un sistema di intonazione."""
    try:
        import math

        if not system_name:
            return []

        name_lower = system_name.lower()

        # TET systems
        if 'tet' in name_lower:
            n = params.get('n', 12)
            if isinstance(n, int) and n > 0:
                return utils.get_tet_scale(n)

        # Pythagorean systems
        elif 'pythagorean' in name_lower:
            p12_cents, p7_cents = utils.get_pythagorean_scales()
            if '12' in name_lower:
                return p12_cents
            elif '7' in name_lower:
                return p7_cents

        # Rank-1 systems
        elif 'rank-1' in name_lower:
            generator_cents = params.get('generator_cents')
            period_cents = params.get('period_cents', 1200.0)

            if isinstance(generator_cents, (int, float)) and isinstance(period_cents, (int, float)):
                max_steps = int(period_cents / generator_cents) + 1
                steps = [(k * generator_cents) % 1200.0 for k in range(max(1, max_steps))]
                return sorted(list(set(steps)))  # rimuovi duplicati

        # Rank-2 systems
        elif 'rank-2' in name_lower:
            gen1_cents = params.get('generator1_cents')
            gen2_cents = params.get('generator2_cents')

            if isinstance(gen1_cents, (int, float)) and isinstance(gen2_cents, (int, float)):
                # Genera un piccolo reticolo
                steps = []
                for a in range(-6, 7):
                    for b in range(-6, 7):
                        step = (a * gen1_cents + b * gen2_cents) % 1200.0
                        steps.append(step)
                return sorted(list(set(steps)))

        return []

    except (ValueError, TypeError, AttributeError):
        return []


def cross_inference_cqt_f0(cqt_analysis: dict, f0_audio_path: str) -> Optional[dict]:
    """Correlazione tra analisi CQT (da --diapason-analysis --audio-file) e dati F0 (da file specificato in --cross-inference)
    per determinare le note della scala.

    Args:
        cqt_analysis: Risultato dell'analisi diapason con dati CQT
        f0_audio_path: Path al file audio per l'estrazione F0

    Returns:
        Dict con i risultati della correlazione, o None se fallisce
    """
    if not _np or not _lb:
        return None

    if not isinstance(cqt_analysis, dict):
        return None

    # Estrai dati CQT dal primo file
    cqt_data = cqt_analysis.get('cqt_data')
    cqt_freqs = cqt_analysis.get('cqt_freqs')
    cqt_times = cqt_analysis.get('cqt_times')

    if cqt_data is None or cqt_freqs is None or cqt_times is None:
        return None

    # Carica e analizza il secondo file per F0
    try:
        y2, sr2 = _lb.load(f0_audio_path, sr=None)
        if y2.size == 0:
            return None
    except (FileNotFoundError, OSError, ValueError):
        return None

    # Estrai F0 dal secondo file
    f0_vals, time_f0, conf_mask = _extract_f0_values(y2, sr2)

    if f0_vals.size == 0:
        return None

    # Rimuovi outlier da F0
    f0_valid = f0_vals[f0_vals > 50.0]  # Frequenze minime ragionevoli
    if f0_valid.size == 0:
        return None

    # Calcola i picchi CQT mediati nel tempo
    cqt_mean = _np.mean(cqt_data, axis=1)
    peak_indices = []

    # Trova i picchi nell'analisi CQT
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(cqt_mean, height=_np.max(cqt_mean) * 0.1)
        peak_indices = peaks
    except ImportError:
        find_peaks = None  # Define for linter consistency
        # Fallback without scipy: find maxima in the CQT
        for i in range(1, len(cqt_mean) - 1):
            if cqt_mean[i] > cqt_mean[i-1] and cqt_mean[i] > cqt_mean[i+1]:
                if cqt_mean[i] > _np.max(cqt_mean) * 0.1:
                    peak_indices.append(i)

    if len(peak_indices) == 0:
        return None

    # Frequenze dei picchi CQT
    cqt_peak_freqs = cqt_freqs[peak_indices]

    # Trova la frequenza fondamentale stimata da F0
    f0_fundamental = _np.median(f0_valid)

    # Correla i picchi CQT con gli armonici della fondamentale F0
    correlations = []
    base_freq = f0_fundamental

    for cqt_freq in cqt_peak_freqs:
        # Trova l'armonico più vicino
        harmonic_ratio = cqt_freq / base_freq
        nearest_harmonic = round(harmonic_ratio)

        if nearest_harmonic > 0:
            expected_freq = base_freq * nearest_harmonic
            error_cents = 1200 * _np.log2(cqt_freq / expected_freq) if expected_freq > 0 else float('inf')

            correlations.append({
                'cqt_freq': float(cqt_freq),
                'f0_fundamental': float(base_freq),
                'harmonic_number': int(nearest_harmonic),
                'expected_freq': float(expected_freq),
                'error_cents': float(error_cents),
                'ratio': float(cqt_freq / base_freq)
            })

    # Filtra correlazioni con errore ragionevole (< 50 cents)
    good_correlations = [c for c in correlations if abs(c['error_cents']) < 50.0]

    # Ordina per errore crescente
    good_correlations.sort(key=lambda x: abs(x['error_cents']))

    # Estrai ratios dalla correlazione per costruire la scala
    scale_ratios = []
    for corr in good_correlations:
        ratio = corr['ratio']
        # Riduci all'ottava
        while ratio >= 2.0:
            ratio /= 2.0
        while ratio < 1.0:
            ratio *= 2.0
        scale_ratios.append(ratio)

    # Rimuovi duplicati e ordina
    scale_ratios = sorted(list(set(scale_ratios)))

    # Converti in cents per confronto
    scale_cents = []
    for ratio in scale_ratios:
        if ratio > 0:
            cents = 1200 * _np.log2(ratio) if ratio > 0 else 0.0
            scale_cents.append(cents % 1200.0)

    scale_cents = sorted(scale_cents)

    return {
        'f0_fundamental': float(f0_fundamental),
        'f0_values': f0_valid.tolist(),
        'cqt_peak_frequencies': cqt_peak_freqs.tolist(),
        'correlations': good_correlations,
        'inferred_scale_ratios': scale_ratios,
        'inferred_scale_cents': scale_cents,
        'num_correlations': len(good_correlations),
        'cross_inference_method': 'cqt_f0_correlation'
    }
