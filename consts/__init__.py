"""Constants and metadata for THIN/SIM musical intonation systems.

This module centralizes all constants, configuration values, and metadata used across
the THIN and SIM applications for musical tuning system generation and analysis.

Program Metadata:
- Version information and authorship details
- Release dates and licensing information

Mathematical Constants:
- Ellis conversion factor for ratio-to-cents calculations
- Precision tolerances for rational arithmetic
- Default frequency references (A4 = 440 Hz)

MIDI and Audio Constants:
- MIDI note ranges and key mappings
- Frequency bounds for harmonic/subharmonic analysis
- Proximity thresholds for pitch clustering

Excel Export Configuration:
- Column assignments for different tuning system types
- Formatting constants for table generation

File Processing:
- Regular expression patterns for CSD file parsing
- TUN format reference frequencies
- Type definitions for numeric values

This centralized approach ensures consistency across all modules and
simplifies maintenance of configuration values.
"""

import math
import re
from fractions import Fraction
from typing import Union

# Metadata
__program_name__ = "THIN"  # New name in honor of Walter Branchi
__version__ = "PHI"  # Version string literal 'PHI'
__author__ = "LUCA BIMBI"
__date__ = "2025-09-22"  # The date distinguishes releases
__license__ = "MIT"  # See LICENSE file

# Constants
ELLIS_CONVERSION_FACTOR = 1200 / math.log(2)
RATIO_EPS = 1e-9
DEFAULT_DIAPASON = 440.0
DEFAULT_BASEKEY = 60
DEFAULT_OCTAVE = 2.0
MAX_HARMONIC_HZ = 12000.0
MIN_SUBHARMONIC_HZ = 16.0
PROXIMITY_THRESHOLD_HZ = 17.0
MIDI_MIN = 0
MIDI_MAX = 127
MIDI_A4 = 69
SEMITONES_PER_OCTAVE = 12

# Excel column constants
CUSTOM_COLUMN = "D"
HARM_COLUMN = "E"
SUB_COLUMN = "G"
TET_COLUMN = "I"
F0_COLUMN = "L"
FORMANT_COLUMN = "M"

# CSD file pattern matching
PATTERN = re.compile(r"\bf\s*(\d+)\b")

# Type definitions
Numeric = Union[int, float, Fraction]

# Reference frequency for TUN format
F_REF = 8.1757989156437073336
