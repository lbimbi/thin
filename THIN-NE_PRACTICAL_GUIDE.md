# THIN-NE PRACTICAL GUIDE
## Complete User Guide for Musical Tuning System Generation and Analysis

### What is THIN-NE?

THIN-NE (Tonal Harmonic Intonation Network - Enhanced, Version PHI) is a comprehensive command-line tool for creating, analyzing, and converting musical tuning systems. It bridges the gap between mathematical tuning theory and practical musical application, offering powerful features for:

- **Composers**: Create and explore microtonal scales beyond 12-tone equal temperament
- **Researchers**: Analyze recordings to extract tuning systems from real performances
- **Musicians**: Generate custom tunings for electronic and acoustic instruments
- **Educators**: Demonstrate and compare different temperament systems
- **Sound Designers**: Build unique scales for film, games, and experimental music

### Installation

#### Basic Setup (Required)
```bash
python3 -m pip install openpyxl  # For Excel export
```

#### Audio Analysis (Recommended)
```bash
python3 -m pip install librosa scipy matplotlib  # Core audio
python3 -m pip install crepe tensorflow         # High-quality pitch tracking
```

### 5-Minute Quick Start

#### 1. Generate a Basic 12-TET Scale
```bash
./thin-ne.py my_scale
```
This creates `my_scale/my_scale.csd` with a standard 12-tone equal temperament.

#### 2. Create a 19-TET Microtonal Scale
```bash
./thin-ne.py --et 19 1200 --export-tun my_19tet
```
Generates 19 equal divisions of the octave with TUN file for DAW import.

#### 3. Analyze Audio and Extract Tuning
```bash
./thin-ne.py --audio-file singer.wav --diapason-analysis singer_scale
```
Analyzes the recording and extracts the tuning system used.

---

## COMPLETE FUNCTIONALITY GUIDE

### 1. TUNING SYSTEM GENERATION

THIN-NE offers four main methods for generating tuning systems, each suited for different musical and theoretical needs.

#### Equal Temperament (ET) - Dividing Intervals Equally

The `--et` parameter creates equal divisions of any interval, not just the octave. This is the foundation of modern Western music (12-TET) but can be extended to any number of divisions.

**Syntax**: `--et DIVISIONS INTERVAL`
- `DIVISIONS`: How many equal parts to divide the interval into
- `INTERVAL`: Size of the interval in cents (1200 = octave, 1902 = tritave, 702 = perfect fifth)

```bash
# Standard 12-tone equal temperament
./thin-ne.py --et 12 1200 standard_12tet

# 24-TET for quarter-tone music (Arab maqam, contemporary classical)
./thin-ne.py --et 24 1200 quarter_tones

# 19-TET - excellent for Renaissance music (better thirds than 12-TET)
./thin-ne.py --et 19 1200 --export-tun renaissance_19

# 31-TET - very close to 1/4-comma meantone temperament
./thin-ne.py --et 31 1200 --export-scl meantone_approximation

# 53-TET - approximates both Pythagorean and just intervals well
./thin-ne.py --et 53 1200 --export-ableton turkish_53

# Non-octave divisions: Bohlen-Pierce scale (13 equal divisions of tritave 3:1)
./thin-ne.py --et 13 1902 bohlen_pierce

# Wendy Carlos Alpha scale (78 cents per step, non-octave repeating)
./thin-ne.py --et 15 1170 carlos_alpha

# 7 equal divisions of a stretched octave (777 cents) across 3 spans
./thin-ne.py --et 7 777 --span 3 --basenote F#3 stretched_seventh
```

**When to use ET**:
- Creating systematic, reproducible scales
- Exploring non-Western tuning systems (Arab, Turkish, Thai)
- Approximating historical temperaments
- Experimental and electronic music

#### Geometric Progressions - Building from a Single Interval

The `--geometric` parameter generates scales by repeatedly stacking a single interval (the generator). This is how many traditional tuning systems were constructed.

**Syntax**: `--geometric GENERATOR STEPS INTERVAL`
- `GENERATOR`: The interval to stack (as ratio like 3/2 or decimal like 1.5)
- `STEPS`: How many times to stack the generator
- `INTERVAL`: The period for reduction (2/1 for octave, 3/1 for tritave)

```bash
# Pythagorean tuning: 12 perfect fifths (3:2 ratio)
./thin-ne.py --geometric 3/2 12 2/1 pythagorean_classic

# Circle of fifths with 7 notes (traditional diatonic scale)
./thin-ne.py --geometric 3/2 7 2/1 diatonic_pythagorean

# Quarter-comma meantone: slightly flat fifth (696.58 cents)
./thin-ne.py --geometric 1.49534878 11 2/1 quarter_comma_meantone

# 1/3-comma meantone: even flatter fifth for purer minor thirds
./thin-ne.py --geometric 1.49282033 12 2/1 third_comma_meantone

# Golden ratio scale (experimental/new age music)
./thin-ne.py --geometric 1.618034 7 2/1 golden_ratio_scale

# Slendro approximation using tempered fourths
./thin-ne.py --geometric 1.32 5 2/1 gamelan_slendro

# BP scale using 13 steps of specific ratio within tritave
./thin-ne.py --geometric 1.088447 13 3/1 bp_geometric
```

**When to use Geometric**:
- Recreating historical tuning systems
- Understanding the mathematical basis of scales
- Creating systematic but unequal temperaments
- Exploring non-octave-based music

#### Natural/Just Intonation - 5-Limit Lattice System

The `--natural` parameter generates a 5-limit just intonation lattice based on combinations of perfect fifths (3/2) and major thirds (5/4). This creates a two-dimensional grid of pure intervals.

**Syntax**: `--natural A_MAX B_MAX`
- `A_MAX`: Maximum power of 3/2 (perfect fifth) in both directions
- `B_MAX`: Maximum power of 5/4 (major third) in both directions

The system generates ratios using the formula: (3/2)^a × (5/4)^b
- where a ranges from -A_MAX to +A_MAX
- where b ranges from -B_MAX to +B_MAX

```bash
# Small 5-limit lattice (9 ratios)
./thin-ne.py --natural 1 1 small_lattice

# Medium lattice with extended fifths and thirds (25 ratios)
./thin-ne.py --natural 2 2 medium_lattice

# Standard 5-limit just intonation grid (121 ratios)
./thin-ne.py --natural 5 5 just_5_limit

# Extended lattice (169 ratios, exceeds MIDI limit of 128)
./thin-ne.py --natural 6 6 extended_lattice

# Asymmetric lattice emphasizing fifths over thirds
./thin-ne.py --natural 7 3 fifth_emphasis

# Focus on thirds with limited fifths
./thin-ne.py --natural 2 5 third_emphasis
```

**What this generates**:
- A comprehensive network of just intervals
- Includes major/minor thirds, perfect fourths/fifths, major/minor sixths
- Creates comma variants (syntonic comma differences)
- Produces wolf intervals at the edges of the lattice

**When to use Natural**:
- Creating complex just intonation systems
- Building Harry Partch-style lattices
- Exploring comma relationships
- Academic study of 5-limit tuning theory
- Barbershop and a cappella arrangements requiring pure harmonies

#### Danielou Microtonal System - Complex Rational Tuning

The `--danielou` system creates scales using combinations of prime factors 2, 3, and 5, based on Alain Daniélou's theories.

**Syntax**: `--danielou "a,b,c"` where ratio = 2^a × 3^b × 5^c

```bash
# Single custom ratio (major third = 2^-2 × 3^0 × 5^1 = 5/4)
./thin-ne.py --danielou "-2,0,1" major_third_only

# Multiple ratios for a custom scale
./thin-ne.py --danielou "0,0,1" --danielou "1,1,0" --danielou "-1,0,1" \
    --danielou "2,-1,0" --danielou "-2,2,0" custom_danielou_scale

# Complete 53-comma Danielou system (all combinations within limits)
./thin-ne.py --danielou-all danielou_complete_53

# Indian shruti system approximation
./thin-ne.py --danielou "-4,2,1" --danielou "3,-1,0" --danielou "-2,1,1" \
    --danielou "1,0,0" --danielou "-3,0,2" indian_shruti
```

**When to use Danielou**:
- Creating complex just intonation systems
- Indian classical music (ragas)
- Exploring prime-limit tuning theory
- Academic research in microtonality

### 2. AUDIO ANALYSIS - Extracting Tuning from Real Music

THIN-NE can analyze audio recordings to extract the tuning system being used. This is invaluable for ethnomusicology, historical performance research, and understanding how musicians actually tune in practice.

#### Analysis Methods

**LPC (Linear Predictive Coding)** - Best for vocals and monophonic instruments
The LPC method models the vocal tract and is excellent at extracting fundamental frequencies from singing or solo instruments. It's particularly effective when analyzing:
- Solo vocals
- Wind instruments (flute, clarinet, saxophone)
- Bowed strings (violin, cello)
- Traditional instruments with clear pitch

```bash
# Basic LPC analysis for a vocal recording
./thin-ne.py --audio-file singer.wav --analysis lpc vocal_analysis

# LPC with custom parameters for cleaner extraction
./thin-ne.py --audio-file flute.wav --analysis lpc \
    --lpc-order 24 --lpc-preemph 0.97 --lpc-window-ms 30 flute_scale
```

**CQT (Constant-Q Transform)** - Best for polyphonic and complex harmonic content
The CQT method provides logarithmic frequency resolution, matching how we hear music. It excels with:
- Piano and keyboard instruments
- Guitar and plucked strings
- Ensemble recordings
- Bell-like and metallic timbres

```bash
# Basic CQT analysis for piano
./thin-ne.py --audio-file piano.wav --analysis cqt piano_scale

# CQT with fine-tuned parameters for gamelan metallophones
./thin-ne.py --audio-file gamelan.wav --analysis cqt \
    --cqt-bins 84 --cqt-bpo 48 --cqt-fmin 55 gamelan_analysis
```

#### Diapason Analysis - Complete Tuning System Extraction

The `--diapason-analysis` flag performs comprehensive analysis to:
1. Detect the A4 reference frequency (diapason)
2. Extract all pitch frequencies from the audio
3. Cluster frequencies into scale degrees
4. Infer the tuning system type (ET, Pythagorean, Just, etc.)
5. Find matches in the Scala database
6. Generate detailed reports with confidence intervals

```bash
# Complete analysis with all reports
./thin-ne.py --audio-file recording.wav --diapason-analysis full_analysis

# This generates:
# - full_analysis_diapason.txt: Detailed frequency analysis
# - full_analysis_f0.png: Visual frequency plot
# - full_analysis_cqt.png: CQT Plot
# - full_analysis_system.xlsx: Excel workbook with all data
# - full_analysis.csd: Csound file with extracted tuning
```

#### Analysis Parameters Explained

**Frame and Window Parameters**:
- `--frame-size`: Number of samples per analysis window (larger = better frequency resolution)
- `--hop-size`: Samples between frames (smaller = better time resolution)
- `--window`: Window function (hamming for general use, hanning for better frequency resolution)

```bash
# High resolution analysis for slow, sustained tones
./thin-ne.py --audio-file drone.wav --analysis lpc \
    --frame-size 4096 --hop-size 512 --window hanning drone_analysis

# Fast tracking for rapidly changing pitches
./thin-ne.py --audio-file ornaments.wav --analysis cqt \
    --frame-size 1024 --hop-size 128 --window hamming fast_tracking
```

**Clustering Parameters** - Control how frequencies are grouped into scale degrees:
- `--tolerance`: Percentage tolerance for clustering (default 2.0%)
- `--min-ratio-count`: Minimum occurrences to include a ratio
- `--confidence-threshold`: CREPE confidence filter (0.0-1.0)
- `--clustering-method`: Algorithm for grouping (simple/weighted/adaptive)
- `--scale-size-hint`: Expected number of scale degrees
- `--duration-weighting`: Weight frequencies by duration
- `--harmonic-bias`: Favor simple integer ratios

```bash
# Tight clustering for clean studio recording
./thin-ne.py --audio-file studio.wav --diapason-analysis \
    --tolerance 1.0 --min-ratio-count 5 --confidence-threshold 0.95 \
    --clustering-method simple studio_precise

# Adaptive clustering for noisy field recording
./thin-ne.py --audio-file field.wav --diapason-analysis \
    --tolerance 3.0 --min-ratio-count 2 --confidence-threshold 0.6 \
    --clustering-method adaptive --scale-size-hint 7 field_adaptive

# Duration-weighted for ensemble where important notes are held longer
./thin-ne.py --audio-file choir.wav --diapason-analysis \
    --clustering-method weighted --duration-weighting choir_weighted
```

#### Special Analysis Features

**F0-Only Mode**: Skip CQT analysis for faster processing
```bash
./thin-ne.py --audio-file quick.wav --diapason-analysis --f0-only quick_f0
```

**Audio Rendering**: Generate audio files of the extracted frequencies
```bash
./thin-ne.py --audio-file original.wav --diapason-analysis --render rendered
# Creates rendered_audio_f0.wav and rendered_audio_formants.wav
```

**Precision Control**: Set decimal places for display
```bash
./thin-ne.py --audio-file precise.wav --diapason-analysis --digits 6 precise_6dp
```

**Cross-Inference**: Use different sources for F0 and harmonic analysis
```bash
# Use solo instrument for F0, ensemble for harmonic context
./thin-ne.py --audio-file ensemble.wav --cross-inference solo.wav \
    --diapason-analysis cross_analysis
```

### 3. SCALA DATABASE INTEGRATION

The Scala scale archive contains over 5000 historical, theoretical, and experimental tuning systems. THIN-NE can search this database to identify scales or find inspiration.

#### Setting Up Scala Database

1. Download the archive from https://www.huygens-fokker.org/docs/scales.zip
2. Extract to a `scl/` directory in your THIN-NE folder
3. The database is now searchable

#### Scala Matching Features

**Find scales similar to your audio analysis**:
The `--scala-cent` parameter searches for scales within a specified error tolerance (in cents).

```bash
# Find all scales within 5 cents average error
./thin-ne.py --audio-file recording.wav --diapason-analysis --scala-cent 5 close_matches

# More tolerant search for ethnomusicological recordings
./thin-ne.py --audio-file field.wav --diapason-analysis --scala-cent 15 field_matches

# The output shows:
# - Top 5 best matches with average error
# - Scale names and descriptions
# - File names for further investigation
```

**Force export using Scala match instead of inferred system**:
```bash
# Use the best Scala match for export (overrides inferred tuning)
./thin-ne.py --audio-file gamelan.wav --diapason-analysis \
    --scala --export-tun --export-scl scala_based

# Use your own inferred system (default)
./thin-ne.py --audio-file gamelan.wav --diapason-analysis \
    --inferred --export-tun --export-scl inferred_based
```

#### Understanding Scala Search Results

When THIN-NE finds Scala matches, it reports:
- **Position**: Rank from best to worst match
- **Name**: Descriptive name of the scale
- **Error (cents)**: Average deviation from your tuning
- **File**: The .scl filename for manual inspection

Example output:
```
Top 5 matches:
Pos  Name                                   Err (c)    File
  1  Slendro Sunda                           3.42      slendro_sunda.scl
  2  Gamelan Slendro, Kyahi Kanyut           5.18      gamelan_kanyut.scl
  3  Pelog approximation, Kunst              8.73      pelog_kunst.scl
```

### 4. FILE EXPORT FORMATS - Getting Your Tuning Into Software

THIN-NE exports to multiple formats, each optimized for different software and hardware.

#### Csound Format (.csd) - Always Generated

Every THIN-NE run creates a Csound file with cpstun tables. This is the most complete format, preserving all tuning information.

**File structure**:
```csound
<CsoundSynthesizer>
<CsInstruments>
; Your orchestra code here
</CsInstruments>
<CsScore>
; cpstun table automatically inserted:
f 1 0 25 -2 12 2 440 60 1.0 1.059463 1.122462 1.189207 ...
</CsScore>
</CsoundSynthesizer>
```

**Using in Csound**:
- The table number increments if file exists (f 1, f 2, f 3...)
- Use `cpstun` opcode to get frequencies: `ifreq cpstun inote, 1`
- Interval parameter controls octave repetition

#### AnaMark TUN Format (.tun) - Hardware Compatibility

TUN files map all 128 MIDI notes to specific frequencies. Essential for hardware synths and many VST plugins.

**Export options**:
```bash
# Standard TUN with 2 decimal precision
./thin-ne.py --et 19 1200 --export-tun standard_19tet

# Integer cents only (some hardware requires this)
./thin-ne.py --et 19 1200 --export-tun --tun-integer integer_19tet

# Custom diapason reflected in TUN file with a 5 octave generation
./thin-ne.py --et 12 1200 --diapason 432 --export-tun --span 5 a432_tun
```

**TUN file structure**:
```
[Tuning]
Note 0=-3600
Note 1=-3500
...
Note 60=0      ; Middle C at base frequency
Note 61=63     ; 63 cents above middle C (for 19-TET)
...
Note 127=8700
```

**Compatible with**:
- Native Instruments FM8, Massive
- u-he synths (Zebra2, Diva, etc.)
- Pianoteq
- Many hardware synthesizers via MIDI

#### Scala Format (.scl) - Universal Exchange

Scala is the most widely supported microtonal format, readable by hundreds of applications.

**Export examples**:
```bash
# Basic export
./thin-ne.py --natural 7 7 --export-scl just_7limit

# With custom description
./thin-ne.py --geometric 3/2 12 2/1 --export-scl \
    --basenote A4 --diapason 415 baroque_pythagorean
```

**SCL file structure**:
```scala
! my_scale.scl
! Generated by THIN-NE
My Custom Scale, 1/1=440.00 Hz, base=A4
12
 111.73
 203.91
 315.64
 386.31
 498.04
 590.22
 701.96
 813.69
 884.36
 996.09
1088.27
   2/1
```

**Compatible with**:
- Scala software (over 100 applications)
- Max/MSP
- SuperCollider
- Most microtonal software

#### Ableton Live Format (.ascl) - Modern DAW Integration

ASCL extends Scala with Ableton-specific metadata for perfect integration with Push controllers and Live's interface.

**Export with metadata**:
```bash
# Basic ASCL export
./thin-ne.py --et 17 1200 --export-ableton et17_ableton

# With specific base note (affects Push display)
./thin-ne.py --natural 5 5 --basenote D3 --export-ableton just_in_d
```

**ASCL additions to Scala**:
- Note names with microtonal symbols (↑↓ for deviations)
- Reference pitch specification
- Note range for Push controller
- Source documentation

**Using in Ableton**:
1. Place .ascl file in User Library/Tuning System/
2. Load in any Ableton instrument (Operator, Analog, Wavetable, etc.)
3. Push controller automatically shows custom note names

#### Excel Format (.xlsx) - Analysis and Documentation

Excel files are generated automatically when using comparison features, providing comprehensive analysis.

**Automatic generation triggers**:
```bash
# Comparison with 12-TET
./thin-ne.py --et 31 1200 --compare-tet 12 analysis_31

# Comparison with custom fundamental
./thin-ne.py --natural 7 7 --compare-fund 256 analysis_c

# Diapason analysis always creates Excel
./thin-ne.py --audio-file recording.wav --diapason-analysis recording
```

**Excel workbook contains**:
- **System** worksheet: Generated tuning with frequencies, cents, ratios
- **Compare** worksheet: Side-by-side comparison with reference tunings
- Color coding: Visual deviation indicators
- Formulas: Live calculations for custom modifications

**Features**:
- Zebra striping for readability
- Conditional formatting shows intervallic relationships
- Charts for visual analysis
- Print-ready formatting

### 5. PRACTICAL WORKFLOWS

#### Workflow 1: Analyzing Traditional Music
```bash
# Step 1: Extract scale from field recording
./thin-ne.py --audio-file field_recording.wav --diapason-analysis \
    --clustering-method adaptive --harmonic-bias field_analysis

# Step 2: Find closest historical scale
./thin-ne.py --audio-file field_recording.wav --diapason-analysis \
    --scala-cent 5 field_matches

# Step 3: Export for use in composition
./thin-ne.py --audio-file field_recording.wav --diapason-analysis \
    --inferred --export-tun --export-scl --export-ableton field_final
```

#### Workflow 2: Creating Microtonal Compositions
```bash
# Design custom scale with specific intervals
./thin-ne.py --geometric 11/10 8 3/2 --basenote A3 --diapason 432 \
    --export-tun --export-ableton custom_scale

# Convert existing tuning from Excel
./thin-ne.py --convert my_ratios.xlsx --export-tun converted_scale

# Import and modify existing TUN file
./thin-ne.py --import-tun hardware_preset.tun
# Edit the generated ratios file, then:
./thin-ne.py --convert hardware_preset_ratios.txt modified_preset
```

#### Workflow 3: Academic Research
```bash
# Comprehensive analysis with all metrics
./thin-ne.py --audio-file performance.wav --diapason-analysis \
    --digits 3 --render --compare-tet 12 --compare-fund 440 \
    --subharm-fund 880 research_data

# Batch process multiple files
for file in recordings/*.wav; do
    name=$(basename "$file" .wav)
    ./thin-ne.py --audio-file "$file" --diapason-analysis \
        --export-scl "analyzed/$name"
done
```

### 6. ADVANCED CONTROL FEATURES

#### Reference Pitch and Base Note Configuration

**Diapason (A4 Reference)**:
The `--diapason` parameter sets the A4 reference frequency. Historical and regional standards vary widely.

```bash
# Modern standard A4=440 Hz (default)
./thin-ne.py --et 12 1200 modern_standard

# Baroque pitch A4=415 Hz (semitone lower)
./thin-ne.py --et 12 1200 --diapason 415 baroque_pitch

# French Baroque A4=392 Hz (tone lower)
./thin-ne.py --et 12 1200 --diapason 392 french_baroque

# Verdi tuning A4=432 Hz (controversial "natural" tuning)
./thin-ne.py --et 12 1200 --diapason 432 verdi_tuning

# Classical pitch A4=430 Hz (19th century)
./thin-ne.py --et 12 1200 --diapason 430 classical_pitch

# Scientific pitch C4=256 Hz (A4≈430.54)
./thin-ne.py --et 12 1200 --diapason 430.54 scientific_c256
```

**Base Note Specification**:
The `--basenote` parameter sets which note serves as the tuning reference. Supports multiple input formats.

```bash
# Standard note names
./thin-ne.py --et 19 1200 --basenote C4 c_based
./thin-ne.py --et 19 1200 --basenote F#3 f_sharp_based
./thin-ne.py --et 19 1200 --basenote Bb2 b_flat_based

# Microtonal adjustments (+ = +50¢, - = -50¢, ! = +25¢, . = -25¢)
./thin-ne.py --et 24 1200 --basenote "C+4" c_plus_50
./thin-ne.py --et 24 1200 --basenote "A!3" a_plus_25
./thin-ne.py --et 24 1200 --basenote "G.5" g_minus_25
./thin-ne.py --et 24 1200 --basenote "D-2" d_minus_50

# Direct frequency specification (Hz)
./thin-ne.py --et 12 1200 --basenote 261.626 middle_c_freq
./thin-ne.py --et 12 1200 --basenote 528 solfeggio_frequency
./thin-ne.py --et 12 1200 --basenote 256 scientific_c
```

**MIDI Base Key**:
The `--basekey` parameter specifies the MIDI note number for the base (default 60 = Middle C).

```bash
# Different MIDI base points
./thin-ne.py --et 12 1200 --basekey 48 c3_base    # C3 as base
./thin-ne.py --et 12 1200 --basekey 69 a4_base    # A4 as base
./thin-ne.py --et 12 1200 --basekey 36 c2_base    # C2 as base
```

#### Span, Repetition and Octave Control

**Span/Ambitus** - Repeat the pattern multiple times:
```bash
# Single octave (default)
./thin-ne.py --et 12 1200 single_octave

# Three octave range
./thin-ne.py --et 12 1200 --span 3 three_octaves

# Five octave range
./thin-ne.py --et 12 1200 --span 5 5_range

# Extended range for orchestral writing
./thin-ne.py --et 12 1200 --span 7 orchestral_range
```

**Octave Reduction Control**:
```bash
# Normal behavior: reduce all ratios to single octave [1,2)
./thin-ne.py --geometric 3/2 12 2/1 reduced

# No reduction: show raw calculated ratios
./thin-ne.py --geometric 3/2 12 2/1 --no-reduce unreduced

# Example: Pythagorean comma becomes visible
# Without --no-reduce: all ratios between 1.0 and 2.0
# With --no-reduce: 12th fifth = 129.746... (not reduced)
```

**Interval Control for Csound**:
```bash
# Normal: interval=2 for octave repetition in cpstun
./thin-ne.py --natural 5 5 normal_repeat

# Force interval=0 (no automatic octave extension)
./thin-ne.py --natural 5 5 --interval-zero no_repeat
```

#### Comparison and Analysis Features

**Compare with Equal Temperaments**:
The `--compare-tet` parameter generates comparison tables against 12, 24, or 48-TET.

```bash
# Compare 19-TET with standard 12-TET
./thin-ne.py --et 19 1200 --compare-tet 12 et19_vs_12

# Compare just intonation with 12-TET
./thin-ne.py --natural 5 5 --compare-tet 12 just_vs_12

# Compare with quarter-tone 24-TET
./thin-ne.py --et 31 1200 --compare-tet 24 et31_vs_24

# High-resolution comparison with 48-TET
./thin-ne.py --geometric 3/2 12 2/1 --compare-tet 48 pyth_vs_48
```

**TET Alignment Options**:
```bash
# Align to same scale degree (default)
./thin-ne.py --et 19 1200 --compare-tet 12 --compare-tet-align same aligned_same

# Align to nearest pitch
./thin-ne.py --et 19 1200 --compare-tet 12 --compare-tet-align nearest aligned_nearest
```

**Custom Fundamental Comparison**:
```bash
# Compare with different fundamental frequency
./thin-ne.py --et 12 1200 --compare-fund 256 compare_c256

# Multiple fundamentals for harmonic analysis
./thin-ne.py --natural 7 7 --compare-fund 100 --subharm-fund 200 harmonic_analysis
```

**Subharmonic Analysis**:
```bash
# Analyze subharmonic relationships
./thin-ne.py --et 12 1200 --subharm-fund 880 subharm_a5

# Combined harmonic/subharmonic
./thin-ne.py --natural 5 5 --compare-fund 220 --subharm-fund 880 full_harmonic
```

#### File Import and Conversion

**Import TUN Files**:
Extract ratios from existing TUN files for modification.

```bash
# Import hardware synthesizer tuning
./thin-ne.py --import-tun my_synth.tun

# This creates my_synth_ratios.txt with MIDI->ratio mappings
```

**Convert Excel Files**:
Transform Excel tuning data into multiple formats.

```bash
# Basic conversion
./thin-ne.py --convert tuning_data.xlsx converted

# With multiple export formats
./thin-ne.py --convert tuning_data.xlsx --export-tun --export-scl --export-ableton all_formats

# Apply cents shift to all pitches
./thin-ne.py --convert tuning_data.xlsx --cents-delta 10 shifted_up_10c
```

**MIDI Range Control**:
```bash
# Ensure ratios fit within MIDI range (may transpose)
./thin-ne.py --geometric 3/2 50 2/1 --midi-truncate midi_safe

# Without truncation (may exceed MIDI range)
./thin-ne.py --geometric 3/2 50 2/1 full_range
```

#### Cross-Inference Analysis

Cross-inference uses different audio sources for pitch detection and harmonic analysis, useful for separating melody from harmony. It also infers F0 with CQT analysis.

```bash
# Basic cross-inference
./thin-ne.py --audio-file ensemble.wav --cross-inference solo.wav \
    --diapason-analysis cross_basic

# Use solo for clean F0, ensemble for harmonic context
./thin-ne.py --audio-file orchestra.wav --cross-inference violin_solo.wav \
    --diapason-analysis --clustering-method adaptive orchestral_analysis

# Vocal melody with instrumental harmony
./thin-ne.py --audio-file full_mix.wav --cross-inference vocal_only.wav \
    --diapason-analysis --render vocal_instrumental
```

**When to use cross-inference**:
- Solo instrument plays melody while ensemble provides harmony
- Vocal line needs separation from accompaniment
- Reference pitch from one source, scale from another
- Combining multiple recordings of the same piece

### 7. TROUBLESHOOTING

#### Common Issues and Solutions

**Problem**: "Missing dependencies" error
```bash
# Install required packages
python3 -m pip install librosa scipy matplotlib crepe tensorflow openpyxl
```

**Problem**: Scala matching not working
```bash
# Download and extract Scala archive
wget https://www.huygens-fokker.org/docs/scales.zip
unzip scales.zip -d scl/
```

**Problem**: Audio analysis produces no results
```bash
# Try different analysis method
./thin-ne.py --audio-file difficult.wav --analysis lpc  # Instead of cqt

# Adjust confidence threshold
./thin-ne.py --audio-file noisy.wav --diapason-analysis \
    --confidence-threshold 0.5  # Lower threshold for noisy audio
```

**Problem**: TUN file shows wrong base note in DAW
```bash
# This is now fixed in latest version. Re-generate your TUN files:
./thin-ne.py --et 7 777 --basenote c3 --export-tun fixed_tun
```

### 8. TIPS AND BEST PRACTICES

#### For Composers
- Use `--export-ableton` for modern DAW integration
- Set custom diapason with `--diapason 432` for alternative tunings
- Use `--basenote` with microtonal notation for precise control

#### For Researchers
- Always use `--diapason-analysis` for complete analysis
- Add `--digits 6` for publication-ready precision
- Use `--render` to generate audio verification files

#### For Live Performance
- Export to TUN format for hardware synth compatibility
- Use `--midi-truncate` to ensure MIDI range compatibility
- Test with `--compare-tet 12` to see deviations from standard tuning

### 9. COMMAND REFERENCE CARD

```
GENERATION          ANALYSIS            EXPORT
--et N I            --audio-file F      --export-tun
--geometric G S I   --analysis M        --export-scl
--natural A B       --diapason-analysis --export-ableton
--danielou "a,b,c"  --scala-cent N
--danielou-all      --render

CONTROL             COMPARISON          CLUSTERING
--diapason Hz       --compare-tet N     --tolerance %
--basenote Note     --compare-fund Hz   --min-ratio-count N
--basekey MIDI      --subharm-fund Hz   --confidence-threshold
--span N            --scala             --clustering-method
--no-reduce         --inferred          --harmonic-bias
```

### 10. QUICK EXAMPLES

```bash
# Standard 12-TET in C
./thin-ne.py standard_12

# 31-TET for early music
./thin-ne.py --et 31 1200 --export-tun early_music

# Extract scale from recording
./thin-ne.py --audio-file recording.wav --diapason-analysis extracted

# Pythagorean tuning
./thin-ne.py --geometric 3/2 12 2/1 pythag

# Just intonation triad
./thin-ne.py --natural 6 6 just

# Convert Excel to all formats
./thin-ne.py --convert tuning.xlsx --export-tun --export-scl --export-ableton all_formats
```

---

## Support and Resources

- **Source Code**: See repository for latest updates
- **Scala Database**: https://www.huygens-fokker.org/scala/
- **Csound Documentation**: https://csound.com/docs/manual/cpstun.html
- **Issue Reports**: File issues in the project repository

Remember: THIN-NE is a research tool. Always verify results by ear and cross-reference with established tuning references when accuracy is critical.