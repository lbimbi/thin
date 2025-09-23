# THIN-NE: Advanced Musical Tuning Systems Tool

**Complete Musical Intonation Analysis and Generation System**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Audio Analysis](https://img.shields.io/badge/audio-analysis-green.svg)](https://librosa.org/)

THIN-NE is a comprehensive tool for generating and analyzing musical tuning systems. Version PHI provides advanced audio analysis, diapason estimation, and integration with historical scale databases for research, composition, and analysis of microtonal and alternative tuning systems.

## Key Features

- **Audio Analysis**: F0 tracking with CREPE neural networks, formant analysis (CQT/LPC)
- **Smart Clustering**: Adaptive algorithms with tolerance reporting in both percentage and cents
- **Diapason Estimation**: Automatic A4 detection with confidence intervals
- **Scala Integration**: 5000+ historical scales matching
- **Multi-Format Export**: Csound (.csd), AnaMark TUN, Scala (.scl), Ableton Live (.ascl), Excel
- **Tuning Systems**: Equal temperament, geometric progressions, natural intonation, Danielou microtonal

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lucabimbi/THIN.git
cd THIN

# Install core dependencies
pip install openpyxl

# Install audio analysis dependencies (recommended)
pip install librosa scipy matplotlib

# Install high-quality F0 analysis (optional)
pip install crepe tensorflow
```

### Basic Usage

```bash
# Generate 19-TET tuning system
python3 thin-ne.py --et 19 1200 my_19tet

# Analyze audio file for tuning characteristics
python3 thin-ne.py --audio-file voice.wav --diapason-analysis analysis_output

# Generate just intonation with TUN export
python3 thin-ne.py --natural 4 8 --export-tun just_intonation

# Analyze with Scala database matching
python3 thin-ne.py --audio-file historical.wav --diapason-analysis --scala-cent 2.0 scala_match
```

## Supported Tuning Systems

- **Equal Temperament**: Any division of any interval (12-TET, 19-TET, 31-TET, etc.)
- **Geometric Progressions**: Pythagorean, meantone, well temperaments
- **Natural Intonation**: 4:5:6 harmonic series, just intonation with prime limits
- **Danielou Microtonal**: Systematic 53-comma microtonal grid
- **Custom Systems**: Import from TUN files, audio analysis inference

## Audio Analysis Features

### Advanced F0 Tracking
- **YIN Algorithm**: Standard autocorrelation-based detection
- **CREPE Neural Network**: High-quality pitch detection with confidence scores
- **Multi-modal Correlation**: F0 and formant cross-validation

### Clustering with Tolerance Reporting
THIN automatically reports clustering tolerance impact:
```
Clustering tolerance: 2.0% relative ≈ 34.6 cents
Clustering method: adaptive
F0 tracking: CREPE neural network analysis
Estimated A4: 442.3 Hz ± 1.2 Hz
```

### Comprehensive Scale Inference
- **Pattern Recognition**: Multi-family tuning system identification
- **Scala Database**: 5000+ historical scales matching
- **Statistical Analysis**: Confidence intervals and error metrics
- **Visualization**: Diagnostic plots and spectrograms

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| **Csound** | `.csd` | Audio synthesis with `cpstun` opcode |
| **AnaMark TUN** | `.tun` | Integration with software and hardware synthesizers |
| **Scala** | `.scl` | Microtonal community standard |
| **Ableton Live** | `.ascl` | Native Live Scale component integration |
| **Excel** | `.xlsx` | Analysis reports with multiple worksheets |
| **Audio** | `.wav` | Reference tones and F0 renderings |
| **PNG Plots** | `.png` | Visualization and diagnostic graphics |

## Project Structure

```
THIN/
├── thin-ne.py              # Main THIN application (PHI version)
├── audio_analysis/         # Audio processing and F0/formant analysis
├── consts/                 # Configuration constants
├── sim/                    # Core tuning system mathematics
├── tables/                 # Data export and table generation
├── tun_csd/               # Format-specific export handlers
├── utils/                 # Common utilities (note parsing, MIDI, I/O)
└── scl/                   # Scala database (download separately)
```

## 🔧 Advanced Features

### Audio Analysis Pipeline
1. **Multi-format Loading**: WAV, MP3, FLAC via librosa
3. **F0 Extraction**: YIN or CREPE-based fundamental frequency tracking
4. **Formant Analysis**: CQT (musical) or LPC (vocal) spectral analysis
5. **Clustering**: Smart frequency grouping with statistical validation
6. **Pattern Matching**: Scale inference and Scala database comparison
7. **Visualization**: Comprehensive diagnostic plots and reports

### Professional Integration
- **Csound**: GEN -2 tables with exact frequency ratios
- **DAW Compatibility**: Industry-standard TUN format with exact tuning
- **Research Tools**: Excel reports with statistical analysis
- **Live Performance**: Ableton Live integration for real-time microtuning

## Documentation

- **[Comprehensive Manual](THIN_COMPREHENSIVE_MANUAL_EN.md)**: Complete textbook-style documentation (**available soon**)

## Use Cases

### Research & Academia
- Historical tuning analysis and reconstruction
- Cross-cultural musical scale studies
- Performance practice research
- Psychoacoustic tuning perception studies

### Composition & Production
- Microtonal music composition
- Alternative tuning exploration
- DAW integration for microtonal production
- Live performance with custom scales

### Analysis & Education
- Musical performance intonation analysis
- Ensemble tuning assessment
- Educational demonstrations of tuning systems
- Comparative tuning system studies

## Dependencies

### Required (Core)
- Python 3.8+
- openpyxl (Excel export)

### Recommended (Audio Analysis)
- librosa (audio processing)
- scipy (scientific computing)
- matplotlib (visualization)

### Optional (High-Quality F0)
- crepe (neural F0 estimation)
- tensorflow (CREPE backend)

### External Resources
- **Scala Database**: Download from [Huygens-Fokker Foundation](https://www.huygens-fokker.org/docs/scales.zip)

## Example Workflows

### Historical Analysis
```bash
# Analyze baroque harpsichord tuning
python3 thin-ne.py --audio-file baroque.wav \
                   --analysis cqt \
                   --diapason-analysis \
                   --scala-cent 3.0 \
                   --clustering-method adaptive \
                   --scale-size-hint 12 \
                   baroque_analysis
```

### Microtonal Composition
```bash
# Generate 31-TET system with multiple exports
python3 thin-ne.py --et 31 1200 \
                   --export-tun \
                   --export-scl \
                   --export-ableton \
                   microtonal_31tet
```

### Vocal Performance Analysis
```bash
# Analyze choir intonation with weighted clustering
python3 thin-ne.py --audio-file choir.wav \
                   --analysis lpc \
                   --diapason-analysis \
                   --clustering-method weighted \
                   --duration-weighting \
                   --harmonic-bias \
                   --render \
                   choir_intonation
```

## Output Organization

THIN automatically creates organized output directories:

```
analysis_output/
├── analysis_output.csd              # Csound table
├── analysis_output_system.xlsx      # System analysis
├── analysis_output_compare.xlsx     # Comparison table
├── analysis_output_diapason.txt     # Diapason report
├── analysis_output_diapason.png     # F0 tracking plot
├── analysis_output_diapason.wav     # Reference tone
└── analysis_output.tun              # TUN file (if exported)
```

## Contributing

Contributions are welcome! Areas of interest:
- Additional tuning system generators
- New export format support
- Audio analysis algorithm improvements
- Documentation and examples
- Test suite development

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **librosa** team for comprehensive audio analysis
- **CREPE** authors for neural pitch detection
- **Huygens-Fokker Foundation** for maintaining the Scala archive
- **Csound** community for the cpstun tuning framework
- Global microtonal music research community

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/lucabimbi/SIM/issues)
- **Documentation**: See included manuals
- **Community**: Connect with the microtonal music community

---

**THIN** - Bridging traditional music theory with computational analysis for the exploration of musical tuning systems across cultures and time periods.