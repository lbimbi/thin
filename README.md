# THIN – Sistemi di intonazione musicale / Tuning systems

Version: PHI • Release: 2025-09-06 • Author: LUCA BIMBI • License: MIT

This README is bilingual: Italian first, English follows.

---

## ITALIANO

THIN è un tool a riga di comando per generare e confrontare sistemi di intonazione.

Caratteristiche principali:
- Generazione di tabelle `cpstun` (GEN -2) in file `.csd` per Csound;
- Tabelle del sistema e di confronto in `.txt` e, se presente `openpyxl`, anche in `.xlsx`;
- Confronto con TET (12/24/48), serie armonica e subarmonica;
- Export opzionale di file `.tun` (AnaMark TUN);
- Analisi audio opzionale (F0 e formanti) tramite `librosa`, con indicatore di avanzamento a schermo.

Requisiti opzionali:
- Excel: `pip install openpyxl`
- Analisi audio: `pip install librosa scipy`

Uso rapido:
```bash
# default 12-TET su ottava, genera out.csd, out_system.*, out_compare.*
python3 thin.py out

# esporta anche .tun (AnaMark)
python3 thin.py --export-tun out

# analisi audio (LPC) su file voce
python3 thin.py --audio-file voce.wav --analysis lpc out_voice
```

Manuale completo: vedere [thin.md](thin.md).

Licenza: MIT. Vedi [LICENSE](LICENSE).

---

## ENGLISH

THIN is a command-line tool to generate and compare tuning systems.

Key features:
- Create `cpstun` (GEN -2) tables in `.csd` files for Csound;
- System and comparison tables in `.txt` and, if `openpyxl` is installed, `.xlsx`;
- Comparison against TET (12/24/48), harmonic and subharmonic series;
- Optional `.tun` (AnaMark TUN) export;
- Optional audio analysis (F0 & formants) via `librosa`, with on-screen dot progress.

Optional requirements:
- Excel: `pip install openpyxl`
- Audio analysis: `pip install librosa scipy`

Quick start:
```bash
# default 12-TET on octave; produces out.csd, out_system.*, out_compare.*
python3 thin.py out

# also export .tun
python3 thin.py --export-tun out

# audio analysis (LPC) on a voice file
python3 thin.py --audio-file voice.wav --analysis lpc out_voice
```

Full manual: see [thin.md](thin.md).

License: MIT. See [LICENSE](LICENSE).
