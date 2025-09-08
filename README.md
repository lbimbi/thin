# THIN – Sistemi di intonazione musicale / Tuning systems

Version: PHI • Release: 2025-09-07 • Author: LUCA BIMBI • License: MIT

This README is bilingual: Italian first, English follows.

---

## ITALIANO

THIN è un tool a riga di comando per generare e confrontare sistemi di intonazione.

Caratteristiche principali:
- Generazione di tabelle `cpstun` (GEN -2) in file `.csd` per Csound;
- Tabelle del sistema e di confronto in `.txt` e, se presente `openpyxl`, anche in `.xlsx`;
- Confronto con TET (12/24/48), serie armonica e subarmonica;
- Export opzionale di file `.tun` (AnaMark TUN);
- Analisi audio opzionale (F0 e formanti) tramite `librosa`, con indicatore di avanzamento a schermo;
- Grafico PNG diapason (con `--diapason-analysis`), richiede `matplotlib`;
- Render audio opzionale dei risultati di analisi (`--render`): WAV del tracking F0 e sinusoide A4 stimata (30s).
- Integrazione con archivi Scala (.scl): riconoscimento scala migliore, Top‑5 a schermo/Excel/testo e filtro per soglia con `--scala-cent` (elenchi entro X cents); inclusi nel foglio Excel “Diapason” (Scala_Map, Scala_Top5, Scala_Within).
  Nota: è necessario scaricare l'archivio ufficiale delle scale da https://www.huygens-fokker.org/docs/scales.zip e scompattare la cartella `scl/` nella stessa cartella dove si trova `thin.py`.
- Inferenza del sistema con confronto complementare: oltre al sistema primario, propone un sistema di famiglia diversa (regola: TET ↔ Rank‑1; Pyth → TET) per confronto.
- Migliorie di formattazione nel foglio “Diapason” (Excel): header, zebra striping, bordi, auto‑larghezze per migliore leggibilità.

Requisiti opzionali:
- Excel: `pip install openpyxl`
- Analisi audio: `pip install librosa scipy`
- Grafico PNG diapason: `pip install matplotlib`
- F0 HQ monofonico: `pip install crepe`

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

## Riferimenti

1. **Csound Reference Manual** - Documentazione opcode cpstun
2. **Csound. Guida al sound design in 20 lezioni**, Luca Bimbi, edizioni LSWR
3. **I numeri della musica**, Walter Branchi, Edipan edizioni

### Tabella Rapporti Comuni

| Sistema            | Rapporto | Cents | Intervallo |
|--------------------|----------|-------|------------|
| Ottava             | 2/1 | 1200 | P8 |
| Quinta giusta      | 3/2 | 701.96 | P5 |
| Quarta giusta      | 4/3 | 498.04 | P4 |
| Terza maggiore     | 5/4 | 386.31 | M3 |
| Terza minore       | 6/5 | 315.64 | m3 |
| Seconda maggiore   | 9/8 | 203.91 | M2 |
| Seconda minore     | 10/9 | 182.40 | m2 |
| Semitono diatonico | 16/15 | 111.73 | m2 |
| Comma sintonico    | 81/80 | 21.51 | - |

---

## ENGLISH

THIN is a command-line tool to generate and compare tuning systems.

Key features:
- Create `cpstun` (GEN -2) tables in `.csd` files for Csound;
- System and comparison tables in `.txt` and, if `openpyxl` is installed, `.xlsx`;
- Comparison against TET (12/24/48), harmonic and subharmonic series;
- Optional `.tun` (AnaMark TUN) export;
- Optional audio analysis (F0 & formants) via `librosa`, with on-screen progress;
- Diapason PNG plot (with `--diapason-analysis`), requires `matplotlib`;
- Optional audio rendering of analysis results (`--render`): WAV of F0 tracking and 30s estimated A4 sine.
- Scala (.scl) integration: best match recognition, Top‑5 shown in console/Excel/text, and threshold listing via `--scala-cent` (all scales within X cents); included in the Excel “Diapason” sheet (Scala_Map, Scala_Top5, Scala_Within).
  Note: you must download the official scales archive from https://www.huygens-fokker.org/docs/scales.zip and unzip the `scl/` folder into the same directory as `thin.py`.
- Comparative tuning inference: besides the primary fit, a complementary system from a different family is suggested (rule: TET ↔ Rank‑1; Pyth → TET) for cross‑checking.
- Improved “Diapason” sheet formatting (Excel): headers, zebra striping, cell borders, auto column widths for better readability.

Optional requirements:
- Excel: `pip install openpyxl`
- Audio analysis: `pip install librosa scipy`
- Diapason PNG plot: `pip install matplotlib`
- HQ monophonic F0: `pip install crepe`

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
