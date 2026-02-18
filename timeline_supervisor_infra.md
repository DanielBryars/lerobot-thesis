# Thesis Experiments Timeline (Infrastructure / Tooling)

Audience: supervisor. Chronological, headline-level. Focus: evaluation/training infrastructure issues that materially affected results.

Columns: ID | When | Area | Issue / Change | Impact | Status

---

| ID | When | Area | Issue / Change | Impact | Status |
|---:|:-----|:-----|:---------------|:-------|:-------|
| I01 | 2026-01-06 | Eval pipeline | Multiple eval bugs (camera keys, depth naming, processor usage, missing policy.reset) | Large success underestimates (e.g. 23% -> 73%) | Fixed |
| I02 | 2026-01-08..11 | vast.ai ops | Disk mounts misleading; openpi checkpoints huge | Training aborted (“No space left”) | Mitigated with disk checklist |
| I03 | 2026-01-18 | openpi JAX on RTX 5090 | Instability/crashes under WSL2; too slow with autotune off | Abandoned local JAX inference | Pivoted to PyTorch |
| I04 | 2026-01-18..19 | LeRobot Pi0 on Windows | HF repo-id path separator bug (`\` vs `/`) | Model download/load failures | Patched |
| I05 | 2026-01-19 | Windows console | `PYTHONIOENCODING=utf-8` required (unicode checkmark logging) | Silent load issues / random weights risk | Workaround documented |
| I06 | 2026-01-19 | Pi0 inference | Missing postprocessor (denormalization) | 0% success despite “reasonable” actions | Fixed |
| I07 | 2026-01-20 | Whisker viz | Double-prediction bug (visualized != executed due to VAE sampling) | Misleading diagnostics | Fixed |
| I08 | 2026-01-20..22 | Viz tooling | Added whisker visualization, temporal ensemble viz/recording | Faster debugging + thesis figures | Completed |
| I09 | 2026-01-30 | Training resume | `utils/training.py:load_checkpoint()` did not load model weights on resume | Any resumed run effectively restarted with random weights | Fixed |
