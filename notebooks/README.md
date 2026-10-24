# Notebook

`BTP_pipeline_executed.ipynb` is the notebook that produced every reported number (Colab, GPU for FinBERT, 10 seeds). Widget metadata was stripped and markdown headings added.

Things to know before relying on it:

* **No cell has an error output**, but execution counts are non-consecutive (1-5, then 13-19) because earlier cells of the original session were removed. Treat it as the recorded run, not a clean top-to-bottom re-execution.
* **Python >= 3.12 is required** (cell 4 uses same-quote nested f-strings, a SyntaxError on 3.10/3.11). The `language_info` metadata was corrected accordingly. `src/01_data.py` has the portable version.
* Some printed prompts still say "paste this output back to your assistant"; they are leftovers from AI-assisted development and have no effect.
* It uses `LAG = 0` throughout and contains no M7; M7 and LAG = 1 exist only in `src/04_models.py`.
* M0r and M6 were added after a pilot (report Appendix A). The class-balance diagnostics (Table 5.7) are also in `src/05_analysis.py` (Cell 2b).
* Row-bootstrap intervals need the saved predictions; protocol-gap intervals use 2,000 seed resamples, row bootstrap 1,000.

The same pipeline is split across `src/01_data.py` ... `src/05_analysis.py`. The earlier empty stub notebooks were removed.
