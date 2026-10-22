# Report

# Report source and build status

* **Submission source:** `latex/main.tex` and the chapter files under `latex/chapters/`. These are the report files to edit and compile.
* `BTP_Report.pdf` is a previously compiled snapshot; after changing the LaTeX sources, recompile the PDF and check the output before submission.
* `results_draft.md` is a working-notes document, not the submission report. Its tables and claims should remain consistent with `results/EXECUTED_RESULTS.md`.
* `literature/references.md` is a candidate reading list and verification log, not evidence that every cited paper has been read in full. Do not claim a full literature survey.

## Compile

From this directory, compile the report source from `latex/` with `pdflatex main.tex` twice (to resolve references and the table of contents). Check that the logo and all figures resolve from the `latex/` directory. The submitted PDF should be regenerated after the final source edit.
