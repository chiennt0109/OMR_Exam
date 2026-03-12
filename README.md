# OMR_Exam

Desktop Optical Mark Recognition (OMR) exam grading system built with **Python + PySide6 + OpenCV + JSON**.

## Features

### 1) Paper Template Editor
- Load blank exam paper image.
- Define anchors interactively (unlimited), recognition zones, and bubble grids.
- Zone types: `ANCHOR`, `STUDENT_ID`, `EXAM_CODE`, `MCQ_BLOCK`, `TRUE_FALSE_GROUP`, `NUMERIC_GRID`.
- Zoom/pan canvas.
- Draw rectangle zones with click-drag, then move/resize/delete them.
- Grid wizard auto-generation for MCQ/True-False/Numeric blocks (no manual bubble placement).
- Duplicate/snap support in template engine.
- Save template as JSON.

### 2) Exam Session Management
- Manage exam name, date, subjects, student list, template path, answer-key path.
- Import students from CSV or Excel.

### 3) Answer Key Manager
- Multiple subjects and exam codes.
- Section-based scoring (e.g., Q1–20: 0.25, Q21–40: 0.5).

### 4) OMR Scan Recognition Pipeline
1. detect exam sheet
2. perspective correction
3. detect anchors/template alignment
4. extract zones
5. detect bubbles
6. generate answers

Bubble detection uses:
- grayscale
- Gaussian blur
- adaptive threshold
- filled-pixel ratio

### 5) Error Correction Interface
- Error list for: missing anchors/template mismatch, unreadable ID, multiple answers, blanks.
- Correction tab supports preview placeholder + manual edit field.

### 6) Scoring Engine
- Compares detected answers vs. answer key.
- Outputs: correct, wrong, blank, score.

### 7) Export Results
Exports to:
- CSV
- JSON
- XML
- Excel (`.xlsx`)

Excel columns include:
`StudentID | Name | Subject | Score | Correct`

---

## Project Structure

```text
OMR_Exam/
    core/
        omr_engine.py
        template_engine.py
        scoring_engine.py

    editor/
        template_editor.py

    models/
        template.py
        exam_session.py
        answer_key.py

    gui/
        main_window.py

    data/
```

## Run

```bash
python main.py
```

## Suggested dependencies

```bash
pip install PySide6 opencv-python numpy pandas openpyxl
```

## Batch processing and progress

Use **Batch Scan Images** in the GUI to process large sets of sheets; a progress bar and list updates are shown per file.
