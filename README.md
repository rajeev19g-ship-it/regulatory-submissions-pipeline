# Regulatory Submissions Pipeline

An AI-powered regulatory document automation platform covering the full submission lifecycle — from IND through NDA/BLA — with LLM-powered document generation, medical translation, precision medicine biomarker analysis, and drug discovery screening.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FDA](https://img.shields.io/badge/Standard-FDA%20eCTD%203.2.2-red)](https://www.fda.gov/)
[![EMA](https://img.shields.io/badge/Standard-EMA%20eCTD-orange)](https://www.ema.europa.eu/)

---

## Pipeline Overview

| Module | Files | Description |
|--------|-------|-------------|
| `src/ectd/` | `submission.py`, `validator.py` | eCTD package builder, FDA/EMA validation |
| `src/documents/` | `ind_generator.py`, `nda_generator.py` | IND/NDA/BLA document auto-generation |
| `src/translation/` | `medical_translator.py`, `term_validator.py` | LLM medical translation + terminology validation |
| `src/precision/` | `biomarker.py`, `patient_stratification.py` | Biomarker analysis, ML patient stratification |
| `src/drug_discovery/` | `target_predictor.py`, `molecule_screener.py` | ML target prediction, virtual screening |

---

## Key Features

- **eCTD Builder** — Automated eCTD 3.2.2 submission package construction with FDA/EMA conformance validation
- **IND/NDA/BLA Generator** — LLM-powered generation of Module 2 summaries, clinical overviews, and nonclinical written summaries
- **Medical Translation** — GPT-4o powered translation pipeline with MedDRA/SNOMED terminology validation across 12 languages
- **Precision Medicine** — Biomarker-driven patient stratification using ML clustering and predictive modeling
- **Drug Discovery** — ML-based target binding prediction and virtual compound screening

---

### Repository Structure

```
regulatory-submissions-pipeline/
├── .github/workflows/ci.yml        # GitHub Actions CI pipeline
├── src/
│   ├── ectd/
│   │   ├── submission.py           # eCTD package builder
│   │   └── validator.py            # FDA/EMA conformance validator
│   ├── documents/
│   │   ├── ind_generator.py        # IND CTD Module 2 generator
│   │   └── nda_generator.py        # NDA/BLA CTD Module 2 generator
│   ├── translation/
│   │   ├── medical_translator.py   # 12-language medical translator
│   │   └── term_validator.py       # MedDRA/SNOMED CT validator
│   ├── precision/
│   │   ├── biomarker.py            # Predictive biomarker analysis
│   │   └── patient_stratification.py  # ML patient clustering
│   └── drug_discovery/
│       ├── target_predictor.py     # QSAR binding affinity model
│       └── molecule_screener.py    # Virtual screening pipeline
├── tests/
│   └── test_regulatory.py
├── notebooks/
├── data/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```


## Architecture

```
Raw Clinical / Nonclinical Data
         ↓
eCTD Builder           —  CTD folder structure · backbone XML · document registry
         ↓
IND / NDA / BLA Docs   —  CTD Module 2 sections · ICH M4E(R2) · GPT-4o drafting
         ↓
Medical Translation    —  12 languages · back-translation QC · MedDRA validation
         ↓
Precision Medicine     —  Biomarker analysis · patient stratification · CDx
         ↓
Drug Discovery         —  QSAR prediction · ADMET · virtual screening
         ↓
FDA / EMA eCTD Submission Package
```

## Tech Stack

| Category | Libraries |
|----------|-----------|
| LLM/NLP | openai, langchain, transformers, spaCy |
| ML/DL | scikit-learn, TensorFlow, XGBoost |
| Bioinformatics | RDKit, BioPython, DeepChem |
| Document Generation | python-docx, reportlab, Jinja2 |
| Regulatory | lxml, xmlschema, PyPDF2 |
| Testing | pytest, pytest-cov |

---

## Regulatory Standards

- FDA eCTD Technical Specifications v3.2.2
- EMA eCTD Implementation Guide
- ICH M4 — Common Technical Document (CTD)
- ICH E3 — Clinical Study Reports
- ICH Q2(R1) — Validation of Analytical Procedures
- MedDRA 26.0 Terminology
- SNOMED CT

---

## Author

**Girish Rajeev**
Clinical Data Scientist | Data Analyst | Regulatory Standards Leader | AI/ML Solution Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/girish-rajeev-756808138/)
