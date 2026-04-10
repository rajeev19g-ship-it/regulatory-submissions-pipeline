Now let's add the NDA generator:

Stay inside the documents folder
Click "Add file" → "Create new file"
Type in the filename box:

nda_generator.py

Paste this code:

python"""
documents/nda_generator.py
───────────────────────────
LLM-powered NDA/BLA application document generator.

Generates ICH CTD Module 2 documents required for
FDA NDA (New Drug Application) and BLA (Biologics License
Application) submissions including:
    - Quality Overall Summary (m2-3)
    - Nonclinical Written Summary (m2-6)
    - Clinical Summary (m2-7)
    - Summary of Clinical Pharmacology (m2-7-2)
    - Summary of Clinical Efficacy (m2-7-3)
    - Summary of Clinical Safety (m2-7-4)

Regulatory references:
    - 21 CFR Part 314 — NDA regulations
    - 21 CFR Part 601 — BLA regulations
    - ICH M4E(R2) — Efficacy CTD
    - ICH M4Q(R1) — Quality CTD
    - FDA Guidance: M4 CTD Questions and Answers

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai

logger = logging.getLogger(__name__)


# ── NDA data models ───────────────────────────────────────────────────────────

@dataclass
class ClinicalEfficacyData:
    """Clinical efficacy data summary for NDA/BLA."""
    primary_endpoint: str
    primary_endpoint_result: str
    secondary_endpoints: list[str] = field(default_factory=list)
    secondary_results: list[str] = field(default_factory=list)
    key_studies: list[str] = field(default_factory=list)
    patient_population: str = ""
    n_patients_total: int = 0
    n_patients_treated: int = 0
    n_patients_control: int = 0
    study_duration: str = ""
    overall_conclusion: str = ""


@dataclass
class ClinicalSafetyData:
    """Clinical safety data summary for NDA/BLA."""
    total_exposed: int = 0
    exposure_duration: str = ""
    most_common_aes: list[str] = field(default_factory=list)
    serious_aes: list[str] = field(default_factory=list)
    deaths: int = 0
    discontinuations_due_to_ae: int = 0
    discontinuation_rate_pct: float = 0.0
    black_box_warning_proposed: bool = False
    black_box_warning_text: str = ""
    risk_management_strategy: str = ""


@dataclass
class QualityData:
    """Drug substance and drug product quality data summary."""
    drug_substance_name: str = ""
    molecular_formula: str = ""
    molecular_weight: str = ""
    physical_description: str = ""
    synthesis_route_summary: str = ""
    specifications: list[str] = field(default_factory=list)
    stability_summary: str = ""
    container_closure: str = ""


@dataclass
class NDAPackage:
    """Complete NDA/BLA application package data."""
    application_type: str   # NDA or BLA
    drug_name: str
    indication: str
    sponsor: str
    generated_sections: dict[str, str] = field(default_factory=dict)

    def add_section(self, section_id: str, content: str) -> None:
        self.generated_sections[section_id] = content

    def summary(self) -> dict:
        return {
            "application_type":   self.application_type,
            "drug_name":          self.drug_name,
            "indication":         self.indication,
            "sponsor":            self.sponsor,
            "sections_generated": list(self.generated_sections.keys()),
            "total_sections":     len(self.generated_sections),
        }


# ── NDA Generator ─────────────────────────────────────────────────────────────

class NDAGenerator:
    """
    Generates NDA/BLA application document sections using GPT-4o.

    Produces regulatory-quality draft text for CTD Module 2
    sections following FDA 21 CFR Part 314/601 requirements
    and ICH M4 CTD guidance.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.
    model : str
        LLM model. Default 'gpt-4o'.
    application_type : str
        'NDA' or 'BLA'. Default 'NDA'.
    max_tokens : int
        Maximum tokens per section. Default 2500.

    Examples
    --------
    >>> gen = NDAGenerator(application_type="NDA")
    >>> efficacy = ClinicalEfficacyData(
    ...     primary_endpoint="Overall survival",
    ...     primary_endpoint_result="HR=0.72 (95% CI: 0.58-0.89), p=0.002",
    ...     n_patients_total=400,
    ... )
    >>> safety = ClinicalSafetyData(total_exposed=200)
    >>> nda = gen.generate_full_nda("DrugX", "NSCLC", efficacy, safety)
    """

    _SYSTEM_PROMPT = """
You are a senior regulatory affairs director with 20+ years of NDA/BLA
submission experience at major pharmaceutical companies. You have
written CTD Module 2 documents for over 50 successful FDA approvals.

You write regulatory document text following:
- ICH M4E(R2) — Efficacy Clinical Overview and Summary
- ICH M4S(R2) — Safety Nonclinical Overview
- ICH M4Q(R1) — Quality Overall Summary
- FDA 21 CFR Part 314 (NDA) and Part 601 (BLA)
- FDA Guidance documents on CTD format

Writing standards:
- Objective, scientific language throughout
- Past tense for completed studies
- Active voice for regulatory conclusions
- Quantitative data cited precisely
- No promotional language or unsupported claims
- Limitations and risks acknowledged objectively
- DRAFT label required on all outputs

Return only the document section text.
""".strip()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        application_type: str = "NDA",
        max_tokens: int = 2500,
    ) -> None:
        self.model            = model
        self.application_type = application_type.upper()
        self.max_tokens       = max_tokens
        openai.api_key        = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_full_nda(
        self,
        drug_name: str,
        indication: str,
        efficacy: ClinicalEfficacyData,
        safety: ClinicalSafetyData,
        quality: Optional[QualityData] = None,
        sponsor: str = "",
    ) -> NDAPackage:
        """
        Generate all key NDA/BLA CTD Module 2 sections.

        Parameters
        ----------
        drug_name : str
            Drug/product name.
        indication : str
            Proposed indication.
        efficacy : ClinicalEfficacyData
            Clinical efficacy data summary.
        safety : ClinicalSafetyData
            Clinical safety data summary.
        quality : QualityData, optional
            Drug substance/product quality data.
        sponsor : str
            Sponsor company name.

        Returns
        -------
        NDAPackage
            Complete package with all generated CTD sections.
        """
        package = NDAPackage(
            application_type=self.application_type,
            drug_name=drug_name,
            indication=indication,
            sponsor=sponsor,
        )

        sections = [
            ("m2-5",   self._draft_clinical_overview,         (drug_name, indication, efficacy, safety)),
            ("m2-7-2", self._draft_clinical_pharmacology_summary, (drug_name, indication)),
            ("m2-7-3", self._draft_efficacy_summary,          (drug_name, indication, efficacy)),
            ("m2-7-4", self._draft_safety_summary,            (drug_name, indication, safety)),
        ]

        if quality:
            sections.insert(0, ("m2-3", self._draft_quality_summary, (drug_name, quality)))

        for section_id, method, args in sections:
            logger.info("Generating %s section: %s", self.application_type, section_id)
            content = method(*args)
            package.add_section(section_id, content)

        logger.info(
            "%s package generated: %s — %s (%d sections)",
            self.application_type, drug_name, indication,
            len(package.generated_sections),
        )
        return package

    def save_sections(
        self,
        package: NDAPackage,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Save all generated sections as text files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = {}

        for section_id, content in package.generated_sections.items():
            path = output_dir / f"{package.application_type}_{section_id}_{package.drug_name}.txt"
            path.write_text(
                f"DRAFT — FOR REGULATORY REVIEW ONLY\n"
                f"{package.application_type} Application — {package.drug_name}\n"
                f"Section: {section_id} | Indication: {package.indication}\n"
                f"{'='*60}\n\n"
                + content,
                encoding="utf-8",
            )
            saved[section_id] = path
            logger.info("Saved: %s", path)

        return saved

    # ── Section drafters ──────────────────────────────────────────────────────

    def _draft_quality_summary(self, drug_name: str, q: QualityData) -> str:
        """Draft CTD Section 2.3 — Quality Overall Summary."""
        prompt = f"""
Draft CTD Section 2.3 (Quality Overall Summary) for a {self.application_type}.

Drug substance: {drug_name}
Molecular formula: {q.molecular_formula or 'Not provided'}
Molecular weight: {q.molecular_weight or 'Not provided'}
Physical description: {q.physical_description or 'Not provided'}
Synthesis route: {q.synthesis_route_summary or 'Proprietary multi-step synthesis'}
Specifications: {'; '.join(q.specifications) if q.specifications else 'Per ICH Q6A'}
Stability: {q.stability_summary or 'Ongoing stability program per ICH Q1A'}
Container/closure: {q.container_closure or 'Not provided'}

Write the Quality Overall Summary covering:
1. Drug substance — structure, synthesis, characterization, specifications
2. Drug product — composition, manufacture, specifications
3. Stability data summary and proposed shelf life
4. Analytical method validation summary
5. Overall quality risk assessment

Length: 4-5 paragraphs following ICH M4Q(R1) structure.
""".strip()
        return self._call_llm(prompt)

    def _draft_clinical_overview(
        self,
        drug_name: str,
        indication: str,
        efficacy: ClinicalEfficacyData,
        safety: ClinicalSafetyData,
    ) -> str:
        """Draft CTD Section 2.5 — Clinical Overview."""
        prompt = f"""
Draft CTD Section 2.5 (Clinical Overview) for a {self.application_type}.

Drug: {drug_name}
Indication: {indication}
Total patients exposed: {safety.total_exposed}
Primary endpoint: {efficacy.primary_endpoint}
Primary result: {efficacy.primary_endpoint_result}
Key safety findings: {'; '.join(efficacy.secondary_endpoints[:3]) if efficacy.secondary_endpoints else 'See safety summary'}

Write a comprehensive clinical overview covering:
1. Product development rationale and unmet medical need
2. Overview of the clinical pharmacology program
3. Overview of clinical efficacy — key studies and results
4. Overview of clinical safety — exposure, AE profile, risk management
5. Benefit-risk conclusions supporting approval

Length: 5-6 paragraphs following ICH M4E(R2).
Do not use promotional language. Present data objectively.
""".strip()
        return self._call_llm(prompt)

    def _draft_clinical_pharmacology_summary(
        self,
        drug_name: str,
        indication: str,
    ) -> str:
        """Draft CTD Section 2.7.2 — Summary of Clinical Pharmacology."""
        prompt = f"""
Draft CTD Section 2.7.2 (Summary of Clinical Pharmacology Studies)
for a {self.application_type} for {drug_name} in {indication}.

Write a structured clinical pharmacology summary covering:
1. PK overview — absorption, distribution, metabolism, excretion (ADME)
2. Population PK findings — key covariates affecting exposure
3. Exposure-response relationships (PK/PD)
4. Drug interaction studies — CYP inhibition/induction, DDI risk
5. Special populations — renal/hepatic impairment, elderly, pediatric
6. PK conclusions and dosing recommendations

Note any gaps in the clinical pharmacology program.
Length: 4-5 paragraphs.
""".strip()
        return self._call_llm(prompt)

    def _draft_efficacy_summary(
        self,
        drug_name: str,
        indication: str,
        efficacy: ClinicalEfficacyData,
    ) -> str:
        """Draft CTD Section 2.7.3 — Summary of Clinical Efficacy."""
        secondary = "\n".join(
            f"  - {ep}: {res}"
            for ep, res in zip(efficacy.secondary_endpoints, efficacy.secondary_results)
        ) or "  See individual study reports"

        prompt = f"""
Draft CTD Section 2.7.3 (Summary of Clinical Efficacy)
for a {self.application_type}.

Drug: {drug_name}
Indication: {indication}
Patient population: {efficacy.patient_population or indication}
Total patients (ITT): {efficacy.n_patients_total}
Treated: {efficacy.n_patients_treated} | Control: {efficacy.n_patients_control}
Study duration: {efficacy.study_duration or 'As specified in protocols'}

Primary endpoint: {efficacy.primary_endpoint}
Primary result: {efficacy.primary_endpoint_result}

Secondary endpoints:
{secondary}

Key studies: {'; '.join(efficacy.key_studies) if efficacy.key_studies else 'Pivotal Phase 3 study'}
Overall conclusion: {efficacy.overall_conclusion or 'Substantial evidence of effectiveness demonstrated'}

Write a clinical efficacy summary covering:
1. Study design overview of pivotal trial(s)
2. Patient population and baseline characteristics
3. Primary endpoint results with statistical analysis
4. Secondary endpoint results
5. Subgroup analyses and consistency of effect
6. Conclusions on substantial evidence of effectiveness

Length: 5-6 paragraphs. Report statistics exactly as provided.
""".strip()
        return self._call_llm(prompt)

    def _draft_safety_summary(
        self,
        drug_name: str,
        indication: str,
        safety: ClinicalSafetyData,
    ) -> str:
        """Draft CTD Section 2.7.4 — Summary of Clinical Safety."""
        common_aes = "\n".join(f"  - {ae}" for ae in safety.most_common_aes) or "  See integrated safety summary"
        serious_aes = "\n".join(f"  - {ae}" for ae in safety.serious_aes) or "  See integrated safety summary"

        bbw = ""
        if safety.black_box_warning_proposed:
            bbw = f"PROPOSED BLACK BOX WARNING: {safety.black_box_warning_text}"

        prompt = f"""
Draft CTD Section 2.7.4 (Summary of Clinical Safety)
for a {self.application_type}.

Drug: {drug_name}
Indication: {indication}
Total exposed: {safety.total_exposed}
Exposure duration: {safety.exposure_duration or 'Variable per protocol'}
Deaths: {safety.deaths}
Discontinuations due to AE: {safety.discontinuations_due_to_ae} ({safety.discontinuation_rate_pct:.1f}%)

Most common adverse events (>=10%):
{common_aes}

Serious adverse events:
{serious_aes}

{bbw}
Risk management: {safety.risk_management_strategy or 'Standard pharmacovigilance plan'}

Write a clinical safety summary covering:
1. Overall safety exposure (N, duration, demographics)
2. Common adverse events — incidence, severity, management
3. Serious adverse events and deaths
4. Discontinuations and dose modifications due to AEs
5. Special safety topics (if applicable: hepatotoxicity, QTc, immunogenicity)
6. Risk management strategy and proposed labeling
7. Overall safety conclusions and benefit-risk context

Length: 6-7 paragraphs. Present risks objectively and completely.
{"Include prominent discussion of the proposed black box warning." if safety.black_box_warning_proposed else ""}
""".strip()
        return self._call_llm(prompt)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to LLM and return text response."""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise
