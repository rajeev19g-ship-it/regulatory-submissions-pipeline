Now let's build the documents module:

Click src → documents folder
Click "Add file" → "Create new file"
Type in the filename box:

ind_generator.py

Paste this code:

python"""
documents/ind_generator.py
───────────────────────────
LLM-powered IND (Investigational New Drug) application
document generator.

Generates ICH CTD Module 1 and Module 2 documents required
for FDA IND submissions including:
    - Form FDA 1571 cover sheet data
    - Investigator Brochure (IB) narrative sections
    - Clinical Protocol synopsis
    - Pharmacology/Toxicology summary (m2-4)
    - Clinical overview (m2-5)
    - Introductory summary (m2-2)

Regulatory references:
    - 21 CFR Part 312 — IND regulations
    - FDA Guidance: Content and Format of INDs
    - ICH M4E(R2) — Clinical Overview

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai
from jinja2 import Environment, BaseLoader

logger = logging.getLogger(__name__)


# ── IND data models ───────────────────────────────────────────────────────────

@dataclass
class DrugProfile:
    """Core drug/compound profile for IND generation."""
    drug_name: str
    inn_name: str
    drug_class: str
    mechanism_of_action: str
    indication: str
    target_population: str
    dosage_form: str
    route_of_administration: str
    proposed_dose: str
    dose_frequency: str
    treatment_duration: str
    phase: str = "Phase 1"
    sponsor_name: str = ""
    sponsor_address: str = ""
    responsible_investigator: str = ""


@dataclass
class NonclinicalProfile:
    """Nonclinical pharmacology and toxicology data summary."""
    primary_pharmacology: str = ""
    secondary_pharmacology: str = ""
    safety_pharmacology: str = ""
    pk_absorption: str = ""
    pk_distribution: str = ""
    pk_metabolism: str = ""
    pk_excretion: str = ""
    acute_toxicity: str = ""
    repeat_dose_toxicity: str = ""
    genotoxicity: str = ""
    carcinogenicity: str = ""
    reproductive_toxicity: str = ""
    noael: str = ""
    first_in_human_dose_rationale: str = ""


@dataclass
class INDPackage:
    """Complete IND application package data."""
    drug: DrugProfile
    nonclinical: NonclinicalProfile
    protocol_synopsis: str = ""
    generated_sections: dict[str, str] = field(default_factory=dict)

    def add_section(self, section_id: str, content: str) -> None:
        self.generated_sections[section_id] = content

    def to_dict(self) -> dict:
        return {
            "drug_name":           self.drug.drug_name,
            "indication":          self.drug.indication,
            "phase":               self.drug.phase,
            "sections_generated":  list(self.generated_sections.keys()),
        }


# ── IND Generator ─────────────────────────────────────────────────────────────

class INDGenerator:
    """
    Generates IND application document sections using GPT-4o.

    Produces regulatory-quality draft text for key IND sections
    following FDA 21 CFR Part 312 and ICH M4 CTD structure.
    All outputs are clearly marked as drafts requiring
    medical/regulatory review before submission.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env variable.
    model : str
        LLM model. Default 'gpt-4o'.
    max_tokens : int
        Maximum tokens per section. Default 2000.

    Examples
    --------
    >>> drug = DrugProfile(
    ...     drug_name="DrugX",
    ...     inn_name="drugxumab",
    ...     drug_class="Anti-PD-1 monoclonal antibody",
    ...     mechanism_of_action="Blocks PD-1/PD-L1 interaction",
    ...     indication="NSCLC",
    ...     target_population="Adults with stage IIIB/IV NSCLC",
    ...     dosage_form="IV infusion",
    ...     route_of_administration="Intravenous",
    ...     proposed_dose="200mg",
    ...     dose_frequency="Q3W",
    ...     treatment_duration="Until progression or unacceptable toxicity",
    ... )
    >>> gen = INDGenerator()
    >>> ind = gen.generate_full_ind(drug, nonclinical)
    """

    _SYSTEM_PROMPT = """
You are an expert regulatory affairs specialist with 20+ years of experience
writing IND applications for FDA submission under 21 CFR Part 312.

You write regulatory document text following:
- FDA Guidance: Content and Format of Investigational New Drug Applications
- ICH M4E(R2) Clinical Overview guidance
- ICH M4S(R2) Nonclinical Overview guidance

Writing conventions:
- Scientific, precise language in present or past tense as appropriate
- Passive voice for methods, active for conclusions
- No marketing language or unsupported efficacy claims
- Clearly flag any data gaps or limitations
- Reference regulatory guidances where appropriate
- All outputs are DRAFT for medical/regulatory review

Return only the document text — no preamble or meta-commentary.
""".strip()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
    ) -> None:
        self.model      = model
        self.max_tokens = max_tokens
        openai.api_key  = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_full_ind(
        self,
        drug: DrugProfile,
        nonclinical: NonclinicalProfile,
        protocol_synopsis: str = "",
    ) -> INDPackage:
        """
        Generate all key IND document sections.

        Parameters
        ----------
        drug : DrugProfile
            Drug/compound profile.
        nonclinical : NonclinicalProfile
            Nonclinical pharmacology and toxicology data.
        protocol_synopsis : str, optional
            Brief protocol synopsis for clinical section.

        Returns
        -------
        INDPackage
            Complete IND package with all generated sections.
        """
        package = INDPackage(
            drug=drug,
            nonclinical=nonclinical,
            protocol_synopsis=protocol_synopsis,
        )

        sections = [
            ("m2-2",   self._draft_introduction,      (drug,)),
            ("m2-4",   self._draft_nonclinical_overview, (drug, nonclinical)),
            ("m2-5",   self._draft_clinical_overview,  (drug, protocol_synopsis)),
            ("ib-summary", self._draft_ib_summary,     (drug, nonclinical)),
        ]

        for section_id, method, args in sections:
            logger.info("Generating IND section: %s", section_id)
            content = method(*args)
            package.add_section(section_id, content)

        logger.info(
            "IND package generated: %s (%d sections)",
            drug.drug_name, len(package.generated_sections),
        )
        return package

    def save_sections(
        self,
        package: INDPackage,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """
        Save all generated IND sections as text files.

        Parameters
        ----------
        package : INDPackage
            Generated IND package.
        output_dir : str or Path
            Directory to save section files.

        Returns
        -------
        dict[str, Path]
            Mapping of section ID to saved file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = {}

        for section_id, content in package.generated_sections.items():
            path = output_dir / f"IND_{section_id}_{package.drug.drug_name}.txt"
            path.write_text(
                f"DRAFT — FOR REGULATORY REVIEW ONLY\n"
                f"Section: {section_id}\n"
                f"Drug: {package.drug.drug_name}\n"
                f"{'='*60}\n\n"
                + content,
                encoding="utf-8",
            )
            saved[section_id] = path
            logger.info("Saved: %s", path)

        return saved

    # ── Section drafters ──────────────────────────────────────────────────────

    def _draft_introduction(self, drug: DrugProfile) -> str:
        """Draft CTD Section 2.2 — Introduction."""
        prompt = f"""
Draft CTD Section 2.2 (Introduction) for an IND application.

Drug profile:
- Drug name: {drug.drug_name} ({drug.inn_name})
- Drug class: {drug.drug_class}
- Mechanism of action: {drug.mechanism_of_action}
- Proposed indication: {drug.indication}
- Target population: {drug.target_population}
- Development phase: {drug.phase}
- Dosage form: {drug.dosage_form}
- Route: {drug.route_of_administration}
- Proposed dose: {drug.proposed_dose} {drug.dose_frequency}

Write a 2-3 paragraph introduction covering:
1. Brief description of the drug and its pharmacological class
2. Proposed indication and unmet medical need
3. Overview of the development program and this IND submission
""".strip()
        return self._call_llm(prompt)

    def _draft_nonclinical_overview(
        self,
        drug: DrugProfile,
        nc: NonclinicalProfile,
    ) -> str:
        """Draft CTD Section 2.4 — Nonclinical Overview."""
        prompt = f"""
Draft CTD Section 2.4 (Nonclinical Overview) for an IND application.

Drug: {drug.drug_name} — {drug.indication}

Nonclinical data summary:
Primary pharmacology: {nc.primary_pharmacology or 'Not yet available'}
Safety pharmacology: {nc.safety_pharmacology or 'Not yet available'}
PK — Absorption: {nc.pk_absorption or 'Not yet available'}
PK — Distribution: {nc.pk_distribution or 'Not yet available'}
PK — Metabolism: {nc.pk_metabolism or 'Not yet available'}
PK — Excretion: {nc.pk_excretion or 'Not yet available'}
Acute toxicity: {nc.acute_toxicity or 'Not yet available'}
Repeat-dose toxicity: {nc.repeat_dose_toxicity or 'Not yet available'}
Genotoxicity: {nc.genotoxicity or 'Not yet available'}
NOAEL: {nc.noael or 'Not yet determined'}
First-in-human dose rationale: {nc.first_in_human_dose_rationale or 'To be determined'}

Write a structured nonclinical overview covering:
1. Pharmacology (primary and safety)
2. Pharmacokinetics summary
3. Toxicology findings and NOAEL
4. Overall nonclinical risk assessment
5. Adequacy of nonclinical program to support Phase 1

Flag any data gaps clearly.
Length: 4-5 paragraphs.
""".strip()
        return self._call_llm(prompt)

    def _draft_clinical_overview(
        self,
        drug: DrugProfile,
        protocol_synopsis: str,
    ) -> str:
        """Draft CTD Section 2.5 — Clinical Overview."""
        prompt = f"""
Draft CTD Section 2.5 (Clinical Overview) for an IND application.

Drug: {drug.drug_name}
Indication: {drug.indication}
Phase: {drug.phase}
Target population: {drug.target_population}
Proposed dose: {drug.proposed_dose} {drug.dose_frequency}
Treatment duration: {drug.treatment_duration}

Protocol synopsis:
{protocol_synopsis or 'Phase 1 first-in-human dose escalation study'}

Write a clinical overview covering:
1. Product development rationale
2. Overview of clinical pharmacology strategy
3. Overview of the proposed clinical program
4. Benefit-risk assessment for the proposed Phase 1 study
5. Conclusion on adequacy of the program

Length: 4-5 paragraphs. Note any areas where additional data
will be generated during clinical development.
""".strip()
        return self._call_llm(prompt)

    def _draft_ib_summary(
        self,
        drug: DrugProfile,
        nc: NonclinicalProfile,
    ) -> str:
        """Draft Investigator Brochure summary section."""
        prompt = f"""
Draft the Summary section of an Investigator Brochure (IB)
for a Phase 1 clinical trial.

Drug: {drug.drug_name} ({drug.inn_name})
Class: {drug.drug_class}
MOA: {drug.mechanism_of_action}
Indication: {drug.indication}
Dose: {drug.proposed_dose} {drug.dose_frequency} {drug.route_of_administration}

Nonclinical highlights:
- Primary pharmacology: {nc.primary_pharmacology or 'See nonclinical section'}
- Key toxicology finding: {nc.repeat_dose_toxicity or 'See toxicology section'}
- NOAEL: {nc.noael or 'To be determined'}

Write the IB Summary section covering:
1. Chemical/pharmaceutical description
2. Nonclinical studies summary
3. Effects in humans (none for first-in-human — state this clearly)
4. Summary of risks and precautions for investigators

Length: 3-4 paragraphs.
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
            text = response.choices[0].message.content.strip()
            logger.debug("LLM response: %d words", len(text.split()))
            return text
        except openai.OpenAIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise
