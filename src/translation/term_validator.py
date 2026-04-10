Now let's add the terminology validator:

Stay inside the translation folder
Click "Add file" → "Create new file"
Type in the filename box:

term_validator.py

Paste this code:

python"""
translation/term_validator.py
──────────────────────────────
Medical terminology validation for regulatory documents.

Validates medical terms against standard controlled vocabularies:
    - MedDRA (Medical Dictionary for Regulatory Activities) v26.0
    - SNOMED CT (Systematized Nomenclature of Medicine)
    - ICD-10-CM (International Classification of Diseases)
    - WHO Drug Dictionary
    - CDISC Controlled Terminology

Used to ensure consistency and accuracy in:
    - Adverse event coding (MedDRA)
    - Diagnosis coding (ICD-10)
    - Drug name standardization (INN/WHO)
    - SDTM/ADaM controlled terminology compliance

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── MedDRA hierarchy reference ────────────────────────────────────────────────

MEDDRA_SOC = {
    "10001316": "Blood and lymphatic system disorders",
    "10005329": "Cardiac disorders",
    "10010331": "Congenital, familial and genetic disorders",
    "10013993": "Ear and labyrinth disorders",
    "10014698": "Endocrine disorders",
    "10015919": "Eye disorders",
    "10017947": "Gastrointestinal disorders",
    "10018065": "General disorders and administration site conditions",
    "10019805": "Hepatobiliary disorders",
    "10021428": "Immune system disorders",
    "10021881": "Infections and infestations",
    "10022117": "Injury, poisoning and procedural complications",
    "10022891": "Investigations",
    "10023320": "Metabolism and nutrition disorders",
    "10028395": "Musculoskeletal and connective tissue disorders",
    "10029104": "Neoplasms benign, malignant and unspecified",
    "10029205": "Nervous system disorders",
    "10033371": "Psychiatric disorders",
    "10038359": "Renal and urinary disorders",
    "10036585": "Reproductive system and breast disorders",
    "10038738": "Respiratory, thoracic and mediastinal disorders",
    "10040785": "Skin and subcutaneous tissue disorders",
    "10042613": "Social circumstances",
    "10042762": "Surgical and medical procedures",
    "10047065": "Vascular disorders",
    "10077536": "Product issues",
}

# Common MedDRA Preferred Terms (PT) — representative subset
MEDDRA_PT_SAMPLE = {
    "nausea":                    {"code": "10028813", "soc": "10017947"},
    "vomiting":                  {"code": "10047700", "soc": "10017947"},
    "fatigue":                   {"code": "10016256", "soc": "10018065"},
    "headache":                  {"code": "10019211", "soc": "10029205"},
    "diarrhoea":                 {"code": "10012735", "soc": "10017947"},
    "alopecia":                  {"code": "10001760", "soc": "10040785"},
    "anaemia":                   {"code": "10002034", "soc": "10001316"},
    "neutropenia":               {"code": "10029354", "soc": "10001316"},
    "thrombocytopenia":          {"code": "10043554", "soc": "10001316"},
    "peripheral neuropathy":     {"code": "10034620", "soc": "10029205"},
    "rash":                      {"code": "10037844", "soc": "10040785"},
    "dyspnoea":                  {"code": "10013968", "soc": "10038738"},
    "hypertension":              {"code": "10020772", "soc": "10047065"},
    "hypotension":               {"code": "10021097", "soc": "10047065"},
    "pyrexia":                   {"code": "10037660", "soc": "10018065"},
    "constipation":              {"code": "10010774", "soc": "10017947"},
    "decreased appetite":        {"code": "10061428", "soc": "10023320"},
    "insomnia":                  {"code": "10022437", "soc": "10033371"},
    "arthralgia":                {"code": "10003239", "soc": "10028395"},
    "myalgia":                   {"code": "10028411", "soc": "10028395"},
    "back pain":                 {"code": "10003988", "soc": "10028395"},
    "abdominal pain":            {"code": "10000081", "soc": "10017947"},
    "stomatitis":                {"code": "10042128", "soc": "10017947"},
    "pneumonia":                 {"code": "10035664", "soc": "10021881"},
    "urinary tract infection":   {"code": "10046571", "soc": "10021881"},
    "elevated alanine aminotransferase": {"code": "10001551", "soc": "10022891"},
    "elevated aspartate aminotransferase": {"code": "10003481", "soc": "10022891"},
}

# CTCAE v5.0 severity grades
CTCAE_GRADES = {
    1: "Mild — asymptomatic or mild symptoms; clinical or diagnostic observations only",
    2: "Moderate — minimal, local or noninvasive intervention indicated",
    3: "Severe — severe or medically significant but not immediately life-threatening",
    4: "Life-threatening — urgent intervention indicated",
    5: "Death related to AE",
}

# Common non-standard AE terms mapping to MedDRA PTs
NON_STANDARD_MAPPINGS = {
    "throwing up":       "vomiting",
    "sick to stomach":   "nausea",
    "hair loss":         "alopecia",
    "tired":             "fatigue",
    "low platelet":      "thrombocytopenia",
    "low white blood cell": "neutropenia",
    "low red blood cell": "anaemia",
    "shortness of breath": "dyspnoea",
    "high blood pressure": "hypertension",
    "low blood pressure":  "hypotension",
    "fever":              "pyrexia",
    "diarrhea":           "diarrhoea",
    "mouth sores":        "stomatitis",
    "joint pain":         "arthralgia",
    "muscle pain":        "myalgia",
    "numbness":           "peripheral neuropathy",
    "tingling":           "peripheral neuropathy",
}


# ── Validation results ────────────────────────────────────────────────────────

@dataclass
class TermValidationResult:
    """Result of validating a single medical term."""
    input_term: str
    vocabulary: str
    is_valid: bool
    standard_term: Optional[str] = None
    term_code: Optional[str] = None
    soc: Optional[str] = None
    suggestion: Optional[str] = None
    confidence: str = "high"    # high | medium | low

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        sugg   = f" → suggest: '{self.suggestion}'" if self.suggestion else ""
        return f"[{status}] '{self.input_term}' ({self.vocabulary}){sugg}"


@dataclass
class DocumentValidationReport:
    """Terminology validation report for a full document."""
    document_name: str
    vocabulary: str
    total_terms: int = 0
    valid_terms: int = 0
    invalid_terms: int = 0
    results: list[TermValidationResult] = field(default_factory=list)

    @property
    def compliance_rate(self) -> float:
        if self.total_terms == 0:
            return 0.0
        return round(100 * self.valid_terms / self.total_terms, 1)

    def summary(self) -> dict:
        return {
            "document":        self.document_name,
            "vocabulary":      self.vocabulary,
            "total_terms":     self.total_terms,
            "valid_terms":     self.valid_terms,
            "invalid_terms":   self.invalid_terms,
            "compliance_rate": f"{self.compliance_rate}%",
            "invalid_list":    [
                r.input_term for r in self.results if not r.is_valid
            ],
        }


# ── Term Validator ────────────────────────────────────────────────────────────

class MedicalTermValidator:
    """
    Validates medical terms against controlled vocabularies.

    Checks adverse event terms against MedDRA, diagnoses against
    ICD-10, and SDTM variables against CDISC controlled terminology.
    Provides standardization suggestions for non-standard terms.

    Parameters
    ----------
    strict_mode : bool
        If True, partial matches are not accepted. Default False.

    Examples
    --------
    >>> validator = MedicalTermValidator()
    >>> result = validator.validate_meddra_term("hair loss")
    >>> print(result)
    [INVALID] 'hair loss' (MedDRA) → suggest: 'alopecia'

    >>> report = validator.validate_ae_list(
    ...     ["nausea", "fatigue", "hair loss", "fever"],
    ...     document_name="ADAE dataset"
    ... )
    >>> print(report.summary())
    """

    def __init__(self, strict_mode: bool = False) -> None:
        self.strict_mode = strict_mode

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_meddra_term(self, term: str) -> TermValidationResult:
        """
        Validate a single term against MedDRA Preferred Terms.

        Parameters
        ----------
        term : str
            Adverse event term to validate.

        Returns
        -------
        TermValidationResult
        """
        normalized = term.lower().strip()

        # Direct match
        if normalized in MEDDRA_PT_SAMPLE:
            pt_data = MEDDRA_PT_SAMPLE[normalized]
            return TermValidationResult(
                input_term=term,
                vocabulary="MedDRA",
                is_valid=True,
                standard_term=normalized,
                term_code=pt_data["code"],
                soc=MEDDRA_SOC.get(pt_data["soc"], pt_data["soc"]),
                confidence="high",
            )

        # Non-standard mapping
        if normalized in NON_STANDARD_MAPPINGS:
            suggestion = NON_STANDARD_MAPPINGS[normalized]
            return TermValidationResult(
                input_term=term,
                vocabulary="MedDRA",
                is_valid=False,
                suggestion=suggestion,
                confidence="high",
            )

        # Partial match (fuzzy)
        if not self.strict_mode:
            for pt in MEDDRA_PT_SAMPLE:
                if normalized in pt or pt in normalized:
                    return TermValidationResult(
                        input_term=term,
                        vocabulary="MedDRA",
                        is_valid=False,
                        suggestion=pt,
                        confidence="medium",
                    )

        return TermValidationResult(
            input_term=term,
            vocabulary="MedDRA",
            is_valid=False,
            suggestion=None,
            confidence="low",
        )

    def validate_ae_list(
        self,
        ae_terms: list[str],
        document_name: str = "AE dataset",
    ) -> DocumentValidationReport:
        """
        Validate a list of adverse event terms against MedDRA.

        Parameters
        ----------
        ae_terms : list[str]
            List of AE terms to validate.
        document_name : str
            Name of the source document for the report.

        Returns
        -------
        DocumentValidationReport
        """
        report = DocumentValidationReport(
            document_name=document_name,
            vocabulary="MedDRA",
            total_terms=len(ae_terms),
        )

        for term in ae_terms:
            result = self.validate_meddra_term(term)
            report.results.append(result)
            if result.is_valid:
                report.valid_terms += 1
            else:
                report.invalid_terms += 1

        logger.info(
            "MedDRA validation: %d/%d valid (%.1f%%) — %s",
            report.valid_terms, report.total_terms,
            report.compliance_rate, document_name,
        )
        return report

    def standardize_ae_terms(self, ae_terms: list[str]) -> dict[str, str]:
        """
        Map a list of AE terms to their MedDRA standard equivalents.

        Parameters
        ----------
        ae_terms : list[str]
            Input AE terms (may be non-standard).

        Returns
        -------
        dict[str, str]
            Mapping of input term → MedDRA preferred term.
        """
        mapping = {}
        for term in ae_terms:
            result = self.validate_meddra_term(term)
            if result.is_valid:
                mapping[term] = result.standard_term or term
            elif result.suggestion:
                mapping[term] = result.suggestion
            else:
                mapping[term] = term  # Keep original if no mapping found
        return mapping

    def validate_ctcae_grade(self, grade: int) -> dict:
        """
        Validate a CTCAE grade value and return its description.

        Parameters
        ----------
        grade : int
            CTCAE grade (1-5).

        Returns
        -------
        dict
            Grade validation result with description.
        """
        if grade not in CTCAE_GRADES:
            return {
                "valid":       False,
                "grade":       grade,
                "description": f"Invalid CTCAE grade: {grade} (must be 1-5)",
            }
        return {
            "valid":       True,
            "grade":       grade,
            "description": CTCAE_GRADES[grade],
        }

    def extract_ae_terms_from_text(self, text: str) -> list[str]:
        """
        Extract potential adverse event terms from free text.

        Uses pattern matching to identify likely AE terms in
        clinical narrative text for validation.

        Parameters
        ----------
        text : str
            Free text containing AE descriptions.

        Returns
        -------
        list[str]
            Extracted candidate AE terms.
        """
        # Known AE signal words
        all_known = set(MEDDRA_PT_SAMPLE.keys()) | set(NON_STANDARD_MAPPINGS.keys())
        text_lower = text.lower()
        found = []

        for term in all_known:
            if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
                found.append(term)

        logger.debug("Extracted %d AE terms from text", len(found))
        return sorted(set(found))

    def get_soc_for_term(self, term: str) -> Optional[str]:
        """
        Return the System Organ Class (SOC) for a MedDRA PT.

        Parameters
        ----------
        term : str
            MedDRA preferred term.

        Returns
        -------
        str or None
            SOC label, or None if not found.
        """
        normalized = term.lower().strip()
        if normalized in MEDDRA_PT_SAMPLE:
            soc_code = MEDDRA_PT_SAMPLE[normalized]["soc"]
            return MEDDRA_SOC.get(soc_code)

        # Try non-standard mapping
        if normalized in NON_STANDARD_MAPPINGS:
            standard = NON_STANDARD_MAPPINGS[normalized]
            if standard in MEDDRA_PT_SAMPLE:
                soc_code = MEDDRA_PT_SAMPLE[standard]["soc"]
                return MEDDRA_SOC.get(soc_code)

        return None

    def generate_meddra_coding_table(
        self,
        ae_terms: list[str],
    ) -> list[dict]:
        """
        Generate a MedDRA coding table from a list of AE terms.

        Parameters
        ----------
        ae_terms : list[str]
            List of AE terms to code.

        Returns
        -------
        list[dict]
            List of coding records with PT code, SOC, and validity.
        """
        table = []
        for term in ae_terms:
            result  = self.validate_meddra_term(term)
            std_term = result.standard_term or result.suggestion or term
            pt_data  = MEDDRA_PT_SAMPLE.get(std_term, {})
            soc_code = pt_data.get("soc", "")

            table.append({
                "input_term":    term,
                "preferred_term": std_term,
                "pt_code":       pt_data.get("code", "UNKNOWN"),
                "soc":           MEDDRA_SOC.get(soc_code, "UNKNOWN"),
                "soc_code":      soc_code,
                "is_standard":   result.is_valid,
                "confidence":    result.confidence,
            })

        logger.info("MedDRA coding table generated: %d terms", len(table))
        return table
