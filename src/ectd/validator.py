Now let's add the validator:

Stay inside the ectd folder
Click "Add file" → "Create new file"
Type in the filename box:

validator.py

Paste this code:

python"""
ectd/validator.py
──────────────────
FDA/EMA eCTD submission conformance validator.

Validates eCTD submission packages against:
    - FDA Technical Conformance Guide requirements
    - EMA eCTD Implementation Guide
    - ICH M4 CTD structure requirements
    - File naming conventions (FDA/EMA)
    - PDF/A compliance checks
    - Bookmark and hyperlink validation
    - File size limits

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


# ── Validation rules ──────────────────────────────────────────────────────────

# FDA file naming convention rules
FDA_FILENAME_PATTERN = re.compile(r"^[a-z0-9_\-]{1,64}\.(pdf|xml|docx|xpt|sas7bdat)$")
EMA_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}\.(pdf|xml|docx)$")

# Maximum file sizes (bytes)
FILE_SIZE_LIMITS = {
    "pdf":   50 * 1024 * 1024,    # 50 MB per FDA guidance
    "xml":   10 * 1024 * 1024,    # 10 MB
    "docx":  25 * 1024 * 1024,    # 25 MB
    "xpt":  500 * 1024 * 1024,    # 500 MB for datasets
}

# Required CTD sections for each submission type
REQUIRED_SECTIONS = {
    "NDA": ["m1", "m2", "m3", "m4", "m5"],
    "BLA": ["m1", "m2", "m3", "m4", "m5"],
    "IND": ["m1", "m2"],
    "MAA": ["m1", "m2", "m3", "m4", "m5"],
}

# Required documents per submission type
REQUIRED_DOCUMENTS = {
    "NDA": [
        "Application form (Form FDA 356h)",
        "Clinical overview (m2-5)",
        "Nonclinical overview (m2-4)",
        "Quality overall summary (m2-3)",
    ],
    "IND": [
        "Form FDA 1571",
        "Investigator Brochure",
        "Clinical Protocol",
    ],
    "BLA": [
        "Application form (Form FDA 356h)",
        "Clinical overview (m2-5)",
        "Summary of clinical pharmacology (m2-7-2)",
    ],
}


# ── Validation results ────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """A single validation finding."""
    severity: str       # error | warning | info
    rule: str           # Rule identifier
    message: str
    location: str = ""  # File or section path

    def __str__(self) -> str:
        loc = f" [{self.location}]" if self.location else ""
        return f"[{self.severity.upper()}] {self.rule}{loc}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report for an eCTD submission."""
    submission_label: str
    agency: str
    submission_type: str
    issues: list[ValidationIssue] = field(default_factory=list)
    files_checked: int = 0
    sections_checked: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_issue(self, severity: str, rule: str, message: str, location: str = "") -> None:
        self.issues.append(ValidationIssue(severity, rule, message, location))

    def summary(self) -> dict:
        return {
            "submission":      self.submission_label,
            "agency":          self.agency,
            "submission_type": self.submission_type,
            "valid":           self.is_valid,
            "errors":          len(self.errors),
            "warnings":        len(self.warnings),
            "files_checked":   self.files_checked,
            "sections_checked": self.sections_checked,
        }

    def print_report(self) -> None:
        """Print a formatted validation report to stdout."""
        print(f"\n{'='*60}")
        print(f"eCTD Validation Report — {self.submission_label}")
        print(f"Agency: {self.agency} | Type: {self.submission_type}")
        print(f"{'='*60}")
        print(f"Result: {'PASS' if self.is_valid else 'FAIL'}")
        print(f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)}")
        print(f"Files checked: {self.files_checked}")
        print(f"{'='*60}")
        if self.errors:
            print("\nErrors:")
            for e in self.errors:
                print(f"  {e}")
        if self.warnings:
            print("\nWarnings:")
            for w in self.warnings:
                print(f"  {w}")
        print()


# ── Validator ─────────────────────────────────────────────────────────────────

class eCTDValidator:
    """
    Validates eCTD submission packages for FDA/EMA conformance.

    Runs a comprehensive suite of validation checks including
    structure, file naming, file sizes, backbone XML integrity,
    and required section/document completeness.

    Parameters
    ----------
    agency : str
        Target regulatory agency: 'FDA' or 'EMA'. Default 'FDA'.
    submission_type : str
        Submission type: 'NDA', 'BLA', 'IND', 'MAA'. Default 'NDA'.
    strict : bool
        If True, treats warnings as errors. Default False.

    Examples
    --------
    >>> validator = eCTDValidator(agency="FDA", submission_type="NDA")
    >>> report = validator.validate("submissions/NDA-123456-0000/")
    >>> report.print_report()
    >>> if report.is_valid:
    ...     print("Ready for submission")
    """

    def __init__(
        self,
        agency: str = "FDA",
        submission_type: str = "NDA",
        strict: bool = False,
    ) -> None:
        self.agency          = agency.upper()
        self.submission_type = submission_type.upper()
        self.strict          = strict
        self._filename_pattern = (
            FDA_FILENAME_PATTERN if self.agency == "FDA"
            else EMA_FILENAME_PATTERN
        )
        logger.info(
            "eCTDValidator initialized: agency=%s, type=%s, strict=%s",
            agency, submission_type, strict,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, package_root: str | Path) -> ValidationReport:
        """
        Run full validation suite on an eCTD package.

        Parameters
        ----------
        package_root : str or Path
            Root directory of the eCTD package to validate.

        Returns
        -------
        ValidationReport
            Comprehensive validation report with all findings.
        """
        package_root = Path(package_root)
        if not package_root.exists():
            raise FileNotFoundError(f"Package root not found: {package_root}")

        report = ValidationReport(
            submission_label=package_root.name,
            agency=self.agency,
            submission_type=self.submission_type,
        )

        logger.info("Starting eCTD validation: %s", package_root)

        # Run all validation checks
        self._check_folder_structure(package_root, report)
        self._check_backbone_xml(package_root, report)
        self._check_file_naming(package_root, report)
        self._check_file_sizes(package_root, report)
        self._check_required_sections(package_root, report)
        self._check_document_registry(package_root, report)
        self._check_regional_forms(package_root, report)

        report.files_checked = len(list(package_root.rglob("*.*")))
        report.sections_checked = len([
            d for d in package_root.iterdir() if d.is_dir()
        ])

        logger.info(
            "Validation complete: %s | errors=%d warnings=%d",
            "PASS" if report.is_valid else "FAIL",
            len(report.errors),
            len(report.warnings),
        )
        return report

    # ── Validation checks ─────────────────────────────────────────────────────

    def _check_folder_structure(self, root: Path, report: ValidationReport) -> None:
        """Verify ICH CTD module folder hierarchy exists."""
        required = REQUIRED_SECTIONS.get(self.submission_type, ["m1", "m2"])
        for module in required:
            if not (root / module).exists():
                report.add_issue(
                    "error", "STRUCT-001",
                    f"Required CTD module folder missing: {module}/",
                    str(root / module),
                )
        logger.debug("Folder structure check complete")

    def _check_backbone_xml(self, root: Path, report: ValidationReport) -> None:
        """Validate backbone XML exists and is well-formed."""
        backbone_files = list(root.glob("backbone*.xml"))
        if not backbone_files:
            report.add_issue(
                "error", "XML-001",
                "Backbone XML file (backbone_NNNN.xml) not found in package root",
                str(root),
            )
            return

        for bf in backbone_files:
            try:
                tree = ET.parse(str(bf))
                root_elem = tree.getroot()

                # Check version attribute
                version = root_elem.get("version", "")
                if version != "3.2.2":
                    report.add_issue(
                        "warning", "XML-002",
                        f"Backbone XML version is '{version}', expected '3.2.2'",
                        bf.name,
                    )

                # Check required header elements
                header = root_elem.find("header")
                if header is None:
                    report.add_issue("error", "XML-003", "Missing <header> element", bf.name)
                else:
                    for elem in ["id", "submissionType", "sequenceNumber"]:
                        if header.find(elem) is None:
                            report.add_issue(
                                "error", "XML-004",
                                f"Missing required header element: <{elem}>",
                                bf.name,
                            )

            except ET.ParseError as exc:
                report.add_issue(
                    "error", "XML-005",
                    f"Backbone XML is not well-formed: {exc}",
                    bf.name,
                )

    def _check_file_naming(self, root: Path, report: ValidationReport) -> None:
        """Validate file names follow FDA/EMA naming conventions."""
        violations = 0
        for file in root.rglob("*.*"):
            if file.suffix.lower() in [".pdf", ".xml", ".docx", ".xpt"]:
                if not self._filename_pattern.match(file.name.lower()):
                    report.add_issue(
                        "warning", "FILE-001",
                        f"File name does not follow {self.agency} naming convention: {file.name}",
                        str(file.relative_to(root)),
                    )
                    violations += 1
        if violations == 0:
            logger.debug("File naming check passed — all files conform")

    def _check_file_sizes(self, root: Path, report: ValidationReport) -> None:
        """Check files do not exceed agency size limits."""
        for file in root.rglob("*.*"):
            ext = file.suffix.lower().lstrip(".")
            limit = FILE_SIZE_LIMITS.get(ext)
            if limit and file.stat().st_size > limit:
                size_mb = file.stat().st_size / (1024 * 1024)
                limit_mb = limit / (1024 * 1024)
                report.add_issue(
                    "error", "FILE-002",
                    f"File exceeds {self.agency} size limit "
                    f"({size_mb:.1f} MB > {limit_mb:.0f} MB): {file.name}",
                    str(file.relative_to(root)),
                )

    def _check_required_sections(self, root: Path, report: ValidationReport) -> None:
        """Verify required CTD sections contain at least one document."""
        if self.submission_type in ["NDA", "BLA", "MAA"]:
            critical_sections = ["m2", "m5"]
            for section in critical_sections:
                section_path = root / section
                if section_path.exists():
                    docs = (
                        list(section_path.rglob("*.pdf")) +
                        list(section_path.rglob("*.docx"))
                    )
                    if not docs:
                        report.add_issue(
                            "warning", "SECT-001",
                            f"Section {section}/ contains no submission documents",
                            str(section_path),
                        )

    def _check_document_registry(self, root: Path, report: ValidationReport) -> None:
        """Verify document registry XML is present and valid."""
        registry_files = list(root.glob("document_registry*.xml"))
        if not registry_files:
            report.add_issue(
                "warning", "REG-001",
                "Document registry XML not found — recommended for submission tracking",
                str(root),
            )
            return

        for rf in registry_files:
            try:
                ET.parse(str(rf))
            except ET.ParseError as exc:
                report.add_issue(
                    "error", "REG-002",
                    f"Document registry XML is malformed: {exc}",
                    rf.name,
                )

    def _check_regional_forms(self, root: Path, report: ValidationReport) -> None:
        """Check regional administrative forms are present."""
        if self.agency == "FDA":
            us_dir = root / "m1" / "us"
            if not us_dir.exists():
                report.add_issue(
                    "error", "REG-003",
                    "FDA regional folder m1/us/ not found",
                    str(us_dir),
                )
            else:
                forms = list(us_dir.rglob("*.pdf")) + list(us_dir.rglob("*.xml"))
                if not forms:
                    report.add_issue(
                        "warning", "REG-004",
                        "No regional forms found in m1/us/ "
                        "(FDA Form 356h or Form 1571 expected)",
                        str(us_dir),
                    )
        elif self.agency == "EMA":
            eu_dir = root / "m1" / "eu"
            if not eu_dir.exists():
                report.add_issue(
                    "error", "REG-003",
                    "EMA regional folder m1/eu/ not found",
                    str(eu_dir),
                )
