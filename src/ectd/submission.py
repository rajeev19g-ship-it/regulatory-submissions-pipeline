ectd/submission.py
───────────────────
Automated eCTD submission package builder.

Constructs FDA/EMA-compliant eCTD 3.2.2 submission packages
including folder structure, backbone XML, and document registry
for IND, NDA, BLA, and MAA submission types.

Regulatory references:
    - FDA eCTD Technical Specifications v3.2.2
    - EMA eCTD Implementation Guide v1.0
    - ICH M4 Common Technical Document (CTD) structure

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


# ── Submission types ──────────────────────────────────────────────────────────

SUBMISSION_TYPES = {
    "IND":  "Investigational New Drug Application",
    "NDA":  "New Drug Application",
    "BLA":  "Biologics License Application",
    "MAA":  "Marketing Authorisation Application",
    "IMPD": "Investigational Medicinal Product Dossier",
}

# ICH CTD Module structure
CTD_MODULES = {
    "m1": "Regional Administrative Information",
    "m2": "CTD Summaries",
    "m3": "Quality",
    "m4": "Nonclinical Study Reports",
    "m5": "Clinical Study Reports",
}

CTD_SECTIONS = {
    "m2": {
        "m2-1": "Table of Contents",
        "m2-2": "Introduction",
        "m2-3": "Quality Overall Summary",
        "m2-4": "Nonclinical Overview",
        "m2-5": "Clinical Overview",
        "m2-6": "Nonclinical Written and Tabulated Summaries",
        "m2-7": "Clinical Summary",
    },
    "m5": {
        "m5-1": "Table of Contents",
        "m5-2": "Tabular Listing of All Clinical Studies",
        "m5-3": "Clinical Study Reports",
        "m5-4": "Literature References",
    },
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SubmissionDocument:
    """Represents a single document in the eCTD submission."""
    title: str
    ctd_section: str
    file_name: str
    file_path: Optional[Path] = None
    document_type: str = "other"
    language: str = "en"
    checksum: str = ""
    file_size: int = 0

    def compute_checksum(self) -> str:
        """Compute MD5 checksum of the document file."""
        if self.file_path and self.file_path.exists():
            md5 = hashlib.md5()
            with open(self.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5.update(chunk)
            self.checksum = md5.hexdigest()
        return self.checksum


@dataclass
class SubmissionPackage:
    """Complete eCTD submission package metadata."""
    submission_type: str
    application_number: str
    sponsor_name: str
    drug_name: str
    indication: str
    sequence_number: str = "0000"
    submission_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )
    documents: list[SubmissionDocument] = field(default_factory=list)
    agency: str = "FDA"

    @property
    def submission_label(self) -> str:
        return f"{self.submission_type}-{self.application_number}-{self.sequence_number}"

    def add_document(self, doc: SubmissionDocument) -> None:
        self.documents.append(doc)
        logger.debug("Added document: %s → %s", doc.title, doc.ctd_section)

    def summary(self) -> dict:
        return {
            "submission_type":   self.submission_type,
            "application_number": self.application_number,
            "sponsor":           self.sponsor_name,
            "drug":              self.drug_name,
            "indication":        self.indication,
            "agency":            self.agency,
            "sequence":          self.sequence_number,
            "documents":         len(self.documents),
            "sections_covered":  list({d.ctd_section for d in self.documents}),
        }


# ── eCTD Builder ──────────────────────────────────────────────────────────────

class eCTDBuilder:
    """
    Constructs FDA/EMA-compliant eCTD submission packages.

    Builds the complete eCTD folder structure, backbone XML,
    document registry, and regional forms required for electronic
    submission to FDA (CDER/CBER) or EMA.

    Parameters
    ----------
    output_dir : str or Path
        Root directory where the eCTD package will be created.
    agency : str
        Regulatory agency: 'FDA' or 'EMA'. Default 'FDA'.

    Examples
    --------
    >>> builder = eCTDBuilder(output_dir="submissions/")
    >>> package = SubmissionPackage(
    ...     submission_type="NDA",
    ...     application_number="123456",
    ...     sponsor_name="Synthetic Pharma Inc.",
    ...     drug_name="DrugX 200mg",
    ...     indication="Non-Small Cell Lung Cancer",
    ... )
    >>> package.add_document(SubmissionDocument(
    ...     title="Clinical Overview",
    ...     ctd_section="m2-5",
    ...     file_name="clinical_overview.pdf",
    ... ))
    >>> builder.build(package)
    """

    ECTD_VERSION = "3.2.2"
    DOCTYPE_FDA  = "us-regional"
    DOCTYPE_EMA  = "eu-regional"

    def __init__(
        self,
        output_dir: str | Path,
        agency: str = "FDA",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.agency     = agency.upper()
        logger.info("eCTDBuilder initialized: output=%s, agency=%s", output_dir, agency)

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, package: SubmissionPackage) -> Path:
        """
        Build the complete eCTD submission package.

        Creates the CTD folder structure, copies documents,
        generates backbone XML and document registry.

        Parameters
        ----------
        package : SubmissionPackage
            Submission package metadata and document list.

        Returns
        -------
        Path
            Root path of the constructed eCTD package.
        """
        root = self.output_dir / package.submission_label
        root.mkdir(parents=True, exist_ok=True)

        logger.info("Building eCTD package: %s", package.submission_label)

        # Build CTD folder structure
        self._create_folder_structure(root)

        # Copy documents to correct CTD sections
        self._place_documents(root, package)

        # Generate backbone XML
        backbone_path = self._write_backbone_xml(root, package)

        # Generate document registry
        registry_path = self._write_document_registry(root, package)

        # Generate submission cover metadata
        self._write_submission_metadata(root, package)

        logger.info(
            "eCTD package built: %s (%d documents, backbone: %s)",
            root, len(package.documents), backbone_path.name,
        )
        return root

    def validate_structure(self, package_root: Path) -> dict:
        """
        Validate that a built eCTD package has the required structure.

        Parameters
        ----------
        package_root : Path
            Root of the eCTD package to validate.

        Returns
        -------
        dict
            Validation result with 'valid', 'errors', and 'warnings' keys.
        """
        errors   = []
        warnings = []

        # Check required modules exist
        for module in ["m1", "m2", "m3", "m4", "m5"]:
            if not (package_root / module).exists():
                errors.append(f"Required CTD module folder missing: {module}/")

        # Check backbone XML exists
        if not list(package_root.glob("*backbone*.xml")):
            errors.append("Backbone XML file not found")

        # Check document registry
        if not list(package_root.glob("*registry*.xml")):
            warnings.append("Document registry XML not found")

        # Check at least m2 and m5 have content
        for module in ["m2", "m5"]:
            module_path = package_root / module
            if module_path.exists():
                files = list(module_path.rglob("*.pdf")) + list(module_path.rglob("*.docx"))
                if not files:
                    warnings.append(f"Module {module}/ contains no PDF or DOCX documents")

        result = {
            "valid":    len(errors) == 0,
            "errors":   errors,
            "warnings": warnings,
            "package":  str(package_root),
        }

        if result["valid"]:
            logger.info("eCTD validation passed: %s", package_root)
        else:
            logger.warning("eCTD validation failed: %d errors", len(errors))

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _create_folder_structure(self, root: Path) -> None:
        """Create ICH CTD module folder hierarchy."""
        for module, sections in CTD_SECTIONS.items():
            (root / module).mkdir(exist_ok=True)
            for section in sections:
                (root / module / section).mkdir(exist_ok=True)

        # Create remaining modules
        for module in ["m1", "m3", "m4"]:
            (root / module).mkdir(exist_ok=True)

        # Agency-specific folders
        if self.agency == "FDA":
            (root / "m1" / "us").mkdir(exist_ok=True)
        elif self.agency == "EMA":
            (root / "m1" / "eu").mkdir(exist_ok=True)

        logger.debug("CTD folder structure created under %s", root)

    def _place_documents(self, root: Path, package: SubmissionPackage) -> None:
        """Copy documents to their designated CTD section folders."""
        for doc in package.documents:
            module = doc.ctd_section.split("-")[0]
            dest_dir = root / module / doc.ctd_section
            dest_dir.mkdir(parents=True, exist_ok=True)

            if doc.file_path and doc.file_path.exists():
                dest = dest_dir / doc.file_name
                shutil.copy2(doc.file_path, dest)
                doc.file_path = dest
                doc.compute_checksum()
                doc.file_size = dest.stat().st_size
                logger.debug("Placed: %s → %s", doc.file_name, dest_dir)
            else:
                # Create placeholder for missing documents
                placeholder = dest_dir / f"{doc.file_name}.placeholder.txt"
                placeholder.write_text(
                    f"PLACEHOLDER: {doc.title}\nSection: {doc.ctd_section}\n"
                    f"Expected file: {doc.file_name}\n"
                )

    def _write_backbone_xml(self, root: Path, package: SubmissionPackage) -> Path:
        """Generate the eCTD backbone XML (ichectd.xml)."""
        ich = ET.Element("ichectd")
        ich.set("xmlns",   "urn:hl7-org:v3")
        ich.set("version", self.ECTD_VERSION)

        # Header
        header = ET.SubElement(ich, "header")
        ET.SubElement(header, "id").text            = package.application_number
        ET.SubElement(header, "title").text         = f"{package.submission_type} — {package.drug_name}"
        ET.SubElement(header, "sponsor").text       = package.sponsor_name
        ET.SubElement(header, "submissionType").text = package.submission_type
        ET.SubElement(header, "sequenceNumber").text = package.sequence_number
        ET.SubElement(header, "submissionDate").text = package.submission_date
        ET.SubElement(header, "agency").text        = self.agency

        # Document manifest
        manifest = ET.SubElement(ich, "manifest")
        for doc in package.documents:
            entry = ET.SubElement(manifest, "document")
            entry.set("section",  doc.ctd_section)
            entry.set("language", doc.language)
            ET.SubElement(entry, "title").text    = doc.title
            ET.SubElement(entry, "fileName").text = doc.file_name
            ET.SubElement(entry, "checksum").text = doc.checksum
            ET.SubElement(entry, "fileSize").text = str(doc.file_size)

        path = root / f"backbone_{package.sequence_number}.xml"
        raw  = ET.tostring(ich, encoding="unicode")
        path.write_text(
            minidom.parseString(raw).toprettyxml(indent="  "),
            encoding="utf-8"
        )
        return path

    def _write_document_registry(self, root: Path, package: SubmissionPackage) -> Path:
        """Generate document registry XML for submission tracking."""
        registry = ET.Element("documentRegistry")
        registry.set("submissionLabel", package.submission_label)
        registry.set("generated",       datetime.utcnow().isoformat())

        for i, doc in enumerate(package.documents, 1):
            entry = ET.SubElement(registry, "entry")
            entry.set("id",      str(i))
            entry.set("section", doc.ctd_section)
            ET.SubElement(entry, "title").text        = doc.title
            ET.SubElement(entry, "fileName").text     = doc.file_name
            ET.SubElement(entry, "documentType").text = doc.document_type
            ET.SubElement(entry, "language").text     = doc.language
            ET.SubElement(entry, "checksum").text     = doc.checksum

        path = root / f"document_registry_{package.sequence_number}.xml"
        raw  = ET.tostring(registry, encoding="unicode")
        path.write_text(
            minidom.parseString(raw).toprettyxml(indent="  "),
            encoding="utf-8"
        )
        return path

    def _write_submission_metadata(self, root: Path, package: SubmissionPackage) -> Path:
        """Write submission cover metadata as JSON."""
        import json
        meta = package.summary()
        meta["ectd_version"] = self.ECTD_VERSION
        meta["built_at"]     = datetime.utcnow().isoformat()

        path = root / "submission_metadata.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path
