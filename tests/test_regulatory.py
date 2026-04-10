Now let's add the complete test suite for Repo 2:

Click the tests folder from the main repo page
Click "Add file" → "Create new file"
Type in the filename box:

test_regulatory.py

Paste this code:

python"""
tests/test_regulatory.py
─────────────────────────
Unit tests for all Repo 2 modules:
    - eCTD submission builder and validator
    - IND/NDA document generators (mocked LLM)
    - Medical translation pipeline (mocked LLM)
    - Terminology validator
    - Biomarker analyzer
    - Patient stratifier
    - Target predictor
    - Molecule screener

Uses synthetic data — no real patient data or API calls required.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ectd.submission import (
    eCTDBuilder, SubmissionPackage, SubmissionDocument,
)
from src.ectd.validator import eCTDValidator
from src.translation.term_validator import (
    MedicalTermValidator, DocumentValidationReport,
)
from src.precision.biomarker import BiomarkerAnalyzer, BiomarkerDefinition
from src.precision.patient_stratification import PatientStratifier
from src.drug_discovery.target_predictor import (
    TargetPredictor, MolecularDescriptors, BindingPrediction,
)
from src.drug_discovery.molecule_screener import (
    MoleculeScreener, ScreeningCampaign, ScreeningHit,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_package() -> SubmissionPackage:
    """Synthetic NDA submission package."""
    pkg = SubmissionPackage(
        submission_type="NDA",
        application_number="123456",
        sponsor_name="Synthetic Pharma Inc.",
        drug_name="DrugX 200mg",
        indication="Non-Small Cell Lung Cancer",
        sequence_number="0000",
    )
    pkg.add_document(SubmissionDocument(
        title="Clinical Overview",
        ctd_section="m2-5",
        file_name="clinical_overview.pdf",
    ))
    pkg.add_document(SubmissionDocument(
        title="Nonclinical Overview",
        ctd_section="m2-4",
        file_name="nonclinical_overview.pdf",
    ))
    return pkg


@pytest.fixture
def built_package(tmp_path, sample_package) -> Path:
    """Build a complete eCTD package in a temp directory."""
    builder = eCTDBuilder(output_dir=tmp_path, agency="FDA")
    return builder.build(sample_package)


@pytest.fixture
def biomarker_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic biomarker data with binary response."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "PDL1_TPS":   np.random.uniform(0, 100, n),
        "TMB":        np.random.uniform(0, 50, n),
        "EGFR_mut":   np.random.binomial(1, 0.3, n).astype(float),
        "ALK_fusion": np.random.binomial(1, 0.1, n).astype(float),
        "CRP":        np.random.exponential(5, n),
    })
    response = pd.Series(
        (df["PDL1_TPS"] > 50).astype(int) |
        (df["TMB"] > 30).astype(int)
    ).clip(0, 1)
    return df, response


@pytest.fixture
def molecular_library() -> list[MolecularDescriptors]:
    """Synthetic molecular descriptor library."""
    np.random.seed(42)
    compounds = []
    for i in range(50):
        compounds.append(MolecularDescriptors(
            compound_id=f"CPD-{i:04d}",
            molecular_weight=np.random.uniform(200, 600),
            logp=np.random.uniform(-1, 6),
            h_bond_donors=np.random.randint(0, 6),
            h_bond_acceptors=np.random.randint(0, 11),
            rotatable_bonds=np.random.randint(0, 12),
            topological_polar_surface_area=np.random.uniform(20, 160),
            aromatic_rings=np.random.randint(0, 5),
            heavy_atom_count=np.random.randint(15, 45),
            fingerprint_bits=np.random.randint(0, 2, 64).tolist(),
        ))
    return compounds


@pytest.fixture
def trained_predictor(molecular_library) -> TargetPredictor:
    """Pre-trained QSAR predictor."""
    np.random.seed(42)
    pic50_values = [
        5.0 + 3.0 * (1 / (1 + np.exp(-0.01 * (c.molecular_weight - 400))))
        + np.random.normal(0, 0.3)
        for c in molecular_library
    ]
    predictor = TargetPredictor(model_type="random_forest", random_state=42)
    predictor.train(molecular_library, pic50_values, target_name="EGFR")
    return predictor


# ── eCTD Builder tests ────────────────────────────────────────────────────────

class TestECTDBuilder:

    def test_build_creates_root_folder(self, built_package):
        assert built_package.exists()
        assert built_package.is_dir()

    def test_ctd_module_folders_created(self, built_package):
        for module in ["m1", "m2", "m3", "m4", "m5"]:
            assert (built_package / module).exists()

    def test_backbone_xml_created(self, built_package):
        backbone_files = list(built_package.glob("backbone*.xml"))
        assert len(backbone_files) == 1

    def test_document_registry_created(self, built_package):
        registry_files = list(built_package.glob("document_registry*.xml"))
        assert len(registry_files) == 1

    def test_metadata_json_created(self, built_package):
        assert (built_package / "submission_metadata.json").exists()

    def test_metadata_content(self, built_package):
        meta = json.loads(
            (built_package / "submission_metadata.json").read_text()
        )
        assert meta["submission_type"] == "NDA"
        assert meta["application_number"] == "123456"
        assert meta["documents"] == 2

    def test_backbone_xml_valid(self, built_package):
        from xml.etree import ElementTree as ET
        backbone = list(built_package.glob("backbone*.xml"))[0]
        tree = ET.parse(str(backbone))
        root = tree.getroot()
        assert root.tag == "ichectd"
        assert root.get("version") == "3.2.2"

    def test_submission_label(self):
        pkg = SubmissionPackage(
            submission_type="IND",
            application_number="999999",
            sponsor_name="Test Sponsor",
            drug_name="TestDrug",
            indication="Test Indication",
            sequence_number="0001",
        )
        assert pkg.submission_label == "IND-999999-0001"

    def test_package_summary(self, sample_package):
        summary = sample_package.summary()
        assert summary["submission_type"] == "NDA"
        assert summary["documents"] == 2


# ── eCTD Validator tests ──────────────────────────────────────────────────────

class TestECTDValidator:

    def test_validate_built_package(self, built_package):
        validator = eCTDValidator(agency="FDA", submission_type="NDA")
        report = validator.validate(built_package)
        assert isinstance(report.is_valid, bool)

    def test_missing_package_raises(self, tmp_path):
        validator = eCTDValidator()
        with pytest.raises(FileNotFoundError):
            validator.validate(tmp_path / "nonexistent")

    def test_report_summary_keys(self, built_package):
        validator = eCTDValidator(agency="FDA", submission_type="NDA")
        report = validator.validate(built_package)
        summary = report.summary()
        assert "valid" in summary
        assert "errors" in summary
        assert "warnings" in summary
        assert "files_checked" in summary

    def test_errors_and_warnings_lists(self, built_package):
        validator = eCTDValidator(agency="FDA", submission_type="NDA")
        report = validator.validate(built_package)
        assert isinstance(report.errors, list)
        assert isinstance(report.warnings, list)

    def test_ind_requires_fewer_modules(self, tmp_path):
        builder = eCTDBuilder(output_dir=tmp_path, agency="FDA")
        pkg = SubmissionPackage(
            submission_type="IND",
            application_number="111111",
            sponsor_name="Test",
            drug_name="TestDrug",
            indication="Test",
        )
        root = builder.build(pkg)
        validator = eCTDValidator(agency="FDA", submission_type="IND")
        report = validator.validate(root)
        assert isinstance(report.is_valid, bool)


# ── MedicalTermValidator tests ────────────────────────────────────────────────

class TestMedicalTermValidator:

    def test_valid_meddra_term(self):
        validator = MedicalTermValidator()
        result = validator.validate_meddra_term("nausea")
        assert result.is_valid
        assert result.term_code == "10028813"

    def test_invalid_term_with_suggestion(self):
        validator = MedicalTermValidator()
        result = validator.validate_meddra_term("hair loss")
        assert not result.is_valid
        assert result.suggestion == "alopecia"

    def test_nonstandard_fever_maps_to_pyrexia(self):
        validator = MedicalTermValidator()
        result = validator.validate_meddra_term("fever")
        assert not result.is_valid
        assert result.suggestion == "pyrexia"

    def test_validate_ae_list(self):
        validator = MedicalTermValidator()
        terms = ["nausea", "fatigue", "hair loss", "fever", "headache"]
        report = validator.validate_ae_list(terms, "Test AE list")
        assert isinstance(report, DocumentValidationReport)
        assert report.total_terms == 5
        assert report.valid_terms + report.invalid_terms == 5

    def test_compliance_rate(self):
        validator = MedicalTermValidator()
        report = validator.validate_ae_list(
            ["nausea", "fatigue", "headache"], "Test"
        )
        assert report.compliance_rate == 100.0

    def test_standardize_ae_terms(self):
        validator = MedicalTermValidator()
        mapping = validator.standardize_ae_terms(["fever", "nausea", "hair loss"])
        assert mapping["fever"] == "pyrexia"
        assert mapping["nausea"] == "nausea"
        assert mapping["hair loss"] == "alopecia"

    def test_ctcae_grade_valid(self):
        validator = MedicalTermValidator()
        result = validator.validate_ctcae_grade(3)
        assert result["valid"]
        assert "Severe" in result["description"]

    def test_ctcae_grade_invalid(self):
        validator = MedicalTermValidator()
        result = validator.validate_ctcae_grade(6)
        assert not result["valid"]

    def test_get_soc_for_term(self):
        validator = MedicalTermValidator()
        soc = validator.get_soc_for_term("nausea")
        assert soc == "Gastrointestinal disorders"

    def test_meddra_coding_table(self):
        validator = MedicalTermValidator()
        table = validator.generate_meddra_coding_table(
            ["nausea", "fatigue", "fever"]
        )
        assert len(table) == 3
        assert all("pt_code" in row for row in table)
        assert all("soc" in row for row in table)

    def test_extract_ae_terms_from_text(self):
        validator = MedicalTermValidator()
        text = "Patient reported nausea and fatigue during treatment."
        found = validator.extract_ae_terms_from_text(text)
        assert "nausea" in found
        assert "fatigue" in found


# ── BiomarkerAnalyzer tests ───────────────────────────────────────────────────

class TestBiomarkerAnalyzer:

    def test_predictive_biomarker_returns_result(self, biomarker_data):
        df, response = biomarker_data
        analyzer = BiomarkerAnalyzer()
        result = analyzer.analyze_predictive_biomarker(
            biomarker_values=df["PDL1_TPS"],
            response=response,
            biomarker_name="PDL1_TPS",
        )
        assert result.auc_roc is not None
        assert 0 <= result.auc_roc <= 1

    def test_auc_in_valid_range(self, biomarker_data):
        df, response = biomarker_data
        analyzer = BiomarkerAnalyzer()
        result = analyzer.analyze_predictive_biomarker(
            df["TMB"], response, "TMB"
        )
        assert 0.0 <= result.auc_roc <= 1.0

    def test_threshold_optimization(self, biomarker_data):
        df, response = biomarker_data
        analyzer = BiomarkerAnalyzer()
        result = analyzer.analyze_predictive_biomarker(
            df["PDL1_TPS"], response, "PDL1_TPS",
            optimize_threshold=True,
        )
        assert result.optimal_threshold is not None
        assert result.sensitivity is not None
        assert result.specificity is not None

    def test_panel_analysis(self, biomarker_data):
        df, response = biomarker_data
        analyzer = BiomarkerAnalyzer()
        panel = analyzer.analyze_biomarker_panel(
            df, response, panel_name="Oncology Panel"
        )
        assert panel.composite_auc > 0
        assert len(panel.feature_importances) == df.shape[1]

    def test_cdx_validation_metrics(self):
        np.random.seed(42)
        n = 100
        ref  = pd.Series(np.random.binomial(1, 0.4, n))
        test = ref.copy()
        test.iloc[:10] = 1 - test.iloc[:10]

        analyzer = BiomarkerAnalyzer()
        metrics  = analyzer.validate_cdx_analytical_performance(
            test, ref, "Test CDx Assay"
        )
        assert "sensitivity_ppa" in metrics
        assert "specificity_npa" in metrics
        assert "overall_percent_agreement" in metrics
        assert 0 <= metrics["sensitivity_ppa"] <= 1


# ── PatientStratifier tests ───────────────────────────────────────────────────

class TestPatientStratifier:

    def test_cluster_returns_result(self, biomarker_data):
        df, _ = biomarker_data
        stratifier = PatientStratifier()
        result = stratifier.cluster_patients(df, n_clusters=3)
        assert result.n_subgroups == 3
        assert result.n_patients == len(df.dropna())

    def test_subgroup_sizes_sum_to_n(self, biomarker_data):
        df, _ = biomarker_data
        stratifier = PatientStratifier()
        result = stratifier.cluster_patients(df, n_clusters=3)
        assert sum(result.subgroup_sizes.values()) == result.n_patients

    def test_silhouette_score_valid(self, biomarker_data):
        df, _ = biomarker_data
        stratifier = PatientStratifier()
        result = stratifier.cluster_patients(df, n_clusters=3)
        assert result.silhouette_score is not None
        assert -1 <= result.silhouette_score <= 1

    def test_optimal_clusters(self, biomarker_data):
        df, _ = biomarker_data
        stratifier = PatientStratifier()
        result = stratifier.find_optimal_clusters(df, k_range=range(2, 5))
        assert "optimal_k" in result
        assert result["optimal_k"] in range(2, 5)

    def test_enrichment_model_training(self, biomarker_data):
        df, response = biomarker_data
        stratifier = PatientStratifier()
        metrics = stratifier.train_enrichment_model(df, response)
        assert "cv_auc" in metrics
        assert 0 <= metrics["cv_auc"] <= 1

    def test_enrichment_scoring(self, biomarker_data):
        df, response = biomarker_data
        stratifier = PatientStratifier()
        stratifier.train_enrichment_model(df, response)
        patient = {col: float(df[col].iloc[0]) for col in df.columns}
        score = stratifier.score_patient_for_enrollment(patient, "TEST-001")
        assert 0 <= score.enrichment_probability <= 1
        assert score.enrollment_recommendation != ""

    def test_untrained_model_raises(self, biomarker_data):
        df, _ = biomarker_data
        stratifier = PatientStratifier()
        with pytest.raises(RuntimeError, match="not trained"):
            stratifier.score_patient_for_enrollment(
                {col: 1.0 for col in df.columns}
            )


# ── TargetPredictor tests ─────────────────────────────────────────────────────

class TestTargetPredictor:

    def test_train_returns_metrics(self, molecular_library):
        np.random.seed(42)
        pic50 = [6.0 + np.random.normal(0, 0.5) for _ in molecular_library]
        predictor = TargetPredictor(model_type="random_forest")
        metrics = predictor.train(molecular_library, pic50, "EGFR")
        assert "r2_train" in metrics
        assert "rmse_train" in metrics
        assert "cv_r2_mean" in metrics

    def test_predict_returns_binding_prediction(self, trained_predictor, molecular_library):
        result = trained_predictor.predict(molecular_library[0])
        assert isinstance(result, BindingPrediction)
        assert result.predicted_pic50 > 0

    def test_ki_derived_from_pic50(self, trained_predictor, molecular_library):
        result = trained_predictor.predict(molecular_library[0])
        expected_ki = 10 ** (9 - result.predicted_pic50)
        assert abs(result.predicted_ki_nm - expected_ki) < 0.01

    def test_activity_class_labels(self):
        bp = BindingPrediction(
            compound_id="TEST",
            target_name="EGFR",
            predicted_pic50=8.5,
            predicted_ki_nm=3.16,
        )
        assert "highly active" in bp.activity_class

    def test_untrained_predict_raises(self, molecular_library):
        predictor = TargetPredictor()
        with pytest.raises(RuntimeError, match="must be trained"):
            predictor.predict(molecular_library[0])

    def test_admet_prediction(self, trained_predictor, molecular_library):
        admet = trained_predictor.predict_admet(molecular_library[0])
        assert "absorption" in admet
        assert "distribution" in admet
        assert "toxicity" in admet

    def test_lipinski_ro5(self):
        mol = MolecularDescriptors(
            compound_id="HEAVY",
            molecular_weight=600,
            logp=6,
            h_bond_donors=6,
            h_bond_acceptors=11,
        )
        result = mol.lipinski_ro5()
        assert not result["passes_ro5"]
        assert result["n_violations"] > 0

    def test_batch_predict(self, trained_predictor, molecular_library):
        results = trained_predictor.predict_batch(molecular_library[:5])
        assert len(results) == 5
        assert all(isinstance(r, BindingPrediction) for r in results)


# ── MoleculeScreener tests ────────────────────────────────────────────────────

class TestMoleculeScreener:

    def test_campaign_runs(self, molecular_library, trained_predictor):
        screener = MoleculeScreener(pic50_threshold=5.0)
        campaign = screener.run_campaign(
            compounds=molecular_library,
            predictor=trained_predictor,
            target_name="EGFR",
            campaign_name="test_campaign",
        )
        assert isinstance(campaign, ScreeningCampaign)
        assert campaign.library_size == len(molecular_library)

    def test_hit_rate_in_range(self, molecular_library, trained_predictor):
        screener = MoleculeScreener(pic50_threshold=5.0)
        campaign = screener.run_campaign(
            molecular_library, trained_predictor, "EGFR"
        )
        assert 0 <= campaign.hit_rate <= 100

    def test_hits_sorted_by_pic50(self, molecular_library, trained_predictor):
        screener = MoleculeScreener(pic50_threshold=5.0)
        campaign = screener.run_campaign(
            molecular_library, trained_predictor, "EGFR"
        )
        if len(campaign.hits) > 1:
            pic50s = [h.predicted_pic50 for h in campaign.hits]
            assert pic50s == sorted(pic50s, reverse=True)

    def test_scaffold_diversity(self, molecular_library):
        screener = MoleculeScreener()
        result = screener.analyze_scaffold_diversity(molecular_library[:20])
        assert "diversity_score" in result
        assert 0 <= result["diversity_score"] <= 1

    def test_lead_prioritization(self, molecular_library, trained_predictor):
        screener = MoleculeScreener(pic50_threshold=5.0)
        campaign = screener.run_campaign(
            molecular_library, trained_predictor, "EGFR"
        )
        leads = screener.prioritize_leads(campaign, max_leads=5)
        assert len(leads) <= 5

    def test_screening_report_keys(self, molecular_library, trained_predictor):
        screener = MoleculeScreener(pic50_threshold=5.0)
        campaign = screener.run_campaign(
            molecular_library, trained_predictor, "EGFR"
        )
        report = screener.generate_screening_report(campaign)
        assert "campaign_summary" in report
        assert "hit_statistics" in report
        assert "top_10_hits" in report
        assert "prioritized_leads" in report

    def test_druglike_filter_removes_compounds(self, molecular_library, trained_predictor):
        screener = MoleculeScreener(
            filter_preset="fragment",
            pic50_threshold=5.0,
        )
        campaign = screener.run_campaign(
            molecular_library, trained_predictor, "EGFR"
        )
        assert campaign.n_druglike_filtered >= 0
        assert campaign.n_screened <= campaign.library_size
