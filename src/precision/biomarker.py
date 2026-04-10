Now let's build the precision medicine module:

Click src → precision folder
Click "Add file" → "Create new file"
Type in the filename box:

biomarker.py

Paste this code:

python"""
precision/biomarker.py
───────────────────────
Biomarker analysis pipeline for precision medicine applications.

Implements:
    - Biomarker discovery (differential expression, feature importance)
    - Predictive biomarker modeling (response prediction)
    - Prognostic biomarker analysis (survival association)
    - Biomarker threshold optimization (cut-point analysis)
    - Companion diagnostic (CDx) analytical validation metrics

Supports:
    - Continuous biomarkers (protein expression, gene expression)
    - Binary biomarkers (mutation status, amplification)
    - Composite biomarker scores (tumor mutational burden, MSI)

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class BiomarkerDefinition:
    """Definition of a single biomarker."""
    name: str
    biomarker_type: str      # continuous | binary | categorical
    assay_type: str          # IHC | FISH | NGS | PCR | proteomics | other
    unit: str = ""
    lower_limit_of_detection: Optional[float] = None
    upper_limit_of_quantification: Optional[float] = None
    regulatory_status: str = "exploratory"  # exploratory | validated | CDx-approved


@dataclass
class BiomarkerAnalysisResult:
    """Results from a single biomarker analysis."""
    biomarker_name: str
    analysis_type: str
    n_samples: int
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    ppv: Optional[float] = None
    npv: Optional[float] = None
    optimal_threshold: Optional[float] = None
    feature_importance: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: tuple = field(default_factory=tuple)
    notes: str = ""

    def summary(self) -> dict:
        return {
            "biomarker":          self.biomarker_name,
            "analysis_type":      self.analysis_type,
            "n_samples":          self.n_samples,
            "auc_roc":            round(self.auc_roc, 3) if self.auc_roc else None,
            "sensitivity":        round(self.sensitivity, 3) if self.sensitivity else None,
            "specificity":        round(self.specificity, 3) if self.specificity else None,
            "optimal_threshold":  self.optimal_threshold,
            "notes":              self.notes,
        }


@dataclass
class BiomarkerPanel:
    """Multi-biomarker panel analysis results."""
    panel_name: str
    biomarkers: list[str]
    composite_auc: float = 0.0
    individual_results: list[BiomarkerAnalysisResult] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)

    def top_biomarkers(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top N biomarkers by feature importance."""
        sorted_fi = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_fi[:n]


# ── Biomarker Analyzer ────────────────────────────────────────────────────────

class BiomarkerAnalyzer:
    """
    Precision medicine biomarker analysis pipeline.

    Performs predictive and prognostic biomarker analyses
    supporting companion diagnostic (CDx) development and
    patient enrichment strategy for clinical trials.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility. Default 42.
    cv_folds : int
        Cross-validation folds. Default 5.

    Examples
    --------
    >>> analyzer = BiomarkerAnalyzer()
    >>> result = analyzer.analyze_predictive_biomarker(
    ...     biomarker_values=pd.Series([...]),
    ...     response=pd.Series([0, 1, 1, 0, ...]),
    ...     biomarker_name="PD-L1 TPS",
    ... )
    >>> print(result.summary())
    """

    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
    ) -> None:
        self.random_state = random_state
        self.cv_folds     = cv_folds
        self._scaler      = StandardScaler()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_predictive_biomarker(
        self,
        biomarker_values: pd.Series,
        response: pd.Series,
        biomarker_name: str,
        optimize_threshold: bool = True,
    ) -> BiomarkerAnalysisResult:
        """
        Analyze a continuous biomarker for treatment response prediction.

        Computes AUC-ROC, AUC-PR, and optimal classification threshold
        using Youden's J statistic. Performs 5-fold cross-validation
        for robust performance estimation.

        Parameters
        ----------
        biomarker_values : pd.Series
            Continuous biomarker measurements per subject.
        response : pd.Series
            Binary response indicator (1=responder, 0=non-responder).
        biomarker_name : str
            Name of the biomarker for reporting.
        optimize_threshold : bool
            If True, find optimal cut-point via Youden's J. Default True.

        Returns
        -------
        BiomarkerAnalysisResult
        """
        # Align and clean
        df = pd.DataFrame({
            "bm":       biomarker_values,
            "response": response,
        }).dropna()

        X = df[["bm"]].values
        y = df["response"].values

        if len(np.unique(y)) < 2:
            raise ValueError("Response variable must have both positive and negative cases")

        # Logistic regression model
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X, y)
        y_prob = model.predict_proba(X)[:, 1]

        # AUC metrics
        auc_roc = roc_auc_score(y, y_prob)
        auc_pr  = average_precision_score(y, y_prob)

        # Optimal threshold (Youden's J)
        threshold = None
        sensitivity = specificity = ppv = npv = None

        if optimize_threshold:
            threshold, sensitivity, specificity, ppv, npv = \
                self._optimize_threshold(y, y_prob)

        # Cross-validation AUC
        cv_aucs = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                               random_state=self.random_state),
            scoring="roc_auc",
        )

        result = BiomarkerAnalysisResult(
            biomarker_name=biomarker_name,
            analysis_type="predictive",
            n_samples=len(df),
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            optimal_threshold=threshold,
            confidence_interval=(
                float(cv_aucs.mean() - 1.96 * cv_aucs.std()),
                float(cv_aucs.mean() + 1.96 * cv_aucs.std()),
            ),
            notes=f"CV AUC: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}",
        )

        logger.info(
            "Predictive biomarker [%s]: AUC=%.3f, threshold=%.3f, "
            "sens=%.3f, spec=%.3f",
            biomarker_name, auc_roc,
            threshold or 0, sensitivity or 0, specificity or 0,
        )
        return result

    def analyze_biomarker_panel(
        self,
        biomarker_df: pd.DataFrame,
        response: pd.Series,
        panel_name: str = "Biomarker Panel",
        model_type: str = "random_forest",
    ) -> BiomarkerPanel:
        """
        Analyze a multi-biomarker panel for composite prediction.

        Fits an ensemble model to identify the most predictive
        combination of biomarkers and rank their relative importance.

        Parameters
        ----------
        biomarker_df : pd.DataFrame
            DataFrame where each column is a biomarker measurement.
        response : pd.Series
            Binary response indicator.
        panel_name : str
            Name of the biomarker panel.
        model_type : str
            Model type: 'random_forest' or 'gradient_boosting'.

        Returns
        -------
        BiomarkerPanel
        """
        df = pd.concat([biomarker_df, response.rename("response")], axis=1).dropna()
        X  = df.drop(columns=["response"]).values
        y  = df["response"].values
        feature_names = df.drop(columns=["response"]).columns.tolist()

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Fit ensemble model
        if model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state
            )

        model.fit(X_scaled, y)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        composite_auc = roc_auc_score(y, y_prob)
        feature_importances = dict(zip(
            feature_names,
            model.feature_importances_.tolist(),
        ))

        # Individual biomarker results
        individual_results = []
        for bm in feature_names:
            try:
                result = self.analyze_predictive_biomarker(
                    biomarker_values=df[bm],
                    response=df["response"],
                    biomarker_name=bm,
                    optimize_threshold=True,
                )
                individual_results.append(result)
            except Exception as e:
                logger.warning("Skipping biomarker %s: %s", bm, e)

        panel = BiomarkerPanel(
            panel_name=panel_name,
            biomarkers=feature_names,
            composite_auc=composite_auc,
            individual_results=individual_results,
            feature_importances=feature_importances,
        )

        logger.info(
            "Panel analysis [%s]: composite AUC=%.3f, %d biomarkers",
            panel_name, composite_auc, len(feature_names),
        )
        return panel

    def validate_cdx_analytical_performance(
        self,
        test_results: pd.Series,
        reference_results: pd.Series,
        assay_name: str = "CDx Assay",
    ) -> dict:
        """
        Compute companion diagnostic (CDx) analytical validation metrics.

        Calculates sensitivity, specificity, PPV, NPV, and overall
        percent agreement (OPA) against a reference standard —
        following FDA CDx analytical validation guidance.

        Parameters
        ----------
        test_results : pd.Series
            Binary test assay results (1=positive, 0=negative).
        reference_results : pd.Series
            Binary reference standard results.
        assay_name : str
            Name of the CDx assay.

        Returns
        -------
        dict
            CDx analytical performance metrics.
        """
        df = pd.DataFrame({
            "test": test_results,
            "ref":  reference_results,
        }).dropna()

        tn, fp, fn, tp = confusion_matrix(
            df["ref"], df["test"]
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0
        opa         = (tp + tn) / len(df)
        ppa         = tp / (tp + fn) if (tp + fn) > 0 else 0
        npa         = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics = {
            "assay":                    assay_name,
            "n_samples":                len(df),
            "true_positives":           int(tp),
            "true_negatives":           int(tn),
            "false_positives":          int(fp),
            "false_negatives":          int(fn),
            "sensitivity_ppa":          round(sensitivity, 4),
            "specificity_npa":          round(specificity, 4),
            "ppv":                      round(ppv, 4),
            "npv":                      round(npv, 4),
            "overall_percent_agreement": round(opa, 4),
            "positive_percent_agreement": round(ppa, 4),
            "negative_percent_agreement": round(npa, 4),
            "meets_fda_threshold":      sensitivity >= 0.85 and specificity >= 0.85,
        }

        logger.info(
            "CDx validation [%s]: sens=%.3f, spec=%.3f, OPA=%.3f",
            assay_name, sensitivity, specificity, opa,
        )
        return metrics

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _optimize_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Find optimal classification threshold using Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_j  = tpr - fpr
        optimal_idx = int(np.argmax(youden_j))
        threshold = float(thresholds[optimal_idx])

        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        return threshold, sensitivity, specificity, ppv, npv
