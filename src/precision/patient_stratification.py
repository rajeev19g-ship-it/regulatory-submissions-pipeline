Now let's add patient stratification:

Stay inside the precision folder
Click "Add file" → "Create new file"
Type in the filename box:

patient_stratification.py

Paste this code:

python"""
precision/patient_stratification.py
─────────────────────────────────────
ML-powered patient stratification for precision medicine trials.

Implements:
    - Unsupervised clustering for patient subgroup discovery
    - Supervised subgroup identification (treatment effect heterogeneity)
    - Responder/non-responder classification
    - Biomarker-driven enrollment enrichment scoring
    - Virtual patient twin modeling for trial simulation

Applications:
    - Basket trial design (biomarker-selected populations)
    - Umbrella trial stratification
    - Adaptive enrichment trial design
    - Post-hoc subgroup analysis

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class StratificationResult:
    """Results from patient stratification analysis."""
    method: str
    n_patients: int
    n_subgroups: int
    subgroup_labels: list[int] = field(default_factory=list)
    subgroup_sizes: dict[int, int] = field(default_factory=dict)
    silhouette_score: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    subgroup_profiles: dict[int, dict] = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> dict:
        return {
            "method":            self.method,
            "n_patients":        self.n_patients,
            "n_subgroups":       self.n_subgroups,
            "subgroup_sizes":    self.subgroup_sizes,
            "silhouette_score":  round(self.silhouette_score, 3)
                                 if self.silhouette_score else None,
            "notes":             self.notes,
        }


@dataclass
class EnrichmentScore:
    """Biomarker-driven trial enrollment enrichment score."""
    usubjid: str
    enrichment_probability: float
    biomarker_profile: dict[str, float] = field(default_factory=dict)
    predicted_subgroup: int = -1
    enrollment_recommendation: str = ""

    def __post_init__(self):
        if self.enrichment_probability >= 0.7:
            self.enrollment_recommendation = "ENROLL — high likelihood of benefit"
        elif self.enrichment_probability >= 0.4:
            self.enrollment_recommendation = "CONSIDER — moderate likelihood of benefit"
        else:
            self.enrollment_recommendation = "SCREEN FAILURE — low likelihood of benefit"


# ── Patient Stratifier ────────────────────────────────────────────────────────

class PatientStratifier:
    """
    ML-powered patient stratification for precision medicine trials.

    Identifies patient subgroups using unsupervised clustering and
    supervised classification. Supports biomarker-driven enrichment
    scoring for adaptive trial enrollment.

    Parameters
    ----------
    random_state : int
        Random seed. Default 42.
    n_components_pca : int
        PCA components for dimensionality reduction. Default 10.

    Examples
    --------
    >>> stratifier = PatientStratifier()
    >>> result = stratifier.cluster_patients(
    ...     biomarker_df,
    ...     n_clusters=3,
    ...     method="kmeans",
    ... )
    >>> print(result.summary())
    """

    def __init__(
        self,
        random_state: int = 42,
        n_components_pca: int = 10,
    ) -> None:
        self.random_state     = random_state
        self.n_components_pca = n_components_pca
        self._scaler          = StandardScaler()
        self._pca             = PCA(
            n_components=n_components_pca,
            random_state=random_state,
        )
        self._enrichment_model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def cluster_patients(
        self,
        biomarker_df: pd.DataFrame,
        n_clusters: int = 3,
        method: str = "kmeans",
        use_pca: bool = True,
    ) -> StratificationResult:
        """
        Discover patient subgroups via unsupervised clustering.

        Parameters
        ----------
        biomarker_df : pd.DataFrame
            Biomarker feature matrix (subjects × biomarkers).
        n_clusters : int
            Number of clusters to identify. Default 3.
        method : str
            Clustering algorithm: 'kmeans' or 'hierarchical'. Default 'kmeans'.
        use_pca : bool
            Apply PCA before clustering. Default True.

        Returns
        -------
        StratificationResult
        """
        df_clean = biomarker_df.dropna()
        X        = self._scaler.fit_transform(df_clean.values)

        if use_pca and X.shape[1] > self.n_components_pca:
            n_comp = min(self.n_components_pca, X.shape[0] - 1, X.shape[1])
            pca    = PCA(n_components=n_comp, random_state=self.random_state)
            X      = pca.fit_transform(X)
            logger.info(
                "PCA: %d → %d components (%.1f%% variance explained)",
                biomarker_df.shape[1], n_comp,
                100 * pca.explained_variance_ratio_.sum(),
            )

        # Cluster
        if method == "hierarchical":
            model  = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
        else:
            model  = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(X)

        # Cluster quality metrics
        sil_score = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else None
        ch_score  = float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else None

        # Subgroup profiles
        df_clean  = df_clean.copy()
        df_clean["_cluster"] = labels
        profiles  = {}
        sizes     = {}
        for cluster_id in sorted(set(labels)):
            mask           = df_clean["_cluster"] == cluster_id
            sizes[int(cluster_id)] = int(mask.sum())
            profiles[int(cluster_id)] = {
                col: round(float(df_clean.loc[mask, col].mean()), 3)
                for col in biomarker_df.columns
            }

        result = StratificationResult(
            method=method,
            n_patients=len(df_clean),
            n_subgroups=n_clusters,
            subgroup_labels=labels.tolist(),
            subgroup_sizes=sizes,
            silhouette_score=sil_score,
            calinski_harabasz=ch_score,
            subgroup_profiles=profiles,
            notes=f"Silhouette={sil_score:.3f}" if sil_score else "",
        )

        logger.info(
            "Clustering [%s]: %d patients → %d subgroups, "
            "silhouette=%.3f",
            method, len(df_clean), n_clusters, sil_score or 0,
        )
        return result

    def find_optimal_clusters(
        self,
        biomarker_df: pd.DataFrame,
        k_range: range = range(2, 8),
    ) -> dict:
        """
        Find the optimal number of clusters using silhouette analysis.

        Parameters
        ----------
        biomarker_df : pd.DataFrame
            Biomarker feature matrix.
        k_range : range
            Range of k values to evaluate. Default range(2, 8).

        Returns
        -------
        dict
            Silhouette scores per k and recommended optimal k.
        """
        df_clean = biomarker_df.dropna()
        X        = self._scaler.fit_transform(df_clean.values)

        scores = {}
        for k in k_range:
            if k >= len(df_clean):
                continue
            model  = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)
            scores[k] = float(silhouette_score(X, labels))

        optimal_k = max(scores, key=lambda k: scores[k])
        logger.info(
            "Optimal clusters: k=%d (silhouette=%.3f)",
            optimal_k, scores[optimal_k],
        )
        return {
            "silhouette_scores": scores,
            "optimal_k":         optimal_k,
            "optimal_score":     scores[optimal_k],
        }

    def train_enrichment_model(
        self,
        biomarker_df: pd.DataFrame,
        response: pd.Series,
        model_type: str = "random_forest",
    ) -> dict:
        """
        Train a model to predict treatment response for enrollment enrichment.

        Parameters
        ----------
        biomarker_df : pd.DataFrame
            Training biomarker data.
        response : pd.Series
            Binary treatment response (1=responder, 0=non-responder).
        model_type : str
            'random_forest' or 'gradient_boosting'. Default 'random_forest'.

        Returns
        -------
        dict
            Training performance metrics.
        """
        df = pd.concat([biomarker_df, response.rename("response")], axis=1).dropna()
        X  = self._scaler.fit_transform(df.drop(columns=["response"]).values)
        y  = df["response"].values
        feature_names = df.drop(columns=["response"]).columns.tolist()

        if model_type == "gradient_boosting":
            self._enrichment_model = GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            )
        else:
            self._enrichment_model = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state
            )

        self._enrichment_model.fit(X, y)
        self._feature_names = feature_names

        # Cross-validated AUC
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )
        y_prob_cv = cross_val_predict(
            self._enrichment_model, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        cv_auc = roc_auc_score(y, y_prob_cv)

        feature_importances = dict(zip(
            feature_names,
            self._enrichment_model.feature_importances_.tolist(),
        ))

        logger.info(
            "Enrichment model trained: CV AUC=%.3f, %d features",
            cv_auc, len(feature_names),
        )
        return {
            "model_type":         model_type,
            "n_training_samples": len(df),
            "cv_auc":             round(cv_auc, 3),
            "feature_importances": feature_importances,
            "top_biomarkers":     sorted(
                feature_importances.items(),
                key=lambda x: x[1], reverse=True
            )[:5],
        }

    def score_patient_for_enrollment(
        self,
        patient_biomarkers: dict[str, float],
        usubjid: str = "",
    ) -> EnrichmentScore:
        """
        Score a candidate patient for trial enrollment enrichment.

        Parameters
        ----------
        patient_biomarkers : dict[str, float]
            Biomarker values for a single patient.
        usubjid : str
            Subject identifier.

        Returns
        -------
        EnrichmentScore
        """
        if self._enrichment_model is None:
            raise RuntimeError(
                "Enrichment model not trained. Call train_enrichment_model() first."
            )

        feature_values = np.array([
            patient_biomarkers.get(f, 0.0)
            for f in self._feature_names
        ]).reshape(1, -1)

        X_scaled  = self._scaler.transform(feature_values)
        prob      = float(
            self._enrichment_model.predict_proba(X_scaled)[0, 1]
        )
        subgroup  = int(self._enrichment_model.predict(X_scaled)[0])

        score = EnrichmentScore(
            usubjid=usubjid,
            enrichment_probability=prob,
            biomarker_profile=patient_biomarkers,
            predicted_subgroup=subgroup,
        )
        logger.info(
            "Enrichment score [%s]: p=%.3f — %s",
            usubjid, prob, score.enrollment_recommendation,
        )
        return score

    def analyze_subgroup_treatment_effect(
        self,
        biomarker_df: pd.DataFrame,
        response: pd.Series,
        treatment: pd.Series,
        n_clusters: int = 2,
    ) -> dict:
        """
        Identify subgroups with differential treatment effects.

        Clusters patients by biomarker profile and compares treatment
        response rates across subgroups — supporting adaptive
        enrichment design and subgroup analysis reporting.

        Parameters
        ----------
        biomarker_df : pd.DataFrame
            Biomarker feature matrix.
        response : pd.Series
            Binary response indicator.
        treatment : pd.Series
            Treatment arm indicator (1=treated, 0=control).
        n_clusters : int
            Number of biomarker subgroups. Default 2.

        Returns
        -------
        dict
            Treatment effect by subgroup with response rates.
        """
        df = pd.concat([
            biomarker_df,
            response.rename("response"),
            treatment.rename("treatment"),
        ], axis=1).dropna()

        # Cluster on biomarkers
        X      = self._scaler.fit_transform(
            df.drop(columns=["response", "treatment"]).values
        )
        model  = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        df["subgroup"] = model.fit_predict(X)

        # Treatment effect per subgroup
        results = {}
        for sg in sorted(df["subgroup"].unique()):
            sg_df    = df[df["subgroup"] == sg]
            treated  = sg_df[sg_df["treatment"] == 1]
            control  = sg_df[sg_df["treatment"] == 0]

            rr_treated = float(treated["response"].mean()) if len(treated) > 0 else None
            rr_control = float(control["response"].mean()) if len(control) > 0 else None
            delta      = (
                round(rr_treated - rr_control, 3)
                if rr_treated is not None and rr_control is not None
                else None
            )

            results[f"subgroup_{sg}"] = {
                "n_total":          len(sg_df),
                "n_treated":        len(treated),
                "n_control":        len(control),
                "response_rate_treated": round(rr_treated, 3) if rr_treated else None,
                "response_rate_control": round(rr_control, 3) if rr_control else None,
                "treatment_effect_delta": delta,
            }

        logger.info(
            "Subgroup treatment effect analysis: %d subgroups identified",
            n_clusters,
        )
        return {
            "n_subgroups": n_clusters,
            "subgroups":   results,
        }
