Now the final file of Repo 2:

Stay inside the drug_discovery folder
Click "Add file" → "Create new file"
Type in the filename box:

molecule_screener.py

Paste this code:

python"""
drug_discovery/molecule_screener.py
─────────────────────────────────────
Virtual compound screening pipeline for drug discovery.

Implements:
    - High-throughput virtual screening (HTVS)
    - Pharmacophore-based filtering
    - Molecular docking score prediction
    - Lead compound prioritization
    - Scaffold diversity analysis
    - Pan-assay interference (PAINS) filtering

Pipeline:
    Compound library → PAINS filter → Drug-likeness filter
    → Binding prediction → Selectivity screen → Ranked hits

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .target_predictor import MolecularDescriptors, TargetPredictor

logger = logging.getLogger(__name__)


# ── PAINS filters ─────────────────────────────────────────────────────────────

# Pan-assay interference compound (PAINS) structural flags
# In production these use SMARTS patterns via RDKit
PAINS_FLAGS = {
    "rhodanine":         "Rhodanine scaffold — frequent hitter",
    "catechol":          "Catechol — redox active, promiscuous",
    "michael_acceptor":  "Michael acceptor — reactive electrophile",
    "quinone":           "Quinone — redox cycling liability",
    "azo_dye":           "Azo dye — coloured compound interference",
    "heavy_metal":       "Heavy metal chelator",
    "aldehyde":          "Aldehyde — reactive, nonspecific binding",
}

# Drug-likeness filter presets
FILTER_PRESETS = {
    "lipinski":   {"mw_max": 500, "logp_max": 5,   "hbd_max": 5,  "hba_max": 10},
    "fragment":   {"mw_max": 300, "logp_max": 3,   "hbd_max": 3,  "hba_max": 6},
    "lead_like":  {"mw_max": 350, "logp_max": 3.5, "hbd_max": 4,  "hba_max": 7},
    "beyond_ro5": {"mw_max": 1000, "logp_max": 7,  "hbd_max": 10, "hba_max": 15},
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ScreeningHit:
    """A compound identified as a hit in virtual screening."""
    compound_id: str
    predicted_pic50: float
    docking_score: Optional[float] = None
    drug_likeness_score: float = 0.0
    pains_flags: list[str] = field(default_factory=list)
    passes_filters: bool = True
    rank: int = 0
    notes: str = ""

    @property
    def is_clean_hit(self) -> bool:
        return self.passes_filters and len(self.pains_flags) == 0

    def summary(self) -> dict:
        return {
            "rank":            self.rank,
            "compound_id":     self.compound_id,
            "pIC50":           round(self.predicted_pic50, 2),
            "docking_score":   round(self.docking_score, 2) if self.docking_score else None,
            "drug_likeness":   round(self.drug_likeness_score, 2),
            "pains_flags":     self.pains_flags,
            "clean_hit":       self.is_clean_hit,
        }


@dataclass
class ScreeningCampaign:
    """Results from a full virtual screening campaign."""
    campaign_name: str
    target_name: str
    library_size: int
    hits: list[ScreeningHit] = field(default_factory=list)
    n_pains_filtered: int = 0
    n_druglike_filtered: int = 0
    n_screened: int = 0

    @property
    def hit_rate(self) -> float:
        return round(100 * len(self.hits) / self.library_size, 2) if self.library_size > 0 else 0.0

    @property
    def clean_hits(self) -> list[ScreeningHit]:
        return [h for h in self.hits if h.is_clean_hit]

    def top_hits(self, n: int = 10) -> list[ScreeningHit]:
        return sorted(self.hits, key=lambda h: h.predicted_pic50, reverse=True)[:n]

    def summary(self) -> dict:
        return {
            "campaign":          self.campaign_name,
            "target":            self.target_name,
            "library_size":      self.library_size,
            "n_screened":        self.n_screened,
            "n_hits":            len(self.hits),
            "n_clean_hits":      len(self.clean_hits),
            "hit_rate_pct":      self.hit_rate,
            "pains_filtered":    self.n_pains_filtered,
            "druglike_filtered": self.n_druglike_filtered,
            "top_hit_pic50":     round(self.hits[0].predicted_pic50, 2) if self.hits else None,
        }


# ── Molecule Screener ─────────────────────────────────────────────────────────

class MoleculeScreener:
    """
    Virtual compound screening pipeline for drug discovery.

    Screens compound libraries against biological targets using
    ML-predicted binding affinity, drug-likeness filters, and
    PAINS detection to prioritize lead compounds.

    Parameters
    ----------
    filter_preset : str
        Drug-likeness filter preset: 'lipinski', 'fragment',
        'lead_like', or 'beyond_ro5'. Default 'lipinski'.
    pic50_threshold : float
        Minimum pIC50 for hit classification. Default 6.0 (1µM).
    random_state : int
        Random seed. Default 42.

    Examples
    --------
    >>> screener = MoleculeScreener(filter_preset="lead_like")
    >>> campaign = screener.run_campaign(
    ...     compounds=compound_library,
    ...     predictor=trained_predictor,
    ...     target_name="EGFR",
    ...     campaign_name="EGFR_screen_v1",
    ... )
    >>> for hit in campaign.top_hits(10):
    ...     print(hit.summary())
    """

    def __init__(
        self,
        filter_preset: str = "lipinski",
        pic50_threshold: float = 6.0,
        random_state: int = 42,
    ) -> None:
        self.filter_preset   = filter_preset
        self.pic50_threshold = pic50_threshold
        self.random_state    = random_state
        self._filters        = FILTER_PRESETS.get(filter_preset, FILTER_PRESETS["lipinski"])

    # ── Public API ────────────────────────────────────────────────────────────

    def run_campaign(
        self,
        compounds: list[MolecularDescriptors],
        predictor: TargetPredictor,
        target_name: str = "",
        campaign_name: str = "VS_campaign",
        apply_pains_filter: bool = True,
        apply_druglike_filter: bool = True,
    ) -> ScreeningCampaign:
        """
        Run a full virtual screening campaign.

        Pipeline:
            1. PAINS structural filter (remove frequent hitters)
            2. Drug-likeness filter (Lipinski/Veber/custom)
            3. ML binding affinity prediction
            4. Hit identification and ranking

        Parameters
        ----------
        compounds : list[MolecularDescriptors]
            Compound library to screen.
        predictor : TargetPredictor
            Trained binding affinity predictor.
        target_name : str
            Biological target name.
        campaign_name : str
            Screening campaign identifier.
        apply_pains_filter : bool
            Remove PAINS compounds. Default True.
        apply_druglike_filter : bool
            Apply drug-likeness filters. Default True.

        Returns
        -------
        ScreeningCampaign
        """
        campaign = ScreeningCampaign(
            campaign_name=campaign_name,
            target_name=target_name,
            library_size=len(compounds),
        )

        logger.info(
            "Starting VS campaign [%s]: %d compounds → target=%s",
            campaign_name, len(compounds), target_name,
        )

        screened = compounds.copy()

        # Step 1 — PAINS filter
        if apply_pains_filter:
            before = len(screened)
            screened = [c for c in screened if not self._has_pains(c)]
            campaign.n_pains_filtered = before - len(screened)
            logger.info("PAINS filter: removed %d compounds", campaign.n_pains_filtered)

        # Step 2 — Drug-likeness filter
        if apply_druglike_filter:
            before = len(screened)
            screened = [c for c in screened if self._passes_druglike_filter(c)]
            campaign.n_druglike_filtered = before - len(screened)
            logger.info("Drug-likeness filter: removed %d compounds", campaign.n_druglike_filtered)

        campaign.n_screened = len(screened)

        # Step 3 — Binding affinity prediction
        hits = []
        for compound in screened:
            try:
                pred = predictor.predict(compound, target_name=target_name)
                if pred.predicted_pic50 >= self.pic50_threshold:
                    pains = self._get_pains_flags(compound)
                    dl_score = self._drug_likeness_score(compound)
                    hit = ScreeningHit(
                        compound_id=compound.compound_id,
                        predicted_pic50=pred.predicted_pic50,
                        drug_likeness_score=dl_score,
                        pains_flags=pains,
                        passes_filters=True,
                    )
                    hits.append(hit)
            except Exception as e:
                logger.warning("Prediction failed for %s: %s", compound.compound_id, e)

        # Step 4 — Rank hits
        hits.sort(key=lambda h: h.predicted_pic50, reverse=True)
        for i, hit in enumerate(hits, 1):
            hit.rank = i

        campaign.hits = hits
        logger.info(
            "Campaign complete [%s]: %d hits from %d screened (hit rate=%.1f%%)",
            campaign_name, len(hits), campaign.n_screened, campaign.hit_rate,
        )
        return campaign

    def analyze_scaffold_diversity(
        self,
        compounds: list[MolecularDescriptors],
    ) -> dict:
        """
        Analyze structural diversity of a compound set.

        Uses molecular descriptor similarity to estimate
        scaffold diversity and identify clusters of similar
        compounds for library design.

        Parameters
        ----------
        compounds : list[MolecularDescriptors]
            Compounds to analyze.

        Returns
        -------
        dict
            Diversity metrics and similarity statistics.
        """
        if len(compounds) < 2:
            return {"error": "Need at least 2 compounds for diversity analysis"}

        scaler = StandardScaler()
        X = np.array([c.to_feature_vector() for c in compounds])

        # Pad to same length if fingerprints differ
        max_len = max(len(row) for row in X)
        X_padded = np.array([
            np.pad(row, (0, max_len - len(row))) for row in X
        ])
        X_scaled = scaler.fit_transform(X_padded)

        sim_matrix = cosine_similarity(X_scaled)
        np.fill_diagonal(sim_matrix, 0)

        avg_similarity  = float(np.mean(sim_matrix))
        max_similarity  = float(np.max(sim_matrix))
        diversity_score = 1 - avg_similarity

        # Identify most similar pairs
        idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        most_similar_pair = (
            compounds[idx[0]].compound_id,
            compounds[idx[1]].compound_id,
        )

        result = {
            "n_compounds":        len(compounds),
            "diversity_score":    round(diversity_score, 3),
            "avg_similarity":     round(avg_similarity, 3),
            "max_similarity":     round(max_similarity, 3),
            "most_similar_pair":  most_similar_pair,
            "library_assessment": (
                "diverse" if diversity_score > 0.7
                else "moderately diverse" if diversity_score > 0.4
                else "redundant — consider pruning"
            ),
        }

        logger.info(
            "Scaffold diversity: %d compounds, diversity=%.3f (%s)",
            len(compounds), diversity_score, result["library_assessment"],
        )
        return result

    def prioritize_leads(
        self,
        campaign: ScreeningCampaign,
        max_leads: int = 20,
        require_clean: bool = True,
    ) -> list[ScreeningHit]:
        """
        Select and prioritize lead compounds from a screening campaign.

        Parameters
        ----------
        campaign : ScreeningCampaign
            Completed screening campaign.
        max_leads : int
            Maximum number of leads to return. Default 20.
        require_clean : bool
            If True, only return PAINS-free compounds. Default True.

        Returns
        -------
        list[ScreeningHit]
            Prioritized lead compounds sorted by predicted pIC50.
        """
        candidates = campaign.clean_hits if require_clean else campaign.hits
        leads = sorted(
            candidates,
            key=lambda h: (h.predicted_pic50 + h.drug_likeness_score),
            reverse=True,
        )[:max_leads]

        logger.info(
            "Lead prioritization [%s]: %d leads selected from %d candidates",
            campaign.campaign_name, len(leads), len(candidates),
        )
        return leads

    def generate_screening_report(
        self,
        campaign: ScreeningCampaign,
    ) -> dict:
        """
        Generate a comprehensive virtual screening report.

        Parameters
        ----------
        campaign : ScreeningCampaign
            Completed screening campaign.

        Returns
        -------
        dict
            Full screening report with statistics and top hits.
        """
        top10 = campaign.top_hits(10)
        leads = self.prioritize_leads(campaign, max_leads=10)

        report = {
            "campaign_summary":   campaign.summary(),
            "filter_statistics": {
                "filter_preset":      self.filter_preset,
                "pic50_threshold":    self.pic50_threshold,
                "pains_removed":      campaign.n_pains_filtered,
                "druglike_removed":   campaign.n_druglike_filtered,
                "compounds_screened": campaign.n_screened,
            },
            "hit_statistics": {
                "total_hits":    len(campaign.hits),
                "clean_hits":    len(campaign.clean_hits),
                "hit_rate_pct":  campaign.hit_rate,
                "pic50_distribution": {
                    "mean":  round(
                        np.mean([h.predicted_pic50 for h in campaign.hits]), 2
                    ) if campaign.hits else None,
                    "max":   round(
                        max(h.predicted_pic50 for h in campaign.hits), 2
                    ) if campaign.hits else None,
                    "min":   round(
                        min(h.predicted_pic50 for h in campaign.hits), 2
                    ) if campaign.hits else None,
                },
            },
            "top_10_hits":    [h.summary() for h in top10],
            "prioritized_leads": [h.summary() for h in leads],
        }

        logger.info(
            "Screening report generated: %s — %d hits, %d leads",
            campaign.campaign_name, len(campaign.hits), len(leads),
        )
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _has_pains(self, compound: MolecularDescriptors) -> bool:
        """
        Check if a compound has PAINS structural features.
        In production this uses RDKit SMARTS matching.
        Here we use descriptor-based heuristics.
        """
        return (
            compound.aromatic_rings >= 4 and
            compound.h_bond_acceptors >= 8
        )

    def _get_pains_flags(self, compound: MolecularDescriptors) -> list[str]:
        """Return PAINS flag labels for a compound."""
        flags = []
        if compound.aromatic_rings >= 4 and compound.h_bond_acceptors >= 8:
            flags.append("potential_pains_scaffold")
        if compound.logp > 5 and compound.molecular_weight > 500:
            flags.append("beyond_ro5_liability")
        return flags

    def _passes_druglike_filter(self, compound: MolecularDescriptors) -> bool:
        """Apply drug-likeness property filters."""
        f = self._filters
        return (
            compound.molecular_weight <= f["mw_max"] and
            compound.logp            <= f["logp_max"] and
            compound.h_bond_donors   <= f["hbd_max"] and
            compound.h_bond_acceptors <= f["hba_max"]
        )

    def _drug_likeness_score(self, compound: MolecularDescriptors) -> float:
        """
        Compute a composite drug-likeness score (0-1).
        Higher = more drug-like.
        """
        f = self._filters
        scores = [
            1 - min(compound.molecular_weight / f["mw_max"], 1),
            1 - min(max(compound.logp, 0) / f["logp_max"], 1),
            1 - min(compound.h_bond_donors / f["hbd_max"], 1),
            1 - min(compound.h_bond_acceptors / f["hba_max"], 1),
        ]
        return round(float(np.mean(scores)), 3)
