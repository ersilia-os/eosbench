"""Scientific one-liner descriptions for every MoleculeNet label column.

Well-characterised sets (BBBP, BACE, HIV, ClinTox, Tox21) use curated text; SIDER columns
are MedDRA System Organ Class names (self-describing); MUV uses the target table from the
MUV publication; ToxCast's 617 endpoints are built from EPA invitroDB's official assay
annotation (intended target family, biological process, signal direction, organism/tissue).

Each lookup returns ``(description, source)``. Used by
``scripts/add_moleculenet_descriptions.py`` to inject a per-column ``description`` into
metadata.json.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

import _prepare_common as common

# EPA invitroDB v3.3 assay annotation (the methods table carries the endpoint -> target map).
TOXCAST_ASSAY_INFO_URL = (
    "https://gaftp.epa.gov/COMPTOX/High_Throughput_Screening_Data/InVitroDB_V3.3/"
    "Assay_Information/Assay_Information_August_2020.zip"
)
TOXCAST_ZIP = common.DATA_ROOT / "_raw" / "toxcast" / "assay_info.zip"
_TOXCAST_METHODS = "assay_methods_invitrodb_v3_3.xlsx"
TOXCAST_SOURCE = "EPA invitroDB v3.3 (ToxCast/Tox21)"


# --- curated, single-/few-endpoint sets -------------------------------------------------
_BBBP = "MoleculeNet BBBP (Martins et al. 2012)"
_BACE = "MoleculeNet BACE (Subramanian et al. 2016)"
_HIV = "MoleculeNet HIV (NCI DTP AIDS antiviral screen)"
_CLINTOX = "MoleculeNet ClinTox (Gayvert et al. 2016)"
_TOX21 = "Tox21 qHTS (NIH/NCATS, EPA, FDA, NIEHS)"

CURATED: dict[str, dict[str, tuple[str, str]]] = {
    "bbbp": {
        "bbbp": (
            "Blood-brain barrier penetration: 1 if the molecule crosses the BBB.",
            _BBBP,
        ),
    },
    "bace": {
        "bace": (
            "Inhibition of human beta-secretase 1 (BACE-1), an Alzheimer's drug target.",
            _BACE,
        ),
    },
    "hiv": {
        "hiv": (
            "Inhibition of HIV replication in the NCI antiviral screen (active vs inactive).",
            _HIV,
        ),
    },
    "clintox": {
        "FDA_APPROVED": (
            "FDA approval status: 1 if the compound is an FDA-approved drug.",
            _CLINTOX,
        ),
        "CT_TOX": (
            "Clinical-trial toxicity: 1 if the drug failed clinical trials for toxicity reasons.",
            _CLINTOX,
        ),
    },
    "tox21": {
        "NR-AR": (
            "Agonism of the androgen receptor (AR / NR3C4), a nuclear hormone receptor.",
            _TOX21,
        ),
        "NR-AR-LBD": (
            "Agonism at the androgen receptor ligand-binding domain (AR-LBD).",
            _TOX21,
        ),
        "NR-AhR": (
            "Activation of the aryl hydrocarbon receptor (AhR), a xenobiotic-sensing transcription factor.",
            _TOX21,
        ),
        "NR-Aromatase": (
            "Inhibition of aromatase (CYP19A1), which converts androgens to estrogens.",
            _TOX21,
        ),
        "NR-ER": (
            "Agonism of the estrogen receptor (ERalpha / ESR1), a nuclear hormone receptor.",
            _TOX21,
        ),
        "NR-ER-LBD": (
            "Agonism at the estrogen receptor ligand-binding domain (ER-LBD).",
            _TOX21,
        ),
        "NR-PPAR-gamma": (
            "Activation of peroxisome proliferator-activated receptor gamma (PPARgamma / NR1C3).",
            _TOX21,
        ),
        "SR-ARE": (
            "Activation of the antioxidant response element (Nrf2/ARE) oxidative-stress pathway.",
            _TOX21,
        ),
        "SR-ATAD5": (
            "Genotoxicity / DNA-damage response reported via ATAD5 stabilisation.",
            _TOX21,
        ),
        "SR-HSE": (
            "Activation of the heat-shock response element (HSE) proteotoxic-stress pathway.",
            _TOX21,
        ),
        "SR-MMP": ("Disruption of the mitochondrial membrane potential (MMP).", _TOX21),
        "SR-p53": (
            "Activation of the p53 DNA-damage / tumour-suppressor stress pathway.",
            _TOX21,
        ),
    },
}

# MUV target table (Rohrer & Baumann 2009, derived from PubChem BioAssay).
_MUV = "Rohrer & Baumann 2009 (MUV); PubChem BioAssay"
MUV: dict[str, str] = {
    "MUV-466": "S1P1 (sphingosine-1-phosphate receptor 1) agonists - GPCR agonism.",
    "MUV-548": "PKA (cAMP-dependent protein kinase A) inhibitors - kinase inhibition.",
    "MUV-600": "SF1 (steroidogenic factor 1, NR5A1) inhibitors - nuclear receptor.",
    "MUV-644": "Rho-associated kinase 2 (ROCK2) inhibitors - kinase inhibition.",
    "MUV-652": "HIV reverse-transcriptase-associated RNase H inhibitors.",
    "MUV-689": "Eph receptor A4 (EphA4) inhibitors - receptor tyrosine kinase.",
    "MUV-692": "SF1 (steroidogenic factor 1, NR5A1) agonists - nuclear receptor.",
    "MUV-712": "Heat-shock protein 90 (HSP90) inhibitors - molecular chaperone.",
    "MUV-713": "Estrogen receptor-alpha coactivator-binding inhibitors.",
    "MUV-733": "Estrogen receptor-beta coactivator-binding inhibitors.",
    "MUV-737": "Estrogen receptor-alpha coactivator-binding potentiators.",
    "MUV-810": "Focal adhesion kinase (FAK / PTK2) inhibitors - kinase inhibition.",
    "MUV-832": "Cathepsin G inhibitors - serine protease.",
    "MUV-846": "Coagulation factor XIa (FXIa) inhibitors - serine protease.",
    "MUV-852": "Coagulation factor XIIa (FXIIa) inhibitors - serine protease.",
    "MUV-858": "Dopamine D1 receptor allosteric modulators - GPCR.",
    "MUV-859": "Muscarinic acetylcholine receptor M1 allosteric agonists - GPCR.",
}


# ToxCast endpoints present in MoleculeNet (invitroDB v2) but renamed in v3.3, so absent from
# the v3.3 annotation table. Descriptions below are grounded in the EPA annotation of the
# corresponding kept endpoints (ACEA_ER_80hr, TOX21_AR_LUC_MDAKB2_*, TOX21_ERa_BLA_*,
# APR_HepG2_* readouts) — same assay family, target, and readout, different name/cell line.
TOXCAST_LEGACY_SOURCE = (
    "ToxCast/Tox21 (EPA invitroDB); MoleculeNet-era (v2) endpoint name"
)


def _apr_hepat(readout: str, hr: int, direction: str) -> str:
    sig = "increased" if direction == "up" else "decreased"
    return (
        f"{readout} - {sig} signal; primary human hepatocytes "
        f"(Apredica high-content imaging), {hr} h."
    )


TOXCAST_LEGACY: dict[str, str] = {
    "ACEA_T47D_80hr_Positive": (
        "Estrogen receptor (steroidal nuclear receptor) agonism via increased T47D "
        "breast-cell proliferation; ACEA xCELLigence real-time assay, 80 h."
    ),
    "ACEA_T47D_80hr_Negative": (
        "Estrogen receptor (steroidal nuclear receptor) antagonism or cytotoxicity via "
        "decreased T47D breast-cell proliferation; ACEA xCELLigence real-time assay, 80 h."
    ),
    "APR_Hepat_Apoptosis_24hr_up": _apr_hepat("Hepatocyte apoptosis", 24, "up"),
    "APR_Hepat_Apoptosis_48hr_up": _apr_hepat("Hepatocyte apoptosis", 48, "up"),
    "APR_Hepat_CellLoss_24hr_dn": _apr_hepat(
        "Hepatocyte cell-loss (viable cell count)", 24, "dn"
    ),
    "APR_Hepat_CellLoss_48hr_dn": _apr_hepat(
        "Hepatocyte cell-loss (viable cell count)", 48, "dn"
    ),
    "APR_Hepat_DNADamage_24hr_up": _apr_hepat("Hepatocyte DNA damage", 24, "up"),
    "APR_Hepat_DNADamage_48hr_up": _apr_hepat("Hepatocyte DNA damage", 48, "up"),
    "APR_Hepat_DNATexture_24hr_up": _apr_hepat(
        "Hepatocyte nuclear DNA texture", 24, "up"
    ),
    "APR_Hepat_DNATexture_48hr_up": _apr_hepat(
        "Hepatocyte nuclear DNA texture", 48, "up"
    ),
    "APR_Hepat_MitoFxnI_1hr_dn": _apr_hepat(
        "Hepatocyte mitochondrial function (membrane potential)", 1, "dn"
    ),
    "APR_Hepat_MitoFxnI_24hr_dn": _apr_hepat(
        "Hepatocyte mitochondrial function (membrane potential)", 24, "dn"
    ),
    "APR_Hepat_MitoFxnI_48hr_dn": _apr_hepat(
        "Hepatocyte mitochondrial function (membrane potential)", 48, "dn"
    ),
    "APR_Hepat_NuclearSize_24hr_dn": _apr_hepat("Hepatocyte nuclear size", 24, "dn"),
    "APR_Hepat_NuclearSize_48hr_dn": _apr_hepat("Hepatocyte nuclear size", 48, "dn"),
    "APR_Hepat_Steatosis_24hr_up": _apr_hepat(
        "Hepatocyte lipid accumulation (steatosis)", 24, "up"
    ),
    "APR_Hepat_Steatosis_48hr_up": _apr_hepat(
        "Hepatocyte lipid accumulation (steatosis)", 48, "up"
    ),
    "TOX21_AR_LUC_MDAKB2_Antagonist": (
        "Antagonism of the androgen receptor (AR / NR3C4); Tox21 luciferase reporter in "
        "MDA-kb2 human breast cells."
    ),
    "TOX21_AR_LUC_MDAKB2_Antagonist2": (
        "Antagonism of the androgen receptor (AR / NR3C4); Tox21 luciferase reporter in "
        "MDA-kb2 human breast cells (confirmatory readout)."
    ),
    "TOX21_ERa_LUC_BG1_Agonist": (
        "Agonism of estrogen receptor alpha (ERalpha / ESR1); Tox21 luciferase reporter in "
        "BG1 human ovarian cells."
    ),
    "TOX21_ERa_LUC_BG1_Antagonist": (
        "Antagonism of estrogen receptor alpha (ERalpha / ESR1); Tox21 luciferase reporter "
        "in BG1 human ovarian cells."
    ),
}


def _sider_description(name: str) -> str:
    return f"Adverse drug reactions in the MedDRA System Organ Class '{name}'."


_SIDER = "MoleculeNet SIDER (MedDRA system organ classes)"


def _download_toxcast_zip() -> Path:
    if not TOXCAST_ZIP.exists():
        TOXCAST_ZIP.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        print(f"  downloading {TOXCAST_ASSAY_INFO_URL}")
        urllib.request.urlretrieve(TOXCAST_ASSAY_INFO_URL, TOXCAST_ZIP)
    return TOXCAST_ZIP


def _toxcast_oneliner(row) -> str:
    """Compose a concise endpoint description from invitroDB structured fields."""

    def s(v):
        return str(v).strip() if pd.notna(v) and str(v).strip() else None

    family = s(row.get("intended_target_family"))
    sub = s(row.get("intended_target_family_sub"))
    bio = s(row.get("biological_process_target"))
    signal = {"gain": "increased", "loss": "decreased"}.get(
        s(row.get("signal_direction")) or "", s(row.get("signal_direction"))
    )
    organism = s(row.get("organism"))
    tissue = s(row.get("tissue"))
    timepoint = row.get("timepoint_hr")

    head = family.capitalize() if family else "Assay endpoint"
    if sub and sub.lower() not in head.lower():
        head += f" ({sub})"

    parts = [head]
    if bio and bio.lower() != (family or "").lower():
        parts.append(bio)
    tail = []
    if signal:
        tail.append(f"{signal} signal")
    loc = " ".join(x for x in (organism, tissue) if x)
    if loc:
        tail.append(loc)
    if pd.notna(timepoint):
        try:
            tail.append(f"{float(timepoint):g} h")
        except (TypeError, ValueError):
            pass

    text = "; ".join(parts)
    if tail:
        text += " - " + "; ".join(tail)
    return text + "."


def _toxcast_map() -> dict[str, str]:
    """Map each ToxCast endpoint name (aenm) to a one-liner from invitroDB."""
    z = zipfile.ZipFile(_download_toxcast_zip())
    with z.open(_TOXCAST_METHODS) as f:
        m = pd.read_excel(f)
    m = m.drop_duplicates("assay_component_endpoint_name").set_index(
        "assay_component_endpoint_name"
    )
    return {aenm: _toxcast_oneliner(row) for aenm, row in m.iterrows()}


def describe_columns(family: str, columns: list[str]) -> dict[str, dict]:
    """Return ``{column: {"description": str|None, "description_source": str}}``."""
    out: dict[str, dict] = {}

    if family in CURATED:
        for col in columns:
            desc, src = CURATED[family].get(col, (None, None))
            out[col] = {"description": desc, "description_source": src}
        return out

    if family == "sider":
        for col in columns:
            out[col] = {
                "description": _sider_description(col),
                "description_source": _SIDER,
            }
        return out

    if family == "muv":
        for col in columns:
            out[col] = {"description": MUV.get(col), "description_source": _MUV}
        return out

    if family == "toxcast":
        tmap = _toxcast_map()
        for col in columns:
            if col in tmap:
                out[col] = {
                    "description": tmap[col],
                    "description_source": TOXCAST_SOURCE,
                }
            elif col in TOXCAST_LEGACY:
                out[col] = {
                    "description": TOXCAST_LEGACY[col],
                    "description_source": TOXCAST_LEGACY_SOURCE,
                }
            else:
                out[col] = {
                    "description": None,
                    "description_source": "not in invitroDB v3.3",
                }
        return out

    # Unknown family: leave descriptions blank.
    return {col: {"description": None, "description_source": None} for col in columns}
