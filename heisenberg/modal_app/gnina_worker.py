"""
GNINA docking worker — Modal serverless CPU deployment.

Replaces the old GPU-based docking worker. Docks each ligand (given as a
SMILES string) against a protein receptor (given as a PDB file) using
GNINA's CPU-only CNN scoring (`--cnn fast`, `--no_gpu`), and returns the same
{name, confidence_score, confidence_raw, all_poses} shape the old handler
produced, so lib/gnina.ts and agent3.py only need to change *where* they
send the request, not how they parse the response.

Deploy:
    modal deploy modal_app/gnina_worker.py

This prints a URL like:
    https://<workspace>--gnina-worker-dock.modal.run
Set that as GNINA_ENDPOINT_URL in .env (and wherever the Next.js app / agent3.py run).

Binding site: the box is determined by a 3-tier fallback, each tier trying
the next only if the previous finds nothing:
    1. Co-crystallized ligand — parse HETATM records already in the
       receptor PDB, filter out crystallization artifacts (waters, ions,
       cryoprotectants, buffers, sugars, common cofactors), box the
       largest qualifying residue.
    2. Pocket prediction — run fpocket on the receptor and box its
       top-ranked (by druggability score) predicted pocket.
    3. Blind docking — a padded box over the whole receptor, same as the
       original always-blind behavior. Used only when both above fail.
"""

import base64
import gzip
import re
import subprocess
import tempfile
import time
from pathlib import Path

import modal
from pydantic import BaseModel

BOX_PADDING_ANGSTROM = 10.0
EXHAUSTIVENESS = 8
GNINA_TIMEOUT_SECONDS = 180
FPOCKET_TIMEOUT_SECONDS = 60

LIGAND_BOX_PADDING = 4.0   # GNINA/Vina's own --autobox_add default
POCKET_BOX_PADDING = 4.0
MIN_BOX_SIZE = 15.0        # floor per axis; a real binding pocket is rarely smaller
MIN_LIGAND_ATOMS = 6       # heavy-atom floor to count a HETATM group as "drug-like"

# Crystallization artifacts to exclude when looking for a real co-crystallized
# ligand: waters, ions, cryoprotectants/solvents, buffers, common sugars,
# common cofactors. Anything else with enough atoms is treated as a candidate.
ARTIFACT_RESNAMES = {
    "HOH", "WAT", "DOD",
    "SO4", "PO4", "NO3", "CL", "BR", "IOD", "FLC",
    "GOL", "EDO", "PEG", "PGE", "DMS", "MPD", "BME",
    "MES", "HED", "TRS", "EPE", "CIT", "ACT", "FMT",
    "NAG", "MAN", "BMA", "FUC", "GAL", "SIA", "BGC",
    "ZN", "MG", "CA", "FE", "CU", "MN", "CO", "NI", "NA", "K",
    "GDP", "GTP", "ADP", "ATP", "NAD", "FAD", "COA",
    "UNX", "UNL",
}

# See modal_app/Dockerfile for how the gnina binary + its CUDA/cuDNN runtime
# libs get into the image (no GPU, no driver, no CUDA base image involved --
# validated by docking gnina's own 184L test ligand with zero GPU present).
image = modal.Image.from_dockerfile("modal_app/Dockerfile").pip_install(
    "rdkit", "fastapi[standard]"
)

app = modal.App("gnina-worker", image=image)


class Ligand(BaseModel):
    name: str
    smiles: str


class DockRequest(BaseModel):
    protein_pdb_b64: str
    ligands: list[Ligand]
    samples_per_complex: int = 10


def _box_from_coords(
    xs: list[float], ys: list[float], zs: list[float], padding: float
) -> dict:
    """Centroid + padded bounding box from a set of atom coordinates, with a
    floor on each dimension so a small/planar group doesn't produce a box too
    tight for GNINA's search to be meaningful."""
    size_x = max((max(xs) - min(xs)) + 2 * padding, MIN_BOX_SIZE)
    size_y = max((max(ys) - min(ys)) + 2 * padding, MIN_BOX_SIZE)
    size_z = max((max(zs) - min(zs)) + 2 * padding, MIN_BOX_SIZE)
    return {
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
        "center_z": (min(zs) + max(zs)) / 2,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
    }


def _parse_pdb_atoms(pdb_path: Path, record_types: tuple[str, ...]) -> list[dict]:
    """Parse fixed-width PDB ATOM/HETATM lines into dicts with resName,
    chainID, resSeq, altLoc, and x/y/z. Shared by every tier below."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(record_types):
                continue
            alt_loc = line[16].strip()
            if alt_loc not in ("", "A"):
                continue
            atoms.append({
                "res_name": line[17:20].strip(),
                "chain_id": line[21].strip(),
                "res_seq": line[22:26].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
            })
    return atoms


def _blind_box(pdb_path: Path) -> dict:
    """Bounding box (+padding) over every receptor atom, for blind docking.
    Tier 3: used only when no co-crystallized ligand or predicted pocket is
    available."""
    atoms = _parse_pdb_atoms(pdb_path, ("ATOM", "HETATM"))
    if not atoms:
        raise ValueError("No ATOM/HETATM records found in receptor PDB")

    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    zs = [a["z"] for a in atoms]
    box = _box_from_coords(xs, ys, zs, BOX_PADDING_ANGSTROM)
    box["method"] = "blind_docking"
    return box


def _modified_residue_names(pdb_path: Path) -> set[str]:
    """Residue names declared by MODRES records -- modified standard amino/
    nucleic acids (e.g. M3L = trimethyllysine, MSE = selenomethionine) that
    are covalently part of the polymer chain, not free ligands, even though
    they show up as HETATM. Parsed generically instead of hardcoding every
    known PTM code."""
    names = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODRES"):
                names.add(line[12:15].strip())
    return names


def _ligand_box(pdb_path: Path) -> dict | None:
    """Tier 1: box the largest co-crystallized ligand in the receptor PDB,
    excluding known crystallization artifacts and modified-residue chain
    members. Returns None if nothing qualifies (signals Tier 2 should be
    tried)."""
    atoms = _parse_pdb_atoms(pdb_path, ("HETATM",))
    excluded = ARTIFACT_RESNAMES | _modified_residue_names(pdb_path)

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for a in atoms:
        if a["res_name"] in excluded:
            continue
        key = (a["chain_id"], a["res_seq"], a["res_name"])
        groups.setdefault(key, []).append(a)

    candidates = [g for g in groups.values() if len(g) >= MIN_LIGAND_ATOMS]
    if not candidates:
        return None

    largest = max(candidates, key=len)
    xs = [a["x"] for a in largest]
    ys = [a["y"] for a in largest]
    zs = [a["z"] for a in largest]
    box = _box_from_coords(xs, ys, zs, LIGAND_BOX_PADDING)
    box["method"] = "cocrystallized_ligand"
    box["ligand_name"] = largest[0]["res_name"]
    return box


def _pocket_box(pdb_path: Path) -> dict | None:
    """Tier 2: run fpocket on the receptor and box its top-ranked (by
    druggability score) predicted pocket. Any failure — binary missing,
    no pockets found, unparseable output — is swallowed and returns None
    so the caller falls through to Tier 3 (blind docking) instead of
    breaking the whole docking request."""
    try:
        out_dir = pdb_path.parent / f"{pdb_path.stem}_out"
        subprocess.run(
            ["fpocket", "-f", str(pdb_path)],
            capture_output=True, text=True, timeout=FPOCKET_TIMEOUT_SECONDS,
            check=True,
        )

        info_path = out_dir / f"{pdb_path.stem}_info.txt"
        info_text = info_path.read_text()

        best_rank = None
        best_score = float("-inf")
        for match in re.finditer(
            r"Pocket\s+(\d+)\s*:\s*\n(.*?)(?=\nPocket\s+\d+\s*:|\Z)",
            info_text, re.DOTALL,
        ):
            rank = int(match.group(1))
            block = match.group(2)
            score_match = re.search(
                r"Druggability Score\s*:\s*([\d.eE+-]+)", block
            ) or re.search(r"^\s*Score\s*:\s*([\d.eE+-]+)", block, re.MULTILINE)
            if not score_match:
                continue
            score = float(score_match.group(1))
            if score > best_score:
                best_score = score
                best_rank = rank

        if best_rank is None:
            return None

        pocket_pdb = out_dir / "pockets" / f"pocket{best_rank}_atm.pdb"
        atoms = _parse_pdb_atoms(pocket_pdb, ("ATOM", "HETATM"))
        if not atoms:
            return None

        xs = [a["x"] for a in atoms]
        ys = [a["y"] for a in atoms]
        zs = [a["z"] for a in atoms]
        box = _box_from_coords(xs, ys, zs, POCKET_BOX_PADDING)
        box["method"] = "pocket_prediction"
        box["pocket_rank"] = best_rank
        return box
    except Exception:
        return None


def _determine_box(pdb_path: Path) -> dict:
    """3-tier fallback: co-crystallized ligand -> predicted pocket -> blind."""
    box = _ligand_box(pdb_path)
    if box:
        return box
    box = _pocket_box(pdb_path)
    if box:
        return box
    return _blind_box(pdb_path)


def _embed_ligand(smiles: str, sdf_path: Path) -> None:
    """SMILES -> 3D conformer (RDKit ETKDGv3 + MMFF/UFF), written as SDF."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("RDKit conformer embedding failed")

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()


def _dock_one(receptor_pdb: Path, box: dict, name: str, smiles: str, num_modes: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        lig_sdf = tmp / "ligand.sdf"
        out_sdf = tmp / "out.sdf.gz"

        _embed_ligand(smiles, lig_sdf)

        cmd = [
            "gnina",
            "-r", str(receptor_pdb),
            "-l", str(lig_sdf),
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(EXHAUSTIVENESS),
            "--num_modes", str(max(1, min(num_modes, 20))),
            "--cnn", "fast",
            "--cnn_scoring", "rescore",
            "--no_gpu",
            "--seed", "42",
            "-o", str(out_sdf),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=GNINA_TIMEOUT_SECONDS
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gnina failed for {name}: {proc.stderr[-2000:]}")

        from rdkit import Chem

        poses = []
        with gzip.open(out_sdf) as gz:
            for mol in Chem.ForwardSDMolSupplier(gz):
                if mol is None:
                    continue
                props = mol.GetPropsAsDict()
                poses.append({
                    "cnn_score": float(props.get("CNNscore", 0.0)),
                    "cnn_affinity": float(props.get("CNNaffinity", 0.0)),
                    "vina_affinity": float(props.get("minimizedAffinity", 0.0)),
                })

        if not poses:
            raise RuntimeError(f"gnina produced no poses for {name}")

        poses.sort(key=lambda p: p["cnn_score"], reverse=True)
        best = poses[0]
        return {
            "name": name,
            "confidence_score": best["cnn_score"],
            "confidence_raw": best["vina_affinity"],
            "all_poses": poses,
        }


@app.function(cpu=2, memory=4096, timeout=600)
@modal.fastapi_endpoint(method="POST")
def dock(req: DockRequest) -> dict:
    start = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        receptor_pdb = Path(tmp_dir) / "receptor.pdb"
        receptor_pdb.write_bytes(base64.b64decode(req.protein_pdb_b64))
        box = _determine_box(receptor_pdb)

        results = []
        errors = []
        for lig in req.ligands:
            try:
                results.append(
                    _dock_one(receptor_pdb, box, lig.name, lig.smiles, req.samples_per_complex)
                )
            except Exception as e:
                errors.append({"name": lig.name, "error": str(e)})

    return {
        "results": results,
        "errors": errors,
        "processing_time_seconds": round(time.time() - start, 2),
        "binding_site_method": box.get("method"),
    }
