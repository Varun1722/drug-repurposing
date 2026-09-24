# GNINA docking worker (Modal)

Serverless CPU replacement for the old GPU-based docking worker. See
`gnina_worker.py` for the docking logic and API contract, and `Dockerfile`
for how the gnina binary gets into the image.

## How the image works

`gnina_worker.py` builds its image with `modal.Image.from_dockerfile("modal_app/Dockerfile")`.
That Dockerfile:

1. Installs `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cusparse-cu12`,
   `nvidia-cufft-cu12`, `nvidia-cusolver-cu12`, `nvidia-cuda-runtime-cu12`
   from PyPI (the same wheels PyTorch's own `cu12` build depends on) and
   collects just their `.so` files into `/usr/local/lib/cuda-libs`.
2. Downloads gnina's official prebuilt release binary
   (`gnina.cuda12.8.static` — openbabel/boost/libtorch are statically linked
   into it) straight from GitHub Releases.
3. Fetches the prebuilt `fpocket` binary from the conda-forge channel via
   `micromamba` (a single static binary, no Python/conda install of its own)
   and copies just that binary + its resolved runtime libs into the image.
   `fpocket` isn't packaged for Debian/apt, and its own upstream source
   (last touched years ago) fails to compile under GCC 14's default
   strictness — an old `strcpy(&some_array, ...)` pattern that older GCC
   only warned about is a hard error now. The conda-forge package sidesteps
   that entirely.

No CUDA base image, no GPU, no driver anywhere in the build. The binary is
still dynamically linked against those 7 CUDA/cuDNN libraries (confirmed via
`ldd`) even though `--no_gpu` never touches a GPU, so they have to be present
for the process to start — but that's ~1.5GB of PyPI wheels, not a 60GB+
CUDA devel image.

**This was validated end-to-end** (see chat history / session notes):
`gnina --no_gpu` docking gnina's own 184L reference ligand/receptor pair
inside a container with zero GPU or driver present, using this exact
Dockerfile — 9 poses, real CNN pose scores, no CUDA/driver errors.

## Deploy

```bash
pip install modal
modal setup   # one-time auth
modal deploy modal_app/gnina_worker.py
```

Run this from the repo root (`modal_app/Dockerfile` is resolved relative to
your current working directory, not the script's location).

The first deploy builds the Dockerfile stage on Modal's build infrastructure
(downloads the CUDA pip wheels + the gnina binary — a few minutes, one-time
cost). Modal caches that image; later deploys only rebuild if `Dockerfile`
or the `pip_install` line changes. This prints two URLs:

- `https://<workspace>--gnina-worker-dock.modal.run` — the original
  synchronous `dock` endpoint. Set as `GNINA_ENDPOINT_URL` in `.env` (used
  only by `agent3.py`'s standalone pipeline today).
- `https://<workspace>--gnina-worker-job-api.modal.run` — the async
  submit/poll job queue (`/submit`, `/status/{job_id}`) used by the Next.js
  app. Set as `GNINA_JOB_API_URL` in `.env`. Unauthenticated by design (see
  below) — anyone with this URL can submit/poll jobs.

## Async job queue (`/submit`, `/status/{job_id}`)

`dock` is a single blocking HTTP call — fine for `agent3.py`, but the
Next.js app's dock UI can run many ligands across several targets, well
past what a Vercel serverless function is allowed to stay open for. Instead
of holding a connection open, the app now:

1. `POST /submit` with `{targets, round}` (same shape as the old
   `app/api/dock` body) — validates the batch (`MAX_TARGETS_PER_JOB=10`,
   `MAX_LIGANDS_PER_JOB=50`, both in `gnina_worker.py`), spawns
   `run_docking_job` in the background via `.spawn()`, and returns
   `{jobId}` immediately.
2. `GET /status/{job_id}?since=N` — returns `{events, nextIndex, done}`,
   where `events` is every event appended since index `N`. Poll this on an
   interval until `done` is `true`.

`run_docking_job` writes progress into a `modal.Dict` (`gnina-dock-jobs`) as
it processes each target, so `/status` can return partial results while the
job is still running — mirroring the `progress` / `target_complete` /
`complete` / `error` events the old SSE route used to stream.

Both endpoints are unauthenticated by design — anyone with the URL can
submit/poll jobs. Unlike `dock`, `/submit` triggers background compute the
caller doesn't have to wait on, which is a much easier endpoint to abuse if
left open, so the only backstops are `MAX_TARGETS_PER_JOB`,
`MAX_LIGANDS_PER_JOB`, and a hard `JOB_TIMEOUT_SECONDS=3600` ceiling on
`run_docking_job` itself. If abuse becomes a real problem, the endpoint
previously required an `X-Gnina-Auth` header checked against a Modal
secret — see git history on this file to reinstate it.

## Local test / iteration

```bash
modal serve modal_app/gnina_worker.py
```

gives a temporary URL for iterating without a full deploy.

## Smoke test the deployed endpoint

```bash
python3 - <<'EOF'
import base64, json, urllib.request

pdb_bytes = open("path/to/receptor.pdb", "rb").read()
payload = {
    "protein_pdb_b64": base64.b64encode(pdb_bytes).decode(),
    "ligands": [{"name": "test", "smiles": "CCO"}],
    "samples_per_complex": 5,
}
req = urllib.request.Request(
    "https://<workspace>--gnina-worker-dock.modal.run",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
EOF
```

A successful response has `results[0].confidence_score` populated (the CNN
pose score), an empty `errors` list, and a `binding_site_method` field
telling you which tier the box came from (see below).

## Binding-site box: 3-tier fallback

GNINA needs an explicit box to dock into (unlike DiffDock's blind search).
`gnina_worker.py` determines it per-request, trying each tier in order and
falling through on failure:

1. **`cocrystallized_ligand`** — parses `HETATM` records already in the
   receptor PDB, filters out crystallization artifacts (waters, ions,
   cryoprotectants, buffers, sugars, common cofactors — see
   `ARTIFACT_RESNAMES`), and boxes the largest qualifying residue with 4Å
   padding (GNINA/Vina's own `--autobox_add` default).
2. **`pocket_prediction`** — if no ligand qualifies, runs `fpocket` on the
   receptor and boxes its top-ranked (by druggability score) predicted
   pocket. If the binary is missing or fails for any reason, this tier is
   silently skipped (falls through to Tier 3) rather than breaking the
   request.
3. **`blind_docking`** — a padded box over the whole receptor. The original
   always-on behavior, now used only when both tiers above find nothing.

No caller-side changes are needed: the box is computed entirely inside the
worker from the PDB it already receives, and `binding_site_method` is
additive on the response (existing callers that only read `results` are
unaffected).

**Validated end-to-end on the deployed endpoint** against the sample PDBs in
`structures/`: `8A27.pdb` (has a real co-crystallized ligand) → returns
`cocrystallized_ligand`; `2FMA.pdb` and `3QL9.pdb` (no qualifying ligand) →
return `pocket_prediction`. `3QL9.pdb` specifically exercises the
modified-residue guard (`_modified_residue_names()`, parsed from `MODRES`
records) — it contains `M3L` (trimethyllysine), a modified amino acid
covalently bonded into the chain via `LINK` records, which would otherwise
be a false-positive "ligand" without that check.

## Resource sizing

Currently `cpu=2, memory=4096` on the `dock` function (see
`@app.function(...)` in `gnina_worker.py`). That was sized before real
per-target timing was available on Modal's own hardware — a tiny synthetic
test case (10-atom ligand) peaked around 360MB RSS and ~3.5 cores of
parallel search under `--cpu 4`. Re-measure once a real drug-repurposing
target is docked through this endpoint and tune from there rather than
over-provisioning up front. Modal bills per-second of actual CPU+RAM used
(~$0.047/core/hr, ~$0.008/GiB/hr as of this writing), not per-hour like a
VM, so idle time between requests costs nothing unless `min_containers` is
set to keep a container warm.
