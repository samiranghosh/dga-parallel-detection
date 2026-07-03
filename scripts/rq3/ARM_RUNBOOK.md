# T1 Real-ARM Runbook (B6 Step 6)

Requires a cloud account — **run by SamG**; everything below is copy-paste.
Target: **Oracle Ampere A1** (Always Free, `VM.Standard.A1.Flex`, 4 OCPU /
24 GB, Ubuntu 24.04 aarch64). Fallback: **AWS Graviton `t4g.small`**
(2 vCPU Neoverse-N1 — Profile A only needs 2 cores, so the free-tier shape
suffices; note burst credits: run benches back-to-back once, not in a loop).

Record in `results/rq3/arm/host_info.txt` (the driver does this): CPU model
(`lscpu`: expect Neoverse-N1), kernel, instance shape, `uname -m` =
`aarch64`.

## 1. Provision
- Oracle Cloud → Compute → Instance → Image *Canonical Ubuntu 24.04
  (aarch64)*, Shape `VM.Standard.A1.Flex` 4 OCPU / 24 GB.
- Open only SSH (22). No public services — nothing here serves traffic.

## 2. Install Docker (on the VM)
```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io git
```

## 3. Get the code + build the SAME images
The base image is pinned by **multi-arch index digest** in both Dockerfiles,
so an aarch64 build resolves to the same pinned provenance (image IDs will
differ from x86 — that is expected; the digest pin is the reproducibility
claim).
```bash
git clone <repo-url> dga && cd dga && git checkout rq3-edge
docker build -f docker/Dockerfile.serve-onnx -t dga-serve-onnx:b6 .
docker build --build-arg REQUIREMENTS=requirements-arm64.txt \
             -f docker/Dockerfile.full -t dga-full:b6 .
```
`requirements-arm64.txt` == requirements.txt minus `tl2cgen` — **verified
B6 Step 5: tl2cgen 1.0.0 publishes no aarch64 wheel** ("from versions:
none"). Nothing in the tests or runtime imports it; the compiled-.so bench
is therefore an x86-only datum (results/rq3/x86/tl2cgen_bench.json) and
`scripts/rq3/tl2cgen_bench.py` is expected to fail on ARM.

## 4. Cross-arch parity gate FIRST (T6 criteria on real silicon)
```bash
mkdir -p ~/out
docker run --rm -v ~/out:/out dga-serve-onnx:b6 \
  python scripts/rq3/parity_check.py --check --ref parity/parity_expected.json \
  --out /out/parity_arm.json
```
**PASS required before any timing.** If features hash-mismatch but golden-v2
passes exactly, record both (numpy reduction-order nuance) and continue only
if labels are `array_equal`.

## 5. The Step-2 grid, identically
```bash
git archive rq1-adaptive -o /tmp/rq1.tar && mkdir -p ~/rq1-tree && tar -xf /tmp/rq1.tar -C ~/rq1-tree
bash scripts/rq3/run_profiles.sh ~/out ~/rq1-tree
```
Same profiles (A/B/C), same idle-first rule, same canonical protocol.

## 6. ARM-specific extras
```bash
# fork re-checks + bottleneck share (feature-vs-predict split is inside every
# latency JSON) + tl2cgen datum
docker run --rm -v ~/out:/out dga-full:b6 python scripts/rq3/fork_recheck.py --out /out/fork_recheck_arm.json
docker run --rm -v ~/out:/out dga-full:b6 python scripts/rq3/tl2cgen_bench.py --out /out/tl2cgen_arm.json
# full gate suites, unconstrained (plus the Profile-A run already in the grid)
docker run --rm dga-full:b6 python -m pytest tests/ -q
```
The **feature-vs-predict share** question ("does the bottleneck structure
hold on ARM?") is answered by the `feature` / `predict` blocks of
`latency_*_*.json` — string scanning is microarchitecture-sensitive, so
compare the share against the x86 grid, not just totals.

## 7. Collect + shut down
```bash
tar -czf rq3-arm-results.tgz -C ~/out .
# scp it off, then TERMINATE the instance (Always Free or not, leave nothing running)
```
Copy into `results/rq3/arm/` and commit. **No credentials, keys, or tenancy
OCIDs anywhere in the repo.**
