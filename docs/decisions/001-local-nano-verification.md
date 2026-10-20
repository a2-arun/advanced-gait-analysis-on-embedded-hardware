# ADR-001: Local frozen GaitPT claimed-identity verification on original Nano

Status: accepted engineering direction; deployment validation pending.
Date: 20 October 2026.

The owner requires verification of a supplied identity claim only, and the whole
inference pipeline on the original Jetson Nano. The old application performed 1:N
search with a legacy pairwise model and fixed threshold. The rebuilt research
system uses frozen gait-specific embeddings and currently has no approved policy.
The Nano's JetPack 4 / Python 3.6 / NVIDIA PyTorch 1.10 stack cannot safely use the
modern restricted checkpoint loader directly.

Use the unmodified pinned official GaitPT source and the audited GREW weights.
Prepare checked numeric NPZ weights offline using modern restricted deserialization;
load numeric arrays locally without pickle on the Nano. Use the Nano-compatible
MediaPipe 0.8.5 pose route as a **separate experimental recipe**, with a provisional
quality gate and explicit one-walker operator review. Keep RTMPose and GPGait-family
comparison in the research system until a separately verified Nano port is assigned.
The default result for usable enrolled claims is POLICY_NOT_READY/null. Enable
Boolean comparison only through trusted, recipe-bound calibration evidence.

Remote inference does not satisfy the owner requirement. Importing the modern
research environment would not establish Python 3.6/Nano compatibility. Retaining
the old threshold would invent a valid operating point for a different score scale.
Publishing modified upstream model source would also require license review; the
source therefore remains an external pinned unmodified dependency.

Consequences: no local training; independent references must be re-enrolled; old
assets are archived/preserved; full hardware, capture-quality, calibration and license
gates remain open. Workstation parity proves numeric conversion, not Nano deployment
or a scientific model winner.
