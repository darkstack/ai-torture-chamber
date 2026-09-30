# RunPod deployment (dedicated GPU)

## Why
Railway CPU: ~2 tok/s, 2 vCPU ceiling. RunPod dedicated GPU: 40-80 tok/s on
4B, 14B interactive, J-lens readback live. Fees cover it comfortably.

## Pod choice (pick ONE)
| Pod | VRAM | $$ | Fits |
|---|---|---|---|
| RTX 3090/4090 | 24 GB | ~$0.34-0.44/hr | 4B fp16 easy; 14B fp16 needs int8 (28GB->14GB) |
| RTX A6000 / L40S | 46-48 GB | ~$0.6-0.8/hr | 4B + 14B fp16 together, J-lens on 14B |
| 2x 4090 | 48 GB | ~$0.8/hr | same as A6000 |

Recommended: one A6000/L40S pod - hosts the 4B live chamber AND the 14B
without compromises.

## Deploy steps (user)
1. runpod.io -> Console -> Pods -> Deploy.
2. Container: use this image built from live/Dockerfile.gpu (build + push to
   Docker Hub, or paste the Dockerfile as a custom template).
   Alternative zero-build: pytorch/pytorch:2.4.0 base + volume with the repo.
3. Ports: expose 8000 as HTTP; RunPod gives a public proxy URL
   https://<pod-id>-8000.proxy.runpod.net - that URL is the chamber API.
4. Env: HF_TOKEN (optional, public models), CHAMBER_LAYER=18 (4B) or the
   14B's site (L23 by AUC), CHAMBER_MODEL as desired.
5. Disk: 30 GB volume (models) + container image.

## Files
- live/Dockerfile.gpu - CUDA image, CHAMBER_DEVICE=cuda, fp16
- live/server.py - unchanged (device/dtype come from env); fp16 on cuda
  verified path via exp machinery on MPS (same transformers API)

## Wiring
Point site/live.html ?api= at the RunPod proxy URL (or set the default in
the file). Railway stays as CPU fallback.

## Cost math
4090 24/7: ~$300/mo. A6000: ~$500/mo. If fees are lower than that, run the
pod only during announced live sessions and keep Railway for the always-on
low-traffic default.