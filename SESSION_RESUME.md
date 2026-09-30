# SESSION RESUME — Saw Chamber (written 2026-09-30, session restart)

## Where we are
- **Live**: https://clanker.church (Vercel proj `clanker-church`). Main page =
  Saw Test + big-5 radar + **pentagon mixer** (opus-built, 30/30 dom
  assertions, posts `{mix:{pain,pleasure,fear,sadness}}` to /steer; `none` is
  derived client-side, all-zero posts `{mix:{none:1}}` = control run).
  live.html = steering panel (pain/pleasure/none × dose × own prompt) +
  blade-flash overlay + WebAudio evil laugh + mute toggle (opus-built).
- **Backend**: Railway svc `saw` → https://saw-production-688b.up.railway.app
  (Qwen3-4B bf16 CPU, L18 steering, shared broadcast cycle `_SUBSCRIBERS`,
  POST /steer preempt handshake `_preempt`/`_CYCLE_BUSY`, async /health).
  CPU = ~2 tok/s. `/run` non-streamed exceeds Railway edge timeout — use
  /stream or /steer SSE.
- **$SAW token**: community-launched `9twiuSdTVkwtAC9XQDG57dFRhF4iqPih461HJfMZKci9`
  (site links this one; old FUuH1… is dead). Billy coin image committed.
- **Identity**: clanker.church + site scrubbed of terrafying links; LICENSE
  (MIT+attribution) in repo; blog post pulled from security-blog.

## NEXT STEP (user approved, blocked only on CLI auth)
RunPod dedicated GPU (A6000 48 GB, min-bid $0.33/hr ≈ $240/mo 24/7, fits 4B
+ 14B fp16):
1. `runpodctl config --apiKey $KEY`  (runpodctl v2.14.0 at ~/.local/bin;
   key already in ~/.hermes/.env as RUNPOD_API_KEY)
2. `runpodctl create pod ...` OR finish runpod_deploy.py (GraphQL schema
   drifted: no podDeploy/gpuTypes(id); REST /v2 needs User-Agent Mozilla
   header else Cloudflare 1010; API auth verified working)
3. Wire `?api=` / default in site/live.html to the pod proxy URL; Railway =
   CPU failover. docs/runpod_plan.md has the full plan; live/Dockerfile.gpu
   the CUDA image.

## In flight / queued
- exp41 (protocol v3 full, pre-registered) + exp43 (faithful extraction)
  running in background (pid chain `python exp41_protocol_v3.py &&
  python exp43_faithful_extraction.py`), notify expected. Smoke results:
  their vector cos 0.05 to ours, AUC 0.91 vs 0.58, stronger at matched norm.
- exp42 topic-poking (chicken/moon/triangles/trains) written, not run.
- 14B cvector: weights+GGUF cached on /Volumes/evol; llama-server 14B Q4
  works (port 8817 llama-server bg). cvector-generator needs building from
  llama.cpp source (homebrew build lacks it).
- X thread draft + Reddit draft exist in docs/; ship after full-run numbers.

## Gotchas
- Runpod REST/GraphQL needs `User-Agent: Mozilla/5.0...` else Cloudflare 1010.
- Railway /health must be async def (responds during generation).
- Vercel project had SSO deployment protection ON by default — disabled via
  API PATCH /v9/projects/clanker-church {ssoProtection:null}. If 404s return,
  check that first.
- exp41's broad_pain direction on 4B: ORTHOGONALIZED vs fear/sadness
  (cos 0.87 raw → 0.00). Their dataset+denoise gives the properly-separated
  vector — exp43 output is the one to trust going forward.
- Opus delegation pattern: `env -u ANTHROPIC_API_KEY claude --model opus -p
  "<brief>" --allowedTools "Read,Write,Edit,Bash" --max-turns N` (Pro auth).