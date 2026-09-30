#!/usr/bin/env python3
"""runpod_deploy.py — create the Saw chamber GPU pod (A6000/L40S, on-demand).

Container: pytorch base image, bootstrap = clone the public repo + pip
install + uvicorn server.py on 8000/http. Runpod gives a public proxy URL.
Env-driven: RUNPOD_API_KEY in ~/.hermes/.env. Print the pod id + URL.
"""
import json, os, pathlib, sys, urllib.request

key = [l.split("=", 1)[1].strip() for l in
       open(pathlib.Path.home() / ".hermes/.env") if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

BOOTSTRAP = (
    "set -e; cd /workspace; "
    "git clone -q https://github.com/terrafying/ai-torture-chamber.git repo 2>/dev/null || "
    "(cd repo && git pull -q); "
    "cd repo/live; "
    "pip install -q -r requirements.txt 'transformers>=4.51' 2>&1 | tail -1; "
    "export HF_HOME=/workspace/hf CHAMBER_DEVICE=cuda CHAMBER_DTYPE=float16 "
    "CHAMBER_LAYER=${CHAMBER_LAYER:-18} PORT=8000; "
    "python -m uvicorn server:app --host 0.0.0.0 --port 8000")

MUT = """
mutation ($input: PodCreateInput) {
  podDeploy(input: $input) {
    id
    desiredStatus
    podName
    machine { podHostId }
  }
}"""

GPU_TYPES = ["NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA GeForce RTX 4090"]

def gql(query, variables=None):
    req = urllib.request.Request(
        "https://api.runpod.io/graphql?beta=true",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers=H)
    try:
        r = json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    if r.get("errors"):
        raise RuntimeError(r["errors"])
    return r["data"]

# pick the first available GPU type (schema: gpuTypes takes no id arg —
# query unfiltered, filter client-side; price field = minimumBidPrice)
chosen = None
data = gql("""query { gpuTypes { id lowestPrice { minimumBidPrice } } }""")
by_id = {g["id"]: g for g in data["gpuTypes"]}
for gpu in GPU_TYPES:
    g = by_id.get(gpu)
    if not g:
        print(f"{gpu}: not listed", flush=True)
        continue
    price = g["lowestPrice"]["minimumBidPrice"] if g.get("lowestPrice") else None
    print(f"{gpu}: min-bid ${price}/hr", flush=True)
    if price and price <= 0.85:
        chosen = (gpu, price)
        break
if not chosen:
    sys.exit("no suitable GPU under $0.85/hr")
gpu, price = chosen
print(f"deploying on {gpu} @ min-bid ${price}/hr", flush=True)

vars = {"input": {
    "cloudType": "SECURE",
    "gpuTypeId": gpu,
    "containerImage": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
    "args": ["/bin/sh", "-c", BOOTSTRAP],
    "volumeInGb": 40,
    "volumeMountPath": "/workspace",
    "ports": "8000/http",
    "env": [{"key": "HF_HOME", "value": "/workspace/hf"},
            {"key": "CHAMBER_LAYER", "value": "18"}],
    "supportPublicIp": True,
    "startSsh": False,
    "name": "saw-chamber-gpu",
}}
data = gql(MUT, vars)
pod = data["podDeploy"]
print("pod created:", pod["id"], "| status:", pod["desiredStatus"], flush=True)
open("runs/exp39/runpod_pod.json", "w").write(json.dumps(pod, indent=1))
print(f"proxy url once running: https://{pod['id']}-8000.proxy.runpod.net")
