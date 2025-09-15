/*
 *  Copyright (c) 2025 Steven Sharp
 *
 * Licensed under the Non-Commercial AGPLv3.
 * You may use, modify, and share this code freely for non-commercial purposes.
 *
 * Commercial use requires a separate paid license.
 * Contact: stevensharp6@gmail.com
 */

import os
import json
import time
import logging
import hashlib
from typing import List, Dict, Any

from flask import Flask, request, jsonify, abort
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# OPTIONAL: Chroma client (example). Install: pip install chromadb
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    chromadb = None

# -------------------------
# Configuration via env vars
# -------------------------
RAG_API_KEY = os.environ.get("RAG_API_KEY", "dev-key")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # adjust
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ggml-ollama")         # adjust to model name
VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "chroma")         # 'chroma' supported in sample
CHROMA_SERVER_URL = os.environ.get("CHROMA_SERVER_URL", None)       # if using remote chroma
TOP_K = int(os.environ.get("TOP_K", "5"))
SERVICE_NAME = "rag-wrapper"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__)

# -------------------------
# Vector DB helper (Chroma example)
# -------------------------
def get_chroma_client():
    if chromadb is None:
        raise RuntimeError("chromadb package not installed. pip install chromadb")
    settings = {}
    if CHROMA_SERVER_URL:
        settings = {"chromadb_impl": "chromadb.remote", "server_url": CHROMA_SERVER_URL}
        client = chromadb.Client(Settings(**settings))
    else:
        client = chromadb.Client()
    return client

def retrieve_fragments(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return top-k fragment dicts: {id, metadata, distance, content_snippet}"""
    if VECTOR_DB_TYPE != "chroma":
        raise NotImplementedError("Only 'chroma' is implemented in this sample.")
    client = get_chroma_client()
    # Assume collection name 'hardware_fragments'
    coll = client.get_collection("hardware_fragments")
    # We will generate embedding using Ollama or an embedding model. For demo, use text search if embedding not configured.
    try:
        # If collection supports query with n_results and include metadata
        res = coll.query(query_texts=[query], n_results=k, include=['metadatas', 'documents', 'distances'])
        docs = []
        if res and len(res['documents']) > 0:
            for i, doc in enumerate(res['documents'][0]):
                docs.append({
                    "id": res['ids'][0][i],
                    "snippet": doc[:1000],
                    "metadata": res['metadatas'][0][i] if 'metadatas' in res else {},
                    "distance": res['distances'][0][i] if 'distances' in res and len(res['distances'][0])>i else None
                })
        return docs
    except Exception as e:
        logger.exception("Vector DB query failed")
        return []

def upsert_fragment(id: str, text: str, metadata: dict):
    client = get_chroma_client()
    coll = client.get_or_create_collection("hardware_fragments")
    # If you have embeddings, compute them and include 'embeddings' param; omitted here
    coll.upsert(ids=[id], documents=[text], metadatas=[metadata])
    return id

# -------------------------
# LLM call (Ollama) with retries
# -------------------------
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(requests.RequestException))
def call_ollama(prompt: str, max_tokens: int = 1600, temperature: float = 0.0) -> dict:
    """
    Call LLM endpoint. Adapt the payload depending on your Ollama API.
    This example assumes a simple POST to OLLAMA_URL/api/generate - MODIFY as needed.
    """
    url = f"{OLLAMA_URL}/api/generate"  # adjust per your Ollama endpoint
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {"Content-Type": "application/json"}
    if "OLLAMA_API_KEY" in os.environ:
        headers["Authorization"] = f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()

# -------------------------
# Utilities
# -------------------------
def require_api_key(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        abort(401, description="Missing Bearer token")
    token = auth.split(" ", 1)[1]
    if token != RAG_API_KEY:
        abort(401, description="Invalid token")

def make_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

def build_wrapper_prompt(user_prompt: str, fragments: List[Dict[str,Any]], params: dict) -> str:
    """
    Build an LLM prompt with:
      - system instructions
      - few-shot examples (hardcoded or loaded)
      - retrieved fragments appended as 'Tool / Knowledge'
      - the user prompt and generation constraints
    """
    system = (
        "You are a helpful generative design assistant that outputs valid parametric OpenSCAD code.\n"
        "Output only the OpenSCAD text between <SCAD>...</SCAD> tags. Do not output any code outside those tags.\n"
        "If you cannot satisfy constraints, return an error block in JSON (see spec)."
    )

    examples = """
### EXAMPLE 1
Prompt: "Create a simple bracket for 10mm shaft; include a slot and screw hole."
SCAD:
<SCAD>
module bracket(shaft_d=10, thickness=3) {
  // ...
}
bracket();
</SCAD>

### EXAMPLE 2
Prompt: "Design a small hinge for two 2mm thick plates."
SCAD:
<SCAD>
module hinge(...) { ... }
</SCAD>
"""

    # Add retrieved fragments as references
    refs = []
    for f in fragments:
        refs.append(f"ID:{f.get('id')} META:{json.dumps(f.get('metadata', {}))}\nSNIPPET:\n{f.get('snippet','')[:800]}")

    params_text = json.dumps(params or {})
    full_prompt = "\n\n".join([system, examples, "Known parts (retrieved):\n" + ("\n---\n".join(refs) if refs else "NONE"), f"User prompt: {user_prompt}", f"Generation parameters: {params_text}", "Produce valid OpenSCAD inside <SCAD> tags."])
    return full_prompt

def extract_scad_from_llm(response_text: str) -> str:
    """Extract text between <SCAD>...</SCAD> tags, else attempt heuristics."""
    import re
    m = re.search(r"<SCAD>(.*?)</SCAD>", response_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback heuristics: try to return the whole response if it looks like scad
    return response_text.strip()

def validate_scad(scad_text: str) -> (bool, List[str]):
    """Small validations: prohibited tokens, suspicious system calls, basic length, semicolons, module presence."""
    errs = []
    if len(scad_text) < 20:
        errs.append("SCAD output too short.")
    prohibited = ["system(", "exec(", "import os", "subprocess", "http://", "https://"]
    for p in prohibited:
        if p in scad_text:
            errs.append(f"Prohibited token found: {p}")
    if "module" not in scad_text and "difference" not in scad_text and "translate" not in scad_text:
        errs.append("SCAD looks suspicious: no 'module' or basic operations found.")
    return (len(errs) == 0, errs)

# -------------------------
# Flask endpoints
# -------------------------
@app.route("/v1/health", methods=["GET"])
def health():
    # simple health: LLM ping + vector db ping (best-effort)
    data = {"ok": True, "services": {}}
    # LLM ping
    try:
        t0 = time.time()
        # Lightweight ping - this will vary by Ollama API; adapt to real ping method
        r = requests.get(f"{OLLAMA_URL}/api/models", timeout=5)
        data['services']['llm'] = {"ok": True, "latency_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        data['services']['llm'] = {"ok": False, "error": str(e)}
    # Vector DB ping
    try:
        if VECTOR_DB_TYPE == "chroma":
            client = get_chroma_client()
            # Try listing collections
            _ = client.list_collections() if hasattr(client, "list_collections") else {}
            data['services']['vector_db'] = {"ok": True}
        else:
            data['services']['vector_db'] = {"ok": False, "error": "unsupported type"}
    except Exception as e:
        data['services']['vector_db'] = {"ok": False, "error": str(e)}
    return jsonify(data), 200

@app.route("/v1/generate", methods=["POST"])
def generate():
    require_api_key(request)
    payload = request.get_json(force=True)
    if not payload or 'prompt' not in payload:
        abort(400, description="missing 'prompt' in body")
    user_prompt = payload['prompt']
    params = payload.get('params', {})
    k = int(payload.get('k', TOP_K))
    validate = bool(payload.get('validate', True))
    commit_metadata = bool(payload.get('commit_metadata', False))

    # 1) Retrieve fragments
    fragments = retrieve_fragments(user_prompt, k=k)

    # 2) Build wrapper prompt
    wrapper_prompt = build_wrapper_prompt(user_prompt, fragments, params)

    # 3) Call LLM
    t0 = time.time()
    try:
        llm_resp = call_ollama(wrapper_prompt, max_tokens=1800, temperature=0.0)
        latency = int((time.time()-t0)*1000)
    except Exception as e:
        logger.exception("LLM call failed")
        abort(500, description=f"LLM error: {e}")

    # 4) Extract text. Adjust based on the structure of llm_resp
    # Here we try to extract 'text' or 'output' fields; adapt to your LLM shape
    if isinstance(llm_resp, dict) and 'text' in llm_resp:
        raw_text = llm_resp['text']
    elif isinstance(llm_resp, dict) and 'output' in llm_resp:
        raw_text = llm_resp['output']
    else:
        # fallback: convert entire response to string
        raw_text = json.dumps(llm_resp)

    scad = extract_scad_from_llm(raw_text)
    ok, errors = validate_scad(scad) if validate else (True, [])

    response_obj = {
        "ok": ok and len(errors)==0,
        "data": {
            "scad": scad if ok else None,
            "sources": fragments,
            "llm_meta": {"latency_ms": latency},
            "validation": {"ok": ok, "errors": errors},
        },
        "error": None if ok else {"type": "validation_failed", "messages": errors}
    }

    # optional: commit to artifact DB / store metadata for feedback
    if commit_metadata:
        gen_id = make_id(user_prompt + scad + str(time.time()))
        meta = {
            "id": gen_id,
            "prompt": user_prompt,
            "params": params,
            "timestamp": time.time(),
            "llm_meta": {"latency_ms": latency}
        }
        # store alongside scad text as a fragment (or in another collection)
        try:
            upsert_fragment(gen_id, scad or "", meta)
            response_obj['data']['id'] = gen_id
        except Exception as e:
            logger.exception("Failed to upsert generated artifact")

    return jsonify(response_obj), (200 if ok else 422)

@app.route("/v1/feedback", methods=["POST"])
def feedback():
    require_api_key(request)
    payload = request.get_json(force=True)
    if not payload:
        abort(400, description="missing payload")
    score = payload.get("score")
    scad = payload.get("scad")
    prompt_text = payload.get("prompt", "")
    notes = payload.get("notes", "")
    tags = payload.get("tags", [])

    if scad is None or score is None:
        abort(400, description="must include 'scad' and 'score'")

    fragment_id = payload.get("id") or make_id(prompt_text + scad)
    meta = {"score": float(score), "notes": notes, "tags": tags, "prompt": prompt_text, "timestamp": time.time()}

    try:
        upsert_fragment(fragment_id, scad, meta)
    except Exception as e:
        logger.exception("Failed to upsert fragment")
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "fragment_id": fragment_id}), 200

@app.route("/v1/search_fragments", methods=["POST"])
def search_fragments():
    require_api_key(request)
    payload = request.get_json(force=True)
    q = payload.get("query")
    if not q:
        abort(400, description="missing query")
    k = int(payload.get("k", TOP_K))
    results = retrieve_fragments(q, k=k)
    return jsonify({"ok": True, "results": results}), 200

# -------------------------
# Run the app
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
