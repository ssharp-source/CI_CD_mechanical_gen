# CI_CD_mechanical_gen

### big idea
use github/gitlab to recursivly generate machincal hw through generative AI then QC with simulation and user feedback then feed the score back inot the start of the pipe


### tech stack
ci/cd kicked off with user prompt committed to branch.
at the hart a RAG set up with ollama under the hood with db with talored wrapper promts and Hardware db knoledge of known mechanizume and assembleis 
Openscad or similar as a STL solidmodle output frame work (think code to object) 
as for simulation inital though is physics engine in unity to simulate effects (gravity, collisions, ...) 



### use case 
scale comput based on task, llm using GPUs while openscad would be fine with cpu 


## Deployment & operational notes 

Secrets: store RAG_API_KEY, OLLAMA_API_KEY, vector DB credentials in your secrets manager. Never commit them.
Rate limiting: front the service with an API gateway (or nginx) to enforce request rates and throttling. LLM calls are expensive.
Idempotency: generate deterministic id values from prompt+params+seed so reruns are traceable.
Logging & observability: emit metrics (LLM latencies, retrieval count, validation passes) to Prometheus and traces to a tracing backend. Keep LLM responses in logs briefly for debugging, but redact PII.
Testing: create unit tests for:
build_wrapper_prompt formatting,

extract_scad_from_llm heuristics,
validate_scad checks.

Use a fake LLM server in tests to simulate responses.
Safety: ensure you have strict validation on SCAD and do not execute any generated code in environments that can run arbitrary commands. Treat generated STL rendering and Unity sims as separate processes with resource/time quotas.

Reproducibility: store random_seed, model_name, and temperature in metadata when you store a generation; this allows you to recreate it.



## example usage:


example curl:
curl -X POST "https://rag.example.com/v1/generate" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Small clamp for 10mm shaft with M3 fastener holes","params":{"material":"nylon","target_bbox_mm":[60,20,10]},"k":5}'


expected feedback

curl -X POST "https://rag.example.com/v1/feedback" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"id":"abc123","scad":"module ...","score":0.45,"notes":"hinge clearance too tight"}'
