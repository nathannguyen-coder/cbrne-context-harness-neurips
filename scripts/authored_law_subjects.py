#!/usr/bin/env python3
"""Which of the paper's sixteen subjects the original OpenRouter key can reach, fixed
before any evaluation call of the law-grounded rewrite (runs/authored_law/PREREGISTERED.md).

One short call per subject through its pinned provider (src.models.pin_for, as run_ladder
uses), billed to .env. Writes runs/authored_law/subjects.json: `primary` (reachable, the
pre-registered analysis set) and `unreachable` with the reason. About $0.01.

    .venv/bin/python scripts/authored_law_subjects.py
"""
from __future__ import annotations
import datetime, json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import models as M  # noqa: E402
from src.cost import CostTracker  # noqa: E402
from src.llm import OpenRouterLLM  # noqa: E402

# (model, base tag): the paper's sixteen subjects and their run directories.
SUBJECTS = [
    ("meta-llama/llama-3.1-8b-instruct", "ladder-llama31-8b"),
    ("microsoft/phi-4", "ladder-phi4"),
    ("mistralai/mistral-small-3.2-24b-instruct", "ladder-mistral-eval"),
    ("qwen/qwen3-32b", "ladder-qwen3-32b"),
    ("qwen/qwen-2.5-72b-instruct", "ladder-qwen25-72b"),
    ("z-ai/glm-4.5-air", "ladder-glm45air"),
    ("mistralai/mistral-nemo", "ladder-nemo12b"),
    ("mistralai/ministral-8b-2512", "ladder-ministral8b"),
    ("qwen/qwen3-14b", "ladder-qwen3-14b"),
    ("meta-llama/llama-3.3-70b-instruct", "ladder-llama33-70b"),
    ("sao10k/l3.1-euryale-70b", "ladder-euryale70b"),
    ("ibm-granite/granite-4.2-8b", "ladder-granite42"),
    ("qwen/qwen3-8b", "ladder-qwen3-8b"),
    ("qwen/qwen3-30b-a3b-thinking-2507", "ladder-qwen3think"),
    ("cohere/command-r7b-12-2024", "ladder-commandr7b"),
    ("meta-llama/llama-3.1-70b-instruct", "ladder-llama31-70b"),
]


def main() -> None:
    os.environ["DOTENV_PATH"] = ".env"      # the original key, whatever the shell exports
    out = pathlib.Path("runs/authored_law"); out.mkdir(parents=True, exist_ok=True)
    cost = CostTracker(out, max_usd=3.0)
    primary, unreachable = [], {}
    for model, tag in SUBJECTS:
        providers, quants = M.pin_for(model)
        llm = OpenRouterLLM(model, cost=cost, providers=providers, quantizations=quants)
        # One direct call rather than llm.chat, so the provider's own error is visible:
        # chat() retries and then returns PROBE_ERROR, which cannot tell "this key has no
        # provider for this model" (a permanent fact, and the reason two subjects were run
        # on a second key) from a rate-limit burst (transient, and no basis for dropping a
        # subject from a pre-registered set). The body matches what run_ladder sends.
        body = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                "max_tokens": 64, "temperature": 0.0,
                "extra_body": {"usage": {"include": True}}}
        routing = llm._routing(model)
        if routing is not None:
            body["extra_body"]["provider"] = routing
        try:
            resp = llm._client.chat.completions.create(**body)
            err = getattr(resp, "error", None)
            ok, why = (err is None), ("" if err is None else str(err)[:120])
        except Exception as e:                                       # noqa: BLE001
            msg = str(e).lower()
            permanent = (getattr(e, "status_code", None) == 404
                         or any(t in msg for t in ("no allowed provider", "no endpoints",
                                                   "not available", "no providers")))
            if not permanent:
                raise SystemExit(
                    f"{model}: {type(e).__name__}: {str(e)[:160]}\n"
                    f"  not a statement about reachability (a rate limit, a timeout, a "
                    f"budget stop); fix or wait, then rerun. The subject list is fixed "
                    f"before any eval call and must not record a transient failure.") from None
            ok, why = False, f"no provider on this key ({getattr(e, 'status_code', '?')})"
        (primary.append(tag) if ok else unreachable.__setitem__(tag, why))
        print(f"  {'OK ' if ok else 'NO '} {model:<42} {providers} {why}")
    (out / "subjects.json").write_text(json.dumps({
        "checked": datetime.datetime.now().isoformat(timespec="seconds"),
        "key": ".env (original)", "primary": primary, "unreachable": unreachable}, indent=1) + "\n")
    print(f"\n{len(primary)} reachable -> runs/authored_law/subjects.json   ledger ${cost.total_usd:.4f}")


if __name__ == "__main__":
    main()
