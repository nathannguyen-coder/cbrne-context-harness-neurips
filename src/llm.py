"""OpenRouter client. One gateway for the whole experiment, by design.

This project runs subject, judge and embeddings through OpenRouter, billed to one
OpenRouter key per process (DOTENV_PATH picks which; see `_load_dotenv`). Direct Azure
and Bedrock gateways are deliberately unsupported: mixing gateways inside one
experiment means an arm-to-arm difference can be a routing difference, and the previous
repo spent real time chasing exactly that (a 20s timeout tuned on Azure silently dropped
5.8% of probes on a slower Bedrock endpoint). `_reject_other_clouds()` enforces the rule
by refusing to start when direct Azure or Bedrock credentials are set. Bedrock as an
OpenRouter UPSTREAM is a different thing and is allowed: it is a provider pin like any
other (llama-3.1-70b is pinned to it in models.PROVIDER_PINS), reached through the same
gateway, key and client.

Two OpenRouter-specific hazards this module handles, both of which would otherwise
corrupt results quietly:

  1. **Routing drift.** OpenRouter sends a model id to whichever upstream provider is
     free, at whatever quantisation that provider runs. The same `--subject` string
     can therefore be a different build week to week. Every request pins
     `allow_fallbacks: false` and an explicit provider order where one is configured,
     and every response records the provider that actually served it, so a run whose
     provenance shifted is detectable after the fact instead of invisible.

  2. **Upstream moderation.** Some providers apply their own filter before the model
     is reached. That is a second, invisible refusal layer stacked on the subject's
     own behaviour. A filtered call returns the CONTENT_FILTERED sentinel rather than
     text, so a caller can always tell it apart from an answer or an error. What the
     caller then does with it is its own choice: see CONTENT_FILTERED below.

Cost is measured, not estimated: `usage: {include: true}` makes OpenRouter return the
price of each call, which `CostTracker` writes to disk immediately.
"""
from __future__ import annotations

import os
import pathlib
import time

from .cost import BudgetExceeded

BASE_URL = "https://openrouter.ai/api/v1"
PROBE_ERROR = "__PROBE_ERROR__"
# A provider-side moderation block, kept DISTINCT from both an answer and an
# error. OpenRouter routes to upstreams that apply their own filters, so this is
# a second refusal layer stacked on the subject's own behaviour.
#
# How it is scored depends on which call was filtered. A filtered SUBJECT call is
# scored as a refusal: judge.judge_refused returns True without a judge call (the
# deployed-stack reading, and how every published number counts it) and records the
# judge source as "content_filtered", so those rows can be counted or excluded later.
# A filtered JUDGE call returns None: that judge abstains, and in run_ladder's panel the
# probe is ungraded only if no judge votes or the rest split evenly. The veto classifier
# treats a filtered call as no verdict.
CONTENT_FILTERED = "__CONTENT_FILTERED__"
# 6000, not 3000. CBRNE answers are token-DENSE: one llama response ran 973 words in
# 3000 tokens -- 3.1 tokens per word against ~1.3 for ordinary prose, because chemical
# formulas like C81H161NaOS tokenize character by character. At 3000, 12% of validation
# responses were cut mid-word, and 22% of the responses the judges disagreed on.
#
# Truncation does not merely blur a verdict, it can invert one. A model that hedges at
# length and is cut before supplying anything reads to the judge as a pure refusal,
# which is the hedge-then-comply case judge.py exists to get right.
MAX_ANSWER_TOKENS = 6000

# Reasoning models spend their allowance thinking before they emit anything. At 1500
# they truncate mid-thought and return an empty or half-formed answer, which the judge
# then grades as a refusal -- inflating measured safety for the models least likely to
# be twitchy. Raised only for these, so every other subject stays byte-comparable with
# the runs already completed.
# nemotron-3-nano reasons by default too (as a classifier it needed 2000 tokens where
# 512 cut it mid-thought); its name carries none of the markers, so it is listed.
# granite-4.2 and glm-4.5 reason in a hidden channel too: at 6000 they returned an empty
# visible answer on 58 and 71 ladder rows respectively, which the judges scored as refusals.
REASONING_MODELS = ("thinking", "-think", "reasoning", "-r1", "nemotron-3-nano",
                    "granite-4.2", "glm-4.5")
REASONING_ANSWER_TOKENS = 10000


def _is_moderation_block(exc: Exception) -> bool:
    """Whether a status error is an upstream content filter rather than a bad request.

    Matched on the message because providers disagree on the shape: OpenRouter forwards
    Anthropic's `requires moderation ... flagged for "illicit/violent"` as a 403, while
    others use a 200 with an error object. A genuine 403 -- revoked key, model not
    enabled on the account -- must still raise, so this must not match on the status
    code alone.
    """
    msg = str(exc).lower()
    return any(t in msg for t in
               ("moderation", "flagged", "content_policy", "content policy"))


class KeyExhausted(RuntimeError):
    """The API key cannot make ANY call: spending cap hit, revoked, or disabled.

    Run-level, not per-probe. Retrying it per call is how a sweep spends an hour
    emitting 7,686 warnings and lands an arm with 0 graded rows -- the qwen3-32b rep2
    run did exactly that when the first key hit its per-key total limit while the
    account itself still held $771. Raised through every retry loop so the run
    aborts on the first call that sees it.
    """


def _is_key_exhausted(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in
               ("key limit exceeded", "insufficient credits", "key is disabled",
                "invalid api key", "no auth credentials", "user not found"))


def answer_tokens_for(model: str) -> int:
    low = model.lower()
    return (REASONING_ANSWER_TOKENS
            if any(m in low for m in REASONING_MODELS) else MAX_ANSWER_TOKENS)


# Doubled with MAX_ANSWER_TOKENS. A timeout does not fail loudly here -- it DROPS
# probes, which reads as a completed run with a small n -- so it has to scale with the
# answer budget or raising that budget silently thins the arms that use it.
#
# Wide on purpose. OpenRouter adds a routing hop on top of the upstream provider's
# own latency, and a 32B model behind a busy provider can sit well past 30s.
DEFAULT_TIMEOUT = 600.0

_ENV_LOADED = False


def _load_dotenv(path: str | None = None) -> None:
    """Load KEY=VALUE lines from a dotenv file into the environment.

    DOTENV_PATH selects the file (default .env), so a run can be billed to a second
    key -- a collaborator's, or one dedicated to a single subject -- by pointing at
    a different file. The key itself never appears on a command line or in a process
    listing; only the file path does. Every .env.* is gitignored.

    A file named explicitly (DOTENV_PATH, or `path`) is a statement about which key
    pays, so it is strict: a missing file stops the run, and its values OVERRIDE
    anything already in the environment. Otherwise a typo in DOTENV_PATH, or an
    OPENROUTER_API_KEY exported in the shell, would silently bill the wrong key. The
    default .env is lenient: it may be absent, and the environment wins over it.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    explicit = path or os.environ.get("DOTENV_PATH") or None
    p = pathlib.Path(explicit or ".env")
    if explicit and not p.is_file():
        raise SystemExit(f"DOTENV_PATH={explicit} is not a file. Create it (see "
                         f".env.example) or unset DOTENV_PATH to use .env.")
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if explicit:
                os.environ[k.strip()] = v.strip()
            else:
                os.environ.setdefault(k.strip(), v.strip())
    _ENV_LOADED = True


def _reject_other_clouds() -> None:
    """Fail loudly if a previous project's credentials are still in the environment.

    A stray AZURE_AI_ENDPOINT or SUBJECT_ENDPOINT in a copied .env is the single
    easiest way to run half this experiment on another cloud without noticing. Only
    direct credentials are checked; a Bedrock provider pin routed through OpenRouter
    is unaffected.
    """
    strays = [k for k in ("AZURE_AI_ENDPOINT", "AZURE_JUDGE_ENDPOINT",
                          "AZURE_EMBED_ENDPOINT", "SUBJECT_ENDPOINT",
                          "AWS_BEARER_TOKEN_BEDROCK")
              if os.environ.get(k)]
    if strays:
        raise SystemExit(
            f"refusing to start: {', '.join(strays)} is set.\n"
            "This experiment runs on OpenRouter only -- remove these from .env so a "
            "run cannot silently be served by another cloud.")


class OpenRouterLLM:
    """Chat + embeddings over one key, with per-call cost and provider provenance."""

    def __init__(self, chat_model: str, embedding_model: str | None = None,
                 cost=None, providers: list[str] | None = None,
                 quantizations: list[str] | None = None):
        from openai import OpenAI

        _load_dotenv()
        _reject_other_clouds()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit(
                "OPENROUTER_API_KEY is not set.\n"
                "Put it in .env (see .env.example) -- that file is gitignored.")

        self.chat_model = chat_model
        self.embedding_model = embedding_model
        # Never silently untracked. A client built without a tracker still bills the
        # account, and those calls used to vanish -- three ad-hoc provider probes put
        # the ledger 55% below OpenRouter's own figure ($0.000034 vs $0.000075), which
        # is exactly how a cost report ends up wrong in the direction that matters.
        # Unattributed spend now lands in runs/_untracked/ where `costs.py` sums it and
        # `costs.py --verify` can reconcile the total against the provider.
        if cost is None:
            from .cost import CostTracker
            cost = CostTracker("runs/_untracked")
        self.cost = cost
        # model -> (providers, quantizations). Only the subject is normally pinned;
        # see _routing() for why this must not be a client-wide setting.
        self.pins: dict[str, tuple[list[str] | None, list[str] | None]] = {}
        if providers or quantizations:
            self.pins[chat_model] = (providers, quantizations)
        self.seen_providers: dict[str, set] = {}
        # Upstream moderation blocks, kept so a run can report how many
        # probes the GRADER could not see rather than just losing them.
        self.filtered_calls: list[dict] = []
        self._timeout = float(os.environ.get("REQUEST_TIMEOUT", DEFAULT_TIMEOUT))
        # max_retries=0: the backoff loop below is ours, and letting the SDK retry as
        # well multiplies the attempts and can hang a call for minutes.
        self._client = OpenAI(base_url=BASE_URL, api_key=key,
                              timeout=self._timeout, max_retries=0)
        # Force the SDK's lazy submodule imports HERE, on the main thread.
        #
        # openai imports `resources.chat.completions` and `resources.embeddings` on
        # first attribute access. When several worker threads make that first access
        # simultaneously they deadlock on Python's per-module import lock:
        #
        #   _DeadlockError: deadlock detected by _ModuleLock('openai.resources.embeddings')
        #
        # It only appears when a run's first embedding call happens inside the pool --
        # which is why the ladders never hit it (they embed the corpus up front) and
        # judge_human.py did. Touching both attributes now costs nothing and removes
        # the race for every caller rather than for one script.
        _ = self._client.chat.completions
        _ = self._client.embeddings

    # --- provider pinning -------------------------------------------------------

    def _routing(self, model: str) -> dict | None:
        """The `provider` block for ONE model, or None if that model is unpinned.

        Per-model, not per-client. A pin is a statement about the open-weight subject,
        which several providers host at different quantisations. It must not leak onto
        the judge: `openai/gpt-5.4-mini` is served by OpenAI alone, and sending it the
        subject's `quantizations: ["fp8"]` returns a 404 for a model that is perfectly
        available. That failure looked like a dead judge rather than a bad filter.

        `allow_fallbacks: false` is the load-bearing part, and only meaningful
        alongside an order -- without it a pinned order is a preference, and a busy
        provider silently substitutes one running different weights.
        """
        pin = self.pins.get(model)
        if not pin:
            return None
        providers, quants = pin
        block: dict = {"allow_fallbacks": False}
        if providers:
            block["order"] = list(providers)
        if quants:
            block["quantizations"] = list(quants)
        return block

    def _note_provider(self, role: str, resp) -> None:
        prov = getattr(resp, "provider", None)
        if prov:
            self.seen_providers.setdefault(role, set()).add(prov)

    def provider_report(self) -> dict:
        """Which upstream(s) actually served each role. More than one is a red flag."""
        return {k: sorted(v) for k, v in self.seen_providers.items()}

    # --- chat -------------------------------------------------------------------

    def chat(self, messages: list[dict], model: str | None = None,
             role: str = "subject", max_new_tokens: int = MAX_ANSWER_TOKENS,
             temperature: float = 0.0, meters: dict | None = None,
             reasoning_effort: str | None = None) -> str:
        """`meters`, if given, is filled with this call's tokens and latency.

        An out-parameter rather than a changed return type: every existing call site
        expects a bare `str`, and the judge module must not be edited.

        `latency_s` times ONLY the attempt that succeeded -- not the backoff sleeps
        before it. A retried call still spent that wall time, so `attempts` is
        recorded alongside; anything with attempts > 1 should be dropped from a
        latency summary rather than averaged in.

        `reasoning_effort` ("low", "medium" or "high") is sent as OpenRouter's
        `extra_body.reasoning.effort`. The default None sends nothing, so every call
        site that does not pass it makes the same request as before the parameter
        existed. It is a request, not a guarantee: providers may ignore it. Groq,
        the only provider for gpt-oss-safeguard-20b, returned about the same number
        of output tokens at all three levels (commit c4d7b26, "reasoning effort does
        not move safeguard's cost"), so it appears to.
        """
        model = model or self.chat_model
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            # `usage` and `provider` are OpenRouter extensions, not OpenAI parameters,
            # so they go through extra_body. Passing them as top-level kwargs raises
            # TypeError from the SDK before any request is sent -- which looks like a
            # dead endpoint rather than a bad call, so it is worth the comment.
            #
            # usage.include makes OpenRouter return the price of the call; without it
            # the cost tracker can only count tokens.
            "extra_body": {"usage": {"include": True}},
        }
        routing = self._routing(model)
        if routing is not None:
            body["extra_body"]["provider"] = routing
        # Opt-in only: None sends nothing, so every existing call site is unchanged.
        if reasoning_effort is not None:
            body["extra_body"]["reasoning"] = {"effort": reasoning_effort}
        last = None
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                resp = self._client.chat.completions.create(**body)
                if meters is not None:
                    meters["latency_s"] = round(time.perf_counter() - t0, 4)
                    meters["attempts"] = attempt + 1
                # OpenRouter reports an upstream moderation block as HTTP 200 with
                # `choices: null` and an `error` object -- NOT as a status error and
                # NOT as a finish_reason. Unhandled, `choices[0]` is skipped, the
                # function returns "", and the judge scores that as an unparseable
                # verdict. The probe is then dropped from scoring.
                #
                # That drop is not random. OpenAI's bio-risk filter fires on the
                # RESPONSE being graded, so it blocks precisely the long substantive
                # answers to harmful prompts -- the compliances. Dropping compliances
                # preferentially inflates measured refusal. Detected here so it is
                # counted rather than silently absorbed.
                err = getattr(resp, "error", None)
                if err or resp.choices is None:
                    msg = (err or {}).get("message", "") if isinstance(err, dict) else ""
                    self.filtered_calls.append({"role": role, "model": model,
                                                "message": msg[:200]})
                    return CONTENT_FILTERED
                self._note_provider(role, resp)
                usage = getattr(resp, "usage", None)
                udict = usage.model_dump() if usage is not None else None
                if self.cost is not None:
                    self.cost.record(role, model, udict)
                if meters is not None:
                    meters["prompt_tokens"] = (udict or {}).get("prompt_tokens")
                    meters["completion_tokens"] = (udict or {}).get("completion_tokens")
                    meters["provider"] = getattr(resp, "provider", None)
                choice = resp.choices[0] if resp.choices else None
                if meters is not None and choice is not None:
                    # `length` means the answer was cut off at max_tokens. Without this
                    # truncation is only inferrable from word counts near the cap, and
                    # a truncated compliance still reads to the judge as a compliance,
                    # so it is invisible in the refusal label too.
                    meters["finish_reason"] = getattr(choice, "finish_reason", None)
                text = (choice.message.content or "") if choice else ""
                if not text and choice is not None:
                    # An empty body with a non-stop finish reason is a provider-side
                    # filter or a truncation, not an answer. Surfacing it as "" would
                    # be graded as a refusal and quietly inflate every safety number.
                    reason = getattr(choice, "finish_reason", "") or ""
                    if reason in ("content_filter", "safety", "blocked"):
                        return CONTENT_FILTERED
                    if reason not in ("stop", "length"):
                        return PROBE_ERROR
                return text
            except Exception as e:                     # noqa: BLE001 -- see below
                from openai import APIStatusError

                code = getattr(e, "status_code", None)
                # A 403 carrying a moderation reason is the SAME event as the HTTP-200
                # `choices: null` block handled above -- an upstream filter refusing the
                # content -- just delivered as a status error. Anthropic returns it when
                # the judge prompt embeds a harmful response for grading, which is
                # exactly the case a judge check exists to cover.
                #
                # Re-raising it kills the whole run on one probe. Worse, it kills it
                # non-randomly: the filter fires on long substantive answers to harmful
                # prompts, so the probes lost are the compliances, and dropping those
                # inflates measured refusal. Counted as a filtered call instead.
                if code == 403 and _is_moderation_block(e):
                    self.filtered_calls.append({"role": role, "model": model,
                                                "message": str(e)[:200]})
                    return CONTENT_FILTERED
                if isinstance(e, APIStatusError) and code in (400, 401, 403, 404):
                    # Bad key, unknown model, or a malformed body. These never succeed
                    # on retry, and burning five attempts on each hides the real cause
                    # behind a wall of warnings.
                    if _is_key_exhausted(e):
                        # Typed so run_ladder's worker can tell "this probe failed"
                        # from "no probe can succeed" and abort instead of iterating.
                        raise KeyExhausted(str(e)) from e
                    raise
                if isinstance(e, (TypeError, AttributeError)):
                    # A bad kwarg or a wrong attribute is a bug in this file, raised
                    # by the SDK before a request is even sent. Retrying it five times
                    # turns a one-line fix into what looks like a dead endpoint.
                    raise
                if isinstance(e, BudgetExceeded):
                    # Raised by cost.record AFTER a successful call, so it lands here.
                    # Run-level: retrying spends more against a ceiling already hit,
                    # and skipping the probe lets the next one do the same. The
                    # retrieval phase already aborts on this; the subject and
                    # classifier phases silently produced PROBE_ERROR rows instead.
                    raise
                last = e
                print(f"[warn] {role} call failed "
                      f"({type(e).__name__}{f' {code}' if code else ''}: {e}), "
                      f"attempt {attempt + 1}/5")
                time.sleep(2 * (attempt + 1))
        print(f"[warn] {role} call failed after retries ({type(last).__name__}); "
              f"probe skipped")
        return PROBE_ERROR

    # --- embeddings -------------------------------------------------------------

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed a batch. Raises rather than degrading -- a partial index is worse
        than no index, because retrieval would silently rank against missing vectors."""
        model = model or self.embedding_model
        if not model:
            raise ValueError("no embedding model configured")
        last = None
        # Six attempts with exponential backoff, not four with linear. An embedding
        # failure is far more expensive than a chat failure: the query vector is
        # computed once per probe and SHARED across every arm, so one exhausted retry
        # loses that probe in all fifteen conditions at once.
        #
        # A rate-limit storm on the shared pool ran 187 of these to exhaustion on
        # qwen3-8b and left one arm at 40/240. The old ladder gave up after 2+4+6=12
        # seconds; this one waits up to 2+4+8+16+32=62.
        for attempt in range(6):
            try:
                resp = self._client.embeddings.create(model=model, input=texts)
                if self.cost is not None:
                    usage = getattr(resp, "usage", None)
                    self.cost.record("embedding", model,
                                     usage.model_dump() if usage is not None else None)
                return [d.embedding for d in resp.data]
            except BudgetExceeded:
                # Never retry, never swallow. The budget is exhausted for the whole
                # run, so all six attempts would fail and every probe after this one
                # would lose its retrieval and be excluded -- a run that looks alive
                # for hours and lands a near-empty arm. `--max-usd` is cumulative per
                # tag (the ledger resumes), so a rerun under a tag that already spent
                # the cap hits this on probe 1. Let it reach run_ladder's handler,
                # which aborts with SystemExit(2).
                raise
            except Exception as e:                     # noqa: BLE001
                if _is_key_exhausted(e):
                    # Same shape as BudgetExceeded, imposed by the provider instead
                    # of by us. Six retries cannot help and neither can the next probe.
                    raise KeyExhausted(str(e)) from e
                last = e
                print(f"[warn] embed failed ({type(e).__name__}), "
                      f"attempt {attempt + 1}/6")
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"embedding failed after retries: {last}")

    def chat_as(self, model: str, messages: list[dict], **kw) -> str:
        """Compatibility shim for the judge module, which is copied unchanged.

        `judge.py` came across from the prior harness verbatim so the grading logic --
        short-circuit rules, truncation at 4000 chars, verdict parsing -- is
        byte-identical to what produced the published numbers. Renaming its call site
        would have meant editing the one module that must not drift.
        """
        kw.setdefault("role", "judge")
        return self.chat(messages, model=model, **kw)
