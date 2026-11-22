"""Which models this experiment runs against, and why each one is on the list.

Resolution order, most specific first: CLI flag > environment variable > default here.

Subjects are chosen by MECHANISM, not by scale. The prior work found two distinct
response profiles, and a taxonomy needs at least one model from each -- three models
each behaving differently reads as scatter, not as a finding:

  framing-responsive  an empty policy wrapper (no clauses inside) raises refusal
                      ~25 points          -> mistral-small, qwen3-32b
  form-responsive     the empty wrapper does nothing; clause-SHAPED text raises
                      refusal instead     -> cohere-command-a

Every subject is gated before its full run. The ceiling is what matters here: above
~50% bare-baseline refusal there is no headroom left for a policy block to raise
anything, so the ladder cannot show an effect either way.

The prior work also imposed a 5% FLOOR, because it was measuring attacks and needed
room to observe a DROP. This experiment measures policy raising refusal, so a very
permissive model is a better subject, not a worse one -- the floor is dropped and
replaced by a gradeability check, since a near-zero reading can equally mean a broken
harness.
"""
from __future__ import annotations

import argparse
import os

from .llm import _load_dotenv

# Start here. Closest available build to the mistral-small the prior ladders used, so
# the numbers are at least interpretable against that work even though this is a fresh
# experiment with a new corpus.
DEFAULT_SUBJECT = "mistralai/mistral-small-3.2-24b-instruct"

# Was openai/gpt-5.4-mini, to stay comparable with the prior work. That is no longer
# possible: OpenAI's biological-risk filter rejects the JUDGE call on the response being
# graded, and it fires on long substantive answers to harmful prompts -- the
# compliances. A grader that cannot see compliances cannot measure refusal, and the loss
# is differential across arms, so it compresses the contrasts the ladder exists to
# measure.
#
# Measured on three probes gpt-5.4-mini could not grade, all confirmed compliances:
#
#   openai/gpt-5.4-mini          BLOCKED     OpenAI bio-risk filter
#   anthropic/claude-sonnet-5    BLOCKED     Anthropic's own filter
#   google/gemma-3-27b-it        graded      $0.08/$0.45 per M
#   x-ai/grok-4.3                graded      $1.25/$2.50
#   anthropic/claude-haiku-4.5   graded      ~$1.00/$5.00
#   qwen/qwen3-32b               graded, but only above ~512 output tokens
#   deepseek/deepseek-v4-flash   graded      $0.06/$0.13
#
# Both frontier labs' filters block their own models from grading CBRNE safety
# evaluations. That is worth stating in the writeup as an infrastructure finding.
#
# Gemma is the default on cost and on independence from the subject family (a mistral
# subject graded by a mistral judge invites self-preference). It is NOT validated
# against human labels yet -- that gates every refusal number here, exactly as the
# unvalidated gpt-5.4-mini did in the prior work.
DEFAULT_JUDGE = "google/gemma-3-27b-it"

# Same model the prior retrieval was tuned on. OpenRouter does not list embedding
# models in /api/v1/models, so availability of this id has to be confirmed live --
# scripts/preflight.py does that before anything else runs.
DEFAULT_EMBEDDING = "openai/text-embedding-3-small"

# Planned subjects, in order. Run one at a time; each is gated first.
SUBJECT_PLAN = [
    # Done.
    ("mistralai/mistral-small-3.2-24b-instruct", "24B, primary"),
    ("qwen/qwen3-32b", "32B, replication"),

    # Weaker and smaller, varying two things independently.
    #
    # FAMILY: if the finding is architectural rather than a quirk of one lineage, it
    # should appear in models that share no training pipeline with the first two.
    ("meta-llama/llama-3.1-8b-instruct", "8B, Meta lineage, cheapest"),
    ("microsoft/phi-4", "~14B, Microsoft lineage, heavily synthetic training mix"),
    # SCALE: same family as an existing subject, so capability is the only thing
    # varying. qwen3 8B against the 32B already run is a within-family ladder.
    ("qwen/qwen3-8b", "8B, scale control against qwen3-32b"),

    # Deliberately NOT gemma-*: it is the judge, and a model grading its own family
    # invites self-preference. Deliberately not ministral-*: same family as the primary.
]

# Upstream pin per subject: (provider, quantization).
#
# Not a performance choice. mistral-small-3.2 is served by three providers at two
# different quantisations (DeepInfra fp8, Parasail bf16, Venice fp8), and OpenRouter
# picks whichever is free -- so an unpinned run can change numerical precision between
# arms, inside one experiment, with nothing in the artifact saying so. Quantisation
# plausibly shifts behaviour on exactly the borderline prompts this study measures, so
# it must be held constant rather than left to routing.
#
# `allow_fallbacks: false` means a pinned provider being down FAILS the run rather
# than silently substituting another -- which is the behaviour we want.
#
# bf16 (Parasail) would be the better choice on precision grounds, but it returns 404
# "no endpoints found" on this account while its uptime reads 100%, which points at an
# account-level provider filter rather than an outage -- check the OpenRouter privacy /
# data-policy settings if bf16 is wanted later. fp8 via DeepInfra is what is actually
# reachable, so that is the pin.
#
# fp8 is a stated limitation, not a problem for validity: the experiment needs
# quantisation held CONSTANT across arms, not maximised. It only becomes a threat if
# results are compared against a run at a different precision.
#
# Slugs are lowercase (`deepinfra`, not `DeepInfra`). Re-check availability with:
#   curl -s https://openrouter.ai/api/v1/models/<model>/endpoints
PROVIDER_PINS = {
    # EVERY subject is pinned, including those served by a single endpoint today.
    # A pin on a one-endpoint model is a no-op now and a guard later: OpenRouter adds
    # providers without notice, and an unpinned model would start splitting traffic
    # mid-experiment with nothing in the artifact to show it had happened.
    #
    # Latency is why this matters. Unpinned llama-3.1-8b served 290 tokens in 2.2s on
    # Novita and 443 tokens in 28.4s on DeepInfra inside a single four-probe run --
    # an 8x swing in per-token latency from routing alone.
    #
    # DeepInfra wherever it is offered, so quantisation is fixed at the provider's
    # published precision rather than whatever a busy router picks. The one exception
    # is llama-3.1-70b, which DeepInfra offers but which is pinned to Bedrock (below).
    "mistralai/mistral-small-3.2-24b-instruct": ("deepinfra", "fp8"),
    # Moved from DeepInfra fp8: that endpoint fell into repetition loops on 4.4% of ladder
    # rows (37 of 240 in one arm); the same weights on Bedrock looped on 0 of 240. Bedrock
    # does not publish a precision, so this subject's quantisation is a stated unknown.
    # Bedrock is reached as an OpenRouter upstream, which llm.py allows; it is direct
    # Bedrock credentials that llm.py refuses.
    "meta-llama/llama-3.1-70b-instruct":        ("amazon-bedrock", None),
    "meta-llama/llama-3.1-8b-instruct":         ("deepinfra", "fp8"),
    "qwen/qwen3-32b":                           ("deepinfra", "fp8"),
    "nousresearch/hermes-3-llama-3.1-70b":      ("deepinfra", "fp8"),
    # Second safety-removed fine-tune on the SAME Llama-3.1-70B base, from a different
    # lab. With the instruct model and hermes this makes a three-way on one base --
    # official safety training, Nous's removal, Sao10k's removal -- so the alignment-
    # regime contrast no longer rests on a single uncensored model. Euryale and hermes
    # share DeepInfra fp8, but the instruct model has since moved to Bedrock at an
    # unpublished precision (above), so a comparison with it differs by provider as
    # well as by post-training. hermes-3 was later dropped from the report
    # (gen_appendix.DROP_SUBJECTS).
    "sao10k/l3.1-euryale-70b":                  ("deepinfra", "fp8"),
    # Expansion set, committed as a block of eight before any of them ran. Chosen for
    # lineage diversity (Cohere and Zhipu are new labs) and to test the pattern in the
    # first ten: both subjects where the clauses lost to the system prompt were 8B
    # models with a cautious baseline. DeepInfra where offered, at its published
    # precision; otherwise the sole or first-party provider, precision unverified.
    "mistralai/ministral-8b-2512":              ("mistral", None),
    "cohere/command-r7b-12-2024":               ("cohere", None),
    "qwen/qwen3-14b":                           ("deepinfra", "fp8"),
    "mistralai/mistral-nemo":                   ("deepinfra", "fp8"),
    # google/gemma-3-12b-it was considered and dropped before running: a sibling of
    # the gemma-3-27b judge, so the panel would be grading its own family's output.
    "z-ai/glm-4.5-air":                         ("novita", "bf16"),
    "meta-llama/llama-3.3-70b-instruct":        ("deepinfra", "fp8"),
    "qwen/qwen-2.5-72b-instruct":               ("deepinfra", "fp8"),
    "microsoft/phi-4":                          ("deepinfra", "bf16"),
    # granite-4.1-8b was withdrawn from OpenRouter mid-project (404, no endpoints).
    # 4.2-8b replaces it: same family and size, but a DIFFERENT MODEL, so its rows are
    # a new subject and not a re-run of the published granite numbers.
    "ibm-granite/granite-4.2-8b":               ("deepinfra", "bf16"),
    # Alibaba does not publish a quantisation for these two, so the provider is fixed
    # and the precision is a stated unknown. Pinning a quantisation we cannot verify
    # would 404 the model.
    "qwen/qwen3-8b":                            ("alibaba", None),
    "qwen/qwen3-30b-a3b-thinking-2507":         ("alibaba", None),
    # The veto classifier is a measured condition, so it is pinned like any subject.
    # Unpinned it split 1595 DeepInfra / 557 Novita inside one 2,152-probe run, which
    # puts provider routing inside the 10% of verdicts that flipped between two runs
    # at temperature 0. No quantisation: DeepInfra does not publish one for this model
    # and pinning an unverified precision 404s it.
    "nvidia/nemotron-3-nano-30b-a3b":           ("deepinfra", None),
    # Second classifier, for replication: purpose-built for policy-conditioned
    # classification, from a different lab, not a subject or judge. Groq is its only
    # provider; no quantisation published.
    "openai/gpt-oss-safeguard-20b":             ("groq", None),
}


def pin_for(model: str) -> tuple[list[str] | None, list[str] | None]:
    """(providers, quantizations) for a model, or (None, None) if unpinned.

    An unpinned subject is allowed but should be recorded as a limitation -- the run
    artifact's `providers` field is what a reader checks to see whether it was.
    """
    hit = PROVIDER_PINS.get(model)
    if not hit:
        return None, None
    provider, quant = hit
    # A None quantisation means the provider does not publish one, and it must come
    # back as None rather than [None]: `_routing` tests the list for truthiness, and
    # [None] is truthy, so the pin would ship `quantizations: [None]` and 404 a model
    # that is perfectly available.
    return [provider], ([quant] if quant else None)


def _env(name: str, fallback: str) -> str:
    _load_dotenv()
    return os.environ.get(name) or fallback


def subject_model() -> str:
    return _env("SUBJECT_MODEL", DEFAULT_SUBJECT)


def judge_model() -> str:
    return _env("JUDGE_MODEL", DEFAULT_JUDGE)


def embedding_model() -> str:
    return _env("EMBEDDING_MODEL", DEFAULT_EMBEDDING)


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--subject", default=subject_model(),
                    help=f"model under test (env SUBJECT_MODEL, default {DEFAULT_SUBJECT})")
    ap.add_argument("--judge", default=judge_model(),
                    help=f"refusal grader (env JUDGE_MODEL, default {DEFAULT_JUDGE})")
    ap.add_argument("--embedding", default=embedding_model(),
                    help=f"retrieval embeddings (env EMBEDDING_MODEL, default {DEFAULT_EMBEDDING})")
    ap.add_argument("--providers", nargs="*", default=None,
                    help="pin upstream provider order, e.g. --providers Mistral")
    ap.add_argument("--quantizations", nargs="*", default=None,
                    help="pin quantisation, e.g. --quantizations bf16")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="hard spend ceiling; the run aborts rather than crossing it")


def resolved(args: argparse.Namespace | None = None) -> dict:
    """What actually got used. Recorded in every run artifact."""
    return {"subject": getattr(args, "subject", None) or subject_model(),
            "judge": getattr(args, "judge", None) or judge_model(),
            "embedding": getattr(args, "embedding", None) or embedding_model(),
            "providers": getattr(args, "providers", None),
            "quantizations": getattr(args, "quantizations", None)}
