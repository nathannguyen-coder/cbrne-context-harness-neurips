#!/usr/bin/env python3
"""Fetch the source documents, chunk them, and report whether the corpus is usable.

No classification, no tagging beyond provenance. The prohibition/permission split is
deliberately NOT done here: it is a decomposition of an effect that has not been shown
to exist yet, and the categories should be derived from whichever chunks actually drive
a gain rather than guessed in advance.

Chunking matches the register of the corpus this replaces -- 40-160 words, one
normative statement per chunk, split on paragraphs and never across a section boundary.

The run ends in a COVERAGE GATE with three numbers, all of which have to be read before
spending anything on a ladder:

  domain balance   chunks per CBRNE domain against FORTRESS-CBRNE's EVAL composition.
                   Explosives is the largest test domain and has the thinnest normative
                   literature, so this is where a corpus fails if it fails.
  fetch losses     which sources did not come back. A silently short corpus looks like
                   a working one.
  dangling refs    chunks opening with an unresolved back-reference ("such activities
                   are prohibited"). Treaties are full of these, and a chunk that has
                   been severed from its antecedent is noise in the arm under test.

    python scripts/build_corpus.py
    python scripts/build_corpus.py --refresh     # re-download instead of using cache
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import sources as S  # noqa: E402

RAW = pathlib.Path("data/corpus/raw")
OUT = pathlib.Path("data/corpus")
MIN_WORDS, MAX_WORDS = 40, 160
UA = {"User-Agent": "answerable-policy/0.1 (research; contact via repo)"}

# A chunk starting with one of these has been cut off from what it refers to.
DANGLING = re.compile(
    r"^\s*(such|these|those|this|it|they|the said|the foregoing|the above)\b",
    re.IGNORECASE)


# --- fetching -------------------------------------------------------------------

def fetch(src: S.Source, refresh: bool = False) -> str | None:
    """Return the source's text, cached on disk. None if it could not be retrieved."""
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    cached = RAW / f"{src.id}.txt"
    if cached.exists() and not refresh:
        return cached.read_text()

    try:
        if src.fmt == "ecfr":
            title, part = src.url.split("/")
            # The API needs a real issue date; the literal string "current" 404s. Each
            # title has its own latest issue, so it is looked up per title and recorded
            # in the chunk provenance -- that date is what makes the corpus
            # reproducible, since the CFR changes underneath a fixed URL.
            date = _ecfr_latest_issue(title)
            url = (f"https://www.ecfr.gov/api/versioner/v1/full/{date}/"
                   f"title-{title}.xml?part={part}")
            r = requests.get(url, headers=UA, timeout=120)
            r.raise_for_status()
            text = _strip_xml(r.text)
            (RAW / f"{src.id}.issuedate").write_text(date)
        elif src.fmt == "pdf":
            r = requests.get(src.url, headers=UA, timeout=90)
            r.raise_for_status()
            tmp = RAW / f"{src.id}.pdf"
            tmp.write_bytes(r.content)
            text = _pdf_text(tmp)
        else:
            r = requests.get(src.url, headers=UA, timeout=60)
            r.raise_for_status()
            text = _html_text(r.text)
    except Exception as e:                                        # noqa: BLE001
        print(f"  [fail] {src.id}: {type(e).__name__}: {str(e)[:90]}")
        return None

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text.split()) < 200:
        print(f"  [fail] {src.id}: only {len(text.split())} words -- "
              f"likely a consent wall or JS-rendered page, not the document")
        return None
    cached.write_text(text)
    return text


def _ecfr_latest_issue(title: str) -> str:
    import requests

    r = requests.get("https://www.ecfr.gov/api/versioner/v1/titles",
                     headers=UA, timeout=60)
    r.raise_for_status()
    for t in r.json()["titles"]:
        if str(t["number"]) == str(title):
            return t["latest_issue_date"]
    raise RuntimeError(f"no such CFR title: {title}")


def _html_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    # Paragraph-per-line, so the chunker's paragraph logic has something to work with.
    parts = [t.get_text(" ", strip=True)
             for t in main.find_all(["p", "li", "h1", "h2", "h3", "h4"])]
    if not parts:
        parts = [main.get_text("\n", strip=True)]
    return "\n\n".join(p for p in parts if p)


def _pdf_text(path: pathlib.Path) -> str:
    import pdfplumber

    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
    return "\n\n".join(out)


def _strip_xml(xml: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(xml, "html.parser")
    parts = [t.get_text(" ", strip=True) for t in soup.find_all(["p", "head"])]
    return "\n\n".join(p for p in parts if p)


# --- chunking -------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])")


def chunk_text(text: str, src: S.Source) -> list[dict]:
    """Split into retrievable units, respecting the document's own structure.

    A treaty's natural unit is the Article and a regulation's is the section, not a
    word count. Merging a short paragraph forward across an `Article 3` heading welds
    two different obligations into one chunk, and the resulting text states a rule
    that neither article contains. So headings are detected first and act as hard
    barriers: short paragraphs merge forward only WITHIN a section, exactly as the
    constitution chunker never merged across a section boundary.

    The heading itself is carried into every chunk of its section, because "Article 4:
    Offences" is real retrieval signal and because a bare paragraph gives the model no
    way to see the text as situated law rather than a stray sentence.
    """
    sections = _split_sections(text)
    chunks: list[dict] = []
    dropped: list[dict] = []
    for heading, body in sections:
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        merged: list[str] = []
        for p in paras:
            if merged and len(merged[-1].split()) < MIN_WORDS:
                merged[-1] += " " + p
            else:
                merged.append(p)
        for para in merged:
            pieces = _split_long(para) if len(para.split()) > MAX_WORDS else [para]
            for piece in pieces:
                n = len(piece.split())
                if n < MIN_WORDS:
                    # A stub is a heading, a page number or a stray table row -- not a
                    # normative statement, and it would only dilute the top-k.
                    continue
                body_text = _clean(piece)
                text_out = f"[{heading}] {body_text}" if heading else body_text
                domain = (S.infer_domain(body_text, fallback="") if src.domain == S.MULTI
                          else src.domain)
                if not domain:
                    # A chunk from a multi-instrument compendium that mentions no
                    # weapon class at all. The UNODC volume is over half extradition,
                    # mutual-legal-assistance and ratification machinery -- 794 of
                    # 2,166 chunks on the first build, none of it policy about
                    # weapons. Left in, it is dead weight competing for top-5 slots
                    # against the clauses actually under test.
                    #
                    # Dropped rather than kept, and written to the audit file so the
                    # rule is inspectable instead of asserted. Single-domain sources
                    # are NEVER filtered this way: their domain comes from the
                    # registry, so Anthropic's usage policy keeps its general clauses.
                    dropped.append({**{k: v for k, v in
                                       (("source_id", src.id), ("section", heading))},
                                    "text": text_out})
                    continue
                chunks.append({
                    "chunk_id": f"{src.id}-{len(chunks) + 1:04d}",
                    "source_id": src.id,
                    "source_url": src.url,
                    "section": heading,
                    "domain": domain,
                    "kind": src.kind,
                    "redistributable": src.redistributable,
                    "text": text_out,
                    "n_words": n,
                    "dangling": bool(DANGLING.match(body_text)),
                })
    return chunks, dropped


# "Article 4", "ARTICLE IV", "§ 555.11", "Sec. 12" -- the structural spine of every
# legal document in this corpus.
_HEADING = re.compile(
    r"^\s*(?:(?:ARTICLE|Article)\s+[0-9IVXLC]+|§+\s*[0-9][0-9.\-]*|Sec\.\s*[0-9]+)"
    r"\s*[.:–-]?\s*(.{0,80})$")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)]. One entry with an empty heading if the doc has no structure."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    cur_head: str = ""
    cur: list[str] = []
    for ln in lines:
        m = _HEADING.match(ln.strip()) if len(ln.strip()) < 100 else None
        if m:
            if cur:
                sections.append((cur_head, cur))
            cur_head, cur = _clean(ln), []
        else:
            cur.append(ln)
    if cur:
        sections.append((cur_head, cur))
    return [(h, "\n".join(b)) for h, b in sections]


def _split_long(text: str) -> list[str]:
    out, cur = [], []
    for s in _SENT.split(text):
        cur.append(s)
        if len(" ".join(cur).split()) >= MAX_WORDS:
            out.append(" ".join(cur))
            cur = []
    if cur:
        tail = " ".join(cur)
        if out and len(tail.split()) < MIN_WORDS:
            out[-1] += " " + tail
        else:
            out.append(tail)
    return out


def _clean(t: str) -> str:
    # PDF table-of-contents dot leaders ("Introduction.........5") extract as one
    # unbroken token hundreds of characters long. Left in, they are meaningless to
    # embed and they break token-length assumptions that word counts cannot catch.
    t = re.sub(r"\.{4,}", " ", t)
    t = re.sub(r"[_\u2026\u00b7]{4,}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# --- main -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    all_dropped: list[dict] = []
    failed: list[S.Source] = []

    for src in S.ALL:
        print(f"[{src.kind:9s}] {src.id}")
        text = fetch(src, args.refresh)
        if text is None:
            failed.append(src)
            continue
        cs, dr = chunk_text(text, src)
        note = f"  ({len(dr)} dropped: no weapon class)" if dr else ""
        print(f"           {len(text.split()):>7,} words -> {len(cs):>4} chunks{note}")
        all_chunks.extend(cs)
        all_dropped.extend(dr)
        time.sleep(0.5)                       # be polite to the source hosts

    policy = [c for c in all_chunks if c["kind"] == "policy"]
    reference = [c for c in all_chunks if c["kind"] == "reference"]
    disclosure = [c for c in all_chunks if c["kind"] == "disclosure"]
    _write(OUT / "cbrne_policy.jsonl", policy)
    _write(OUT / "cbrne_reference.jsonl", reference)
    _write(OUT / "cbrne_disclosure.jsonl", disclosure)
    _write(OUT / "dropped_no_weapon_class.jsonl", all_dropped)

    print("\n" + "=" * 72)
    print("COVERAGE GATE")
    print("=" * 72)

    print(f"\npolicy chunks    {len(policy)}")
    print(f"reference chunks {len(reference)}")
    print(f"disclosure chunks {len(disclosure)}")

    print("\ndomain balance (policy channel) vs FORTRESS-CBRNE EVAL:")
    counts: dict[str, int] = {}
    for c in policy:
        counts[c["domain"]] = counts.get(c["domain"], 0) + 1
    for dom, n_rows in sorted(S.FORTRESS_EVAL_DOMAINS.items(),
                              key=lambda kv: -kv[1]):
        got = counts.get(dom, 0)
        share = got / len(policy) if policy else 0
        need = n_rows / sum(S.FORTRESS_EVAL_DOMAINS.values())
        flag = "  <-- THIN" if share < need / 2 else ""
        print(f"  {dom:32s} {got:>4} chunks ({share:5.1%})  "
              f"test set {n_rows:>3} rows ({need:5.1%}){flag}")
    extra = set(counts) - set(S.FORTRESS_EVAL_DOMAINS)
    for dom in sorted(extra):
        print(f"  {dom:32s} {counts[dom]:>4} chunks "
              f"({counts[dom]/len(policy):5.1%})  (not a test-set domain)")

    dangling = [c for c in policy if c["dangling"]]
    print(f"\ndangling references: {len(dangling)}/{len(policy)} "
          f"({len(dangling)/len(policy):.1%})" if policy else "\nno policy chunks")

    if failed:
        print(f"\nFETCH FAILURES ({len(failed)}) -- corpus is INCOMPLETE:")
        for s in failed:
            print(f"  {s.id:32s} {s.domain:30s} {s.url[:60]}")

    print("\nwritten:")
    print(f"  {OUT / 'cbrne_policy.jsonl'}")
    print(f"  {OUT / 'cbrne_reference.jsonl'}")
    print(f"  {OUT / 'cbrne_disclosure.jsonl'}")


def _write(path: pathlib.Path, chunks: list[dict]) -> None:
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks))


if __name__ == "__main__":
    main()
