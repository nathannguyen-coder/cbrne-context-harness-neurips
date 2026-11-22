"""The document registry: what goes in the corpus, and which CBRNE domain it covers.

Two things are tracked per source and both matter downstream.

`domain` drives the coverage gate. FORTRESS-CBRNE's EVAL split is Explosives 43,
Radiological/Nuclear 35, Chemical 26, Biological 15 -- so the biggest test domain is
Explosives, which has the *thinnest* international normative literature (there is no
Explosives Weapons Convention). If the corpus turns out bio/chem-heavy, the treatment
arm can fail on explosives for coverage reasons and read as the intervention failing.
The gate exists to catch that before a run, not after.

`redistributable` decides what may be committed. Treaties, US CFR, UN resolutions, WHO
and IAEA material are public; the two model-provider usage policies are copyrighted web
pages, fine to *use* as a research corpus but not to republish. Raw downloads stay in
the gitignored `data/corpus/raw/`; for non-redistributable sources only the provenance
and offsets are committed, never the text.
"""
from __future__ import annotations

from dataclasses import dataclass

CHEM = "Chemical"
BIO = "Biological"
NUC = "Radiological and Nuclear (WMD)"
EXP = "Explosives"
TERR = "Terrorism"
CROSS = "Cross-domain"

# Some sources are compendia covering several weapon classes in one file, so a single
# registry-level domain would be a lie. These get their domain assigned PER CHUNK by
# `infer_domain()`. That is only ever used for the coverage report -- it never touches
# what the model sees, and never decides which arm a chunk belongs to.
MULTI = "multi"

# Deliberately narrow and weapon-class specific. Ordered most-specific first: a chunk
# about radiological dispersal should read as nuclear, not as explosives, even though
# it says "device".
_DOMAIN_KEYWORDS = [
    (NUC, ("radioactive", "radiological", "nuclear material", "fissile", "isotope",
           "nuclear weapon", "nuclear facility", "atomic")),
    (BIO, ("biological agent", "toxin", "pathogen", "bacteriological", "microbial",
           "virus", "disease")),
    (CHEM, ("chemical weapon", "toxic chemical", "precursor", "chemical agent",
            "riot control agent")),
    (EXP, ("explosive", "bomb", "detonat", "lethal device", "incendiary")),
    (TERR, ("terrorist", "terrorism", "financing of terrorism")),
]


def infer_domain(text: str, fallback: str = CROSS) -> str:
    """Best-effort weapon-class label for one chunk of a multi-domain source.

    Highest keyword count wins; ties break toward the earlier (more specific) entry.
    A chunk mentioning nothing weapon-specific -- most procedural and ratification
    text -- falls back rather than being forced into a class it does not belong to.
    """
    low = text.lower()
    best, best_n = fallback, 0
    for domain, kws in _DOMAIN_KEYWORDS:
        n = sum(low.count(k) for k in kws)
        if n > best_n:
            best, best_n = domain, n
    return best


@dataclass
class Source:
    id: str
    url: str
    domain: str
    kind: str                 # "policy" (normative) | "reference" (factual)
    fmt: str                  # "html" | "pdf" | "ecfr"
    redistributable: bool
    note: str = ""


# --- the POLICY channel ---------------------------------------------------------
# Normative text: documents that say what may and may not be done. This is what goes
# in the retrieved policy block.
#
# INCLUSION CRITERION
#
#   An instrument issued by a state or an international body that states obligations,
#   prohibitions or permissions concerning CBRNE materials.
#
# Written down because the alternative is deciding document by document, and "we
# removed the sources we judged not to belong" is a researcher degree of freedom rather
# than a method. The condition claims to test retrieved GOVERNING LAW, so what counts
# as governing law has to be checkable by a reader instead of asserted.
#
# Three consequences, all of which the criterion decides rather than taste:
#
#   Anthropic's usage policy is OUT. Issued by a company, not a state or international
#   body. It was also the only non-redistributable source and the only text written as
#   instructions to a model, so it supplied essentially all of this arm's
#   assistant-register content -- meaning any advantage `cbrne` showed could be
#   attributed to 48 chunks of terms of service rather than to law. At 2.1% of the
#   corpus it was reaching 8.3% of retrievals.
#
#   EAR 15 CFR 734 and 744 are IN. US federal regulation on CBRNE-relevant transfers.
#   Their absence was the anomaly: build_boundary.py draws its ANSWER categories from
#   "CWC Art. VI; BWC prophylactic/protective; EAR fundamental research", so the
#   authored clauses cited an instrument the law arm could never retrieve.
#
#   IAEA Code of Conduct, INFCIRC/225 and the WHO framework stay IN. Non-binding, but
#   issued by international bodies and stating what may and may not be done. The
#   criterion turns on the issuer and the normative content, not on enforceability --
#   so the condition is accurately described as law AND international guidance.
#
# Honest limitation: this criterion was written after auditing the corpus, so it
# licenses changes already made. It is stated here so a reader can disagree with the
# rule rather than having to reverse-engineer it from the list.
POLICY: list[Source] = [
    # Chemical. The OPCW's per-article HTML pages are navigation shells -- they
    # yielded 16 chunks for a domain that is 22% of the test set. The full convention
    # PDF is 179 pages and carries Art. I (prohibition), Art. II (the general purpose
    # criterion), Art. VI (activities NOT prohibited) and the Annex on Chemicals.
    Source("cwc-full", "https://www.opcw.org/sites/default/files/documents/CWC/CWC_en.pdf",
           CHEM, "policy", "pdf", True,
           "Chemical Weapons Convention, full text incl. Annex on Chemicals"),

    # Biological
    Source("bwc-text", "https://front.un-arm.org/wp-content/uploads/2020/12/BWC-text-English.pdf",
           BIO, "policy", "pdf", True, "BWC Art. I and the peaceful-purposes clause"),
    # The URL here pointed at "Piloting of the global guidance framework ... in
    # Uganda, August 2023 - July 2024, Kampala" -- a country implementation report,
    # not the framework. 88 chunks of forewords, acknowledgements and pilot logistics
    # that reached the model on 4.1% of retrievals. The note always described the
    # framework, so this was a wrong link rather than a choice.
    #
    # iris.who.int serves the real document only from its bitstream API; the
    # /bitstream/handle/ path returns a 755-byte DSpace shell, the same interstitial
    # failure noted for UNSCR 1540 below.
    Source("who-life-sciences",
           "https://iris.who.int/server/api/core/bitstreams/"
           "4f495bfb-769a-458a-92e3-ea6d3e37d749/content",
           BIO, "policy", "pdf", True,
           "WHO Global guidance framework for the responsible use of the life "
           "sciences (2022), 216pp: dual-use research governance"),
    # The Federal Select Agent Program, all three parallel parts. These are NORMATIVE
    # (registration, security plans, transfer, restricted persons, incident reporting)
    # and not merely the agent list, so they belong in the policy channel -- which is
    # also what lifts Biological off the coverage floor. The BWC is a genuinely short
    # treaty (9 chunks) and cannot carry the domain alone.
    Source("cfr-42-73", "42/73", BIO, "policy", "ecfr", True,
           "HHS select agents and toxins: possession, use, transfer, security"),
    Source("cfr-9-121", "9/121", BIO, "policy", "ecfr", True,
           "APHIS select agents: animal pathogens"),
    Source("cfr-7-331", "7/331", BIO, "policy", "ecfr", True,
           "Plant protection select agents"),

    # Nuclear / radiological
    Source("iaea-code-of-conduct", "https://www-pub.iaea.org/MTCD/publications/PDF/Code-2004_web.pdf",
           NUC, "policy", "pdf", True,
           "IAEA Code of Conduct on the Safety and Security of Radioactive Sources"),
    Source("iaea-infcirc-225", "https://www-pub.iaea.org/MTCD/Publications/PDF/Pub1481_web.pdf",
           NUC, "policy", "pdf", True,
           "IAEA Nuclear Security Series 13 (INFCIRC/225 Rev.5): physical protection "
           "of nuclear material and facilities"),
    # iaea.org 403s any non-browser client, so INFCIRC/274 (CPPNM) and INFCIRC/140
    # (NPT) are unavailable. The US implementing regulations below carry the same
    # substance -- physical protection, material control and accounting, source
    # security -- as text that actually fetches.
    # DROPPED: 10 CFR 73. At 1,414 chunks it was 38% of the whole policy corpus and
    # pushed Nuclear to 52% share against a 29% test-set share. Most of it is guard
    # force qualification, alarm stations, barrier construction and search procedure at
    # licensed facilities -- site security operations, not statements about what may or
    # may not be assisted with. Size alone made it a dilution risk for every other
    # domain's top-5. 10 CFR 37 below covers source security at a fraction of the bulk.
    Source("cfr-10-37", "10/37", NUC, "policy", "ecfr", True,
           "10 CFR 37: security of category 1 and 2 quantities of radioactive material"),

    # Explosives -- the thin domain, and the largest in the test set. There is no
    # Explosives Weapons Convention, so coverage has to be assembled from a domestic
    # regulatory code and the counter-terrorism instruments.
    #
    # treaties.un.org's copy of the Terrorist Bombings Convention is a SCANNED PDF and
    # extracts to zero words; the UNODC compendium below carries the same instrument as
    # real text, which is why a 452-page book is preferred to a 12-page treaty.
    Source("cfr-27-555", "27/555", EXP, "policy", "ecfr", True,
           "27 CFR 555 -- commerce in explosives: licensing, storage, records"),
    Source("unodc-terrorism-instruments",
           "https://www.unodc.org/documents/terrorism/Publications/Int_Instruments_Prevention_and_Suppression_Int_Terrorism/Publication_-_English_-_08-25503_text.pdf",
           MULTI, "policy", "pdf", True,
           "UNODC compendium: Terrorist Bombings, Nuclear Terrorism, Plastic "
           "Explosives Marking and related instruments -- domain assigned per chunk"),
    # DROPPED: UNSCR 1540 via digitallibrary.un.org serves a 22KB file with no /Root
    # object -- an interstitial, not the resolution. Its substance (non-state actor
    # acquisition of NBC weapons and means of delivery) is covered by the UNODC
    # compendium above, so this is a coverage loss of roughly zero rather than a gap.

    # Export controls. build_boundary.py draws its ANSWER categories from "CWC Art. VI;
    # BWC prophylactic/protective; EAR fundamental research" -- so the authored clauses
    # cite an instrument the law corpus did not contain, and the `cbrne` arm could never
    # retrieve it. That asymmetry favoured the clauses on a contrast the paper turns on.
    #
    # 744 is the CBRNE part: end-use controls on chemical, biological, nuclear and
    # missile applications. 734 carries the fundamental-research exclusion the clauses
    # name directly.
    Source("cfr-15-744", "15/744", MULTI, "policy", "ecfr", True,
           "15 CFR 744: EAR end-use controls, chemical/biological/nuclear/missile"),
    Source("cfr-15-734", "15/734", MULTI, "policy", "ecfr", True,
           "15 CFR 734: EAR scope, incl. the fundamental research exclusion"),

    # REMOVED: anthropic-usage-policy. A company's terms of service is not law, and the
    # condition claims to test retrieved GOVERNING LAW. It was also the only
    # non-redistributable source and the only one written as instructions to a model,
    # so it supplied essentially all of the arm's assistant-register text -- meaning any
    # advantage `cbrne` showed could be attributed to 48 chunks of ToS rather than to
    # law. At 2.1% of the corpus it was reaching 8.3% of retrievals, four times its
    # share.
    # DROPPED: openai.com returns 403 to any non-browser client. Worth revisiting with
    # a fetched-by-hand copy if a second model-policy voice is wanted; Anthropic's AUP
    # already supplies that register, so this is not blocking.
]

# --- the REFERENCE channel ------------------------------------------------------
# Factual: what is controlled, and what the ordinary science says. Retrieved into a
# separate block, and only in the exploratory `cbrne+ref` arm.
# --- the DISCLOSURE channel -----------------------------------------------------
# The one corner of the CBRNE regime written about INFORMATION rather than about
# facilities, licences and state obligations. Export control asks exactly the question
# the model faces -- may this technical content be communicated, and to whom -- and it
# carries the permissive side explicitly: published results, fundamental research and
# catalog course instruction are carved OUT of control.
#
# This is the direct test of the EVAL finding. The raw policy corpus failed because
# 0.8% of what it retrieved addressed an assistant, against 85.2% for the constitution.
# If register is the cause, a corpus about disclosure should do better than one about
# fencing a reactor -- and if it does not, the problem is deeper than register.
#
# Anthropic's usage policy is included rather than held out. It is the only genuinely
# assistant-facing source available, and excluding it because it over-refused in the
# main run would be selecting the corpus on the outcome.
DISCLOSURE: list[Source] = [
    Source("ear-734-scope", "15/734", CROSS, "disclosure", "ecfr", True,
           "EAR scope: published information, fundamental research, publicly "
           "available -- the carve-outs that say what MAY be shared"),
    Source("itar-120-defs", "22/120", CROSS, "disclosure", "ecfr", True,
           "ITAR definitions incl. public domain"),
    Source("anthropic-usage-policy-disc", "https://www.anthropic.com/legal/aup",
           CROSS, "disclosure", "html", False,
           "the only assistant-facing source in the set; prohibition-only"),
]

REFERENCE: list[Source] = [
    Source("ag-cw-precursors", "https://www.dfat.gov.au/publications/minisite/theaustraliagroupnet/site/en/documents/common-control-lists/Chemical-Weapons-Precursors-Control-List.pdf",
           CHEM, "reference", "pdf", True, "Australia Group CW precursor control list"),
]

ALL = POLICY + DISCLOSURE + REFERENCE


def by_domain(sources: list[Source]) -> dict[str, list[Source]]:
    out: dict[str, list[Source]] = {}
    for s in sources:
        out.setdefault(s.domain, []).append(s)
    return out


# FORTRESS-CBRNE EVAL composition. The corpus is checked against this, not against an
# even split -- a corpus balanced across domains would still be wrong for this test set.
FORTRESS_EVAL_DOMAINS = {
    EXP: 43,
    NUC: 35,
    CHEM: 26,
    BIO: 15,
    TERR: 1,
}
