"""
domain_physics.py - optional rules for physics papers (decision D4: a general core, physics as a pack).

The core (ingest, index, citations) never imports this file. index.Index uses it only when it is
created with domain="physics" (the app's default, from DOMAIN in .env); with DOMAIN=none the
same code serves any paper.

What it adds now:
  * SYNONYMS: abbreviations and their spelled-out forms, so a BM25 query for "CHSH" also finds
    "Clauser-Horne-Shimony-Holt" and a query for "PPT criterion" finds "partial transpose".
    Expansion is symmetric and only ADDS query terms; it never changes passage text.
  * UNITS: unit words that may follow a number. The Phase-1 number check uses them to compare
    "405 nm" with "405nm" and to know that "20 mW" is one quantity.
"""
FILES_VERSION = "2026-11-13a"

# each group: terms that mean the same thing in a query (lower case; multi-word terms allowed)
SYNONYM_GROUPS = [
    ["chsh", "clauser-horne-shimony-holt", "bell-chsh"],
    ["ppt", "positive partial transpose", "peres-horodecki", "partial transpose"],
    ["eof", "entanglement of formation"],
    ["qkd", "quantum key distribution"],
    ["qber", "quantum bit error rate"],
    ["spdc", "spontaneous parametric down-conversion", "parametric down-conversion"],
    ["hom", "hong-ou-mandel", "two-photon interference"],
    ["mle", "maximum likelihood", "maximum-likelihood"],
    ["hwp", "half-wave plate", "half wave plate"],
    ["qwp", "quarter-wave plate", "quarter wave plate"],
    ["esd", "entanglement sudden death", "sudden death"],
    ["povm", "positive operator-valued measure"],
    ["cptp", "completely positive trace-preserving"],
    ["epr", "einstein-podolsky-rosen"],
    ["ghz", "greenberger-horne-zeilinger"],
    ["bbo", "beta-barium borate", "barium borate"],
    ["ppktp", "periodically poled ktp"],
    ["snspd", "superconducting nanowire"],
    ["apd", "avalanche photodiode"],
    ["wcp", "weak coherent pulse", "weak coherent pulses"],
    ["mdi", "measurement-device-independent"],
    ["lhv", "local hidden variable", "local hidden-variable"],
    ["tomography", "state reconstruction"],
    ["discord", "quantum discord"],
    ["log-negativity", "logarithmic negativity"],
    ["fidelity", "uhlmann fidelity"],
    ["decoherence", "dephasing"],
    ["lindblad", "master equation", "gksl"],
]

SYNONYMS: dict[str, list[str]] = {}
for _group in SYNONYM_GROUPS:
    for _term in _group:
        SYNONYMS.setdefault(_term, [])
        SYNONYMS[_term] += [t for t in _group if t != _term and t not in SYNONYMS[_term]]

UNITS = ["nm", "um", "µm", "mm", "cm", "m", "km", "s", "ms", "µs", "us", "ns", "ps", "fs", "hz", "khz",
         "mhz", "ghz", "thz", "mw", "w", "µw", "uw", "nw", "db", "db/km", "k", "mk", "°c", "ev", "mev",
         "kev", "pairs/s", "counts/s", "cps", "%", "rad", "mrad", "deg", "°", "bits", "bit", "bits/pulse",
         "bits/s", "photons", "pulses"]


def expand_query(query: str) -> str:
    """The query plus the synonyms of every term it contains (for BM25 only)."""
    q = query.lower()
    extra = []
    for term, others in SYNONYMS.items():
        if (f" {term} " in f" {q} ") or (" " in term and term in q):
            extra += [o for o in others if o not in q and o not in extra]
    return query + (" " + " ".join(extra) if extra else "")
