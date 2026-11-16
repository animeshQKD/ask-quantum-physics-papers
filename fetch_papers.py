"""
fetch_papers.py - download the corpus listed in corpus/papers.json (run it on your own computer).

    python fetch_papers.py                       resolve and download everything it can (resumable)
    python fetch_papers.py --only wootters1998 hill1997
    python fetch_papers.py --no-journal          arXiv versions only
    python fetch_papers.py --add-arxiv 1710.08495 --key navarrete2018    add a paper (e.g. your own)
    python fetch_papers.py --report              rewrite papers/REPORT.md from the saved results
    python fetch_papers.py --sort-downloads      rename the PDFs you saved in papers/manual/ (any name)

For every paper it
  1. finds the arXiv version: the arXiv id in the list is only a HINT and is checked (the title and
     the first author must match arXiv's record); without a hint, or when the hint does not match,
     arXiv is searched by title words and first author;
  2. reads the arXiv licence (OAI-PMH), because only some licences allow redistribution;
  3. asks OpenAlex for the published version: DOI, open-access status, licence, and a PDF link;
  4. downloads the arXiv PDF to papers/arxiv/<key>.pdf and an open-access journal PDF to
     papers/journal/<key>.pdf. A file is accepted only if it really is a PDF (starts with %PDF).

Everything ends up in papers/REPORT.md (what was downloaded and every match that needs a human look),
papers/DOWNLOAD_LIST.html (what you download yourself, with links) and papers/resolved.json.
Papers you download yourself go to papers/manual/ under any name; --sort-downloads renames them.

Politeness: at least 3 s between requests to arXiv (its API rules), 1 s to OpenAlex and 2 s to a
publisher; API answers are cached in papers/cache/, so a second run repeats no lookup. Set
OPENALEX_MAILTO in .env to use OpenAlex's faster "polite pool" (optional).

The PDFs are for local work only: papers/ is in .gitignore and never uploaded to the Space.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

import textnorm

FILES_VERSION = "2026-11-12"
PAPERS_DIR = os.getenv("PAPERS_DIR", "papers")
LIST = os.path.join("corpus", "papers.json")
EXTRA = os.path.join("corpus", "extra_papers.json")
USER_AGENT = "paper-research-agent-fetch/0.1 (+https://github.com/animeshQKD/paper-research-agent)"
OAI_ENDPOINTS = ("https://oaipmh.arxiv.org/oai", "https://export.arxiv.org/oai2")
GAP = {"arxiv": 3.1, "openalex": 1.0, "publisher": 2.0}
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
        "oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}
REDISTRIBUTABLE = ("creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/",
                   "creativecommons.org/publicdomain/zero/", "cc-by", "cc-by-sa", "public-domain", "cc0")


# ---------------------------------------------------------------- matching
def norm_title(t: str) -> str:
    t = textnorm.fold_accents(textnorm.clean(t)).lower()
    t = re.sub(r"\$[^$]*\$", " ", t)                 # inline LaTeX in arXiv titles
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def surname_in(surname: str, names: list[str]) -> bool:
    s = norm_title(surname).replace(" ", "")
    return bool(s) and any(s in norm_title(n).replace(" ", "") for n in names)


def accept(paper: dict, title: str, authors: list[str]) -> str:
    """'ok', 'review' (a human should look) or 'no'."""
    sim = similarity(paper["title"], title)
    author = surname_in(paper["authors"][0], authors) if paper.get("authors") else True
    if (sim >= 0.85 and author) or sim >= 0.97:
        return "ok"
    if sim >= 0.70 and author:
        return "review"
    return "no"


def venue_numbers(venue: str) -> list[str]:
    """'Phys. Rev. Lett. 88, 017901' -> ['88', '017901'] (volume and first page)."""
    return re.findall(r"[A-Z]?\d+", venue.split("(")[0])[:2]


def same_publication(paper: dict, rec: dict, doi: str = "") -> bool:
    """The arXiv record says it was published as this paper: the same DOI, or a journal reference
    with the same volume and first page (needed when the arXiv title differs from the journal title:
    'Introducing Quantum Discord' was published as 'Quantum Discord: A Measure of ...')."""
    if doi and rec.get("doi") and rec["doi"].lower() == doi.lower():
        return True
    nums = venue_numbers(paper.get("venue", ""))
    ref = re.sub(r"\s+", " ", rec.get("journal_ref", ""))
    return len(nums) == 2 and all(re.search(rf"(?<!\d){re.escape(n)}(?!\d)", ref) for n in nums)


def redistributable(license_url: str | None) -> bool:
    lic = (license_url or "").lower()
    return any(x in lic for x in REDISTRIBUTABLE) and "-nc" not in lic and "-nd" not in lic


# ---------------------------------------------------------------- parsing (pure, tested offline)
def parse_arxiv_atom(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("a:entry", ATOM):
        title = " ".join((e.findtext("a:title", "", ATOM) or "").split())
        aid_url = e.findtext("a:id", "", ATOM) or ""
        if title.lower() == "error" or "/abs/" not in aid_url:
            continue
        aid = aid_url.split("/abs/")[-1]
        base = re.sub(r"v\d+$", "", aid)
        out.append({"id": base, "version_id": aid, "title": title,
                    "authors": [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)],
                    "published": e.findtext("a:published", "", ATOM)[:10],
                    "doi": e.findtext("arxiv:doi", "", ATOM) or "",
                    "journal_ref": " ".join((e.findtext("arxiv:journal_ref", "", ATOM) or "").split())})
    return out


def parse_arxiv_license(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    lic = root.find(".//ax:license", ATOM)
    return (lic.text or "").strip() if lic is not None else ""


def parse_openalex(data: dict) -> list[dict]:
    out = []
    for w in data.get("results", []):
        locs = w.get("locations") or []
        oa = w.get("open_access") or {}
        journal_pdf, journal_license, landing = "", "", ""
        for loc in locs:
            src = ((loc.get("source") or {}).get("display_name") or "").lower()
            url = (loc.get("landing_page_url") or "") + (loc.get("pdf_url") or "")
            if "arxiv" in src or "arxiv.org" in url:
                continue
            if loc.get("is_oa") and loc.get("version") == "publishedVersion" and loc.get("pdf_url"):
                journal_pdf, journal_license = loc["pdf_url"], loc.get("license") or ""
                landing = loc.get("landing_page_url") or ""
                break
        prim = w.get("primary_location") or {}
        arxiv_id = ""
        for loc in locs:
            url = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
            m = re.search(r"arxiv\.org/(?:abs|pdf)/((?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7})|\d{4}\.\d{4,5})", url)
            if m:
                arxiv_id = m.group(1)
                break
        out.append({"id": w.get("id", ""), "arxiv_id": arxiv_id, "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "title": w.get("display_name") or w.get("title") or "",
                    "year": w.get("publication_year"),
                    "authors": [((a.get("author") or {}).get("display_name") or "") for a in w.get("authorships", [])],
                    "is_oa": bool(oa.get("is_oa")), "oa_status": oa.get("oa_status") or "",
                    "journal_pdf": journal_pdf, "journal_license": journal_license,
                    "landing": landing or prim.get("landing_page_url") or "",
                    "venue": ((prim.get("source") or {}).get("display_name") or "")})
    return out


def arxiv_search_url(paper: dict) -> str:
    words = [w for w in norm_title(paper["title"]).split() if len(w) >= 4][:6]
    terms = [f"ti:{w}" for w in words]
    if paper.get("authors"):
        last = norm_title(paper["authors"][0]).split()
        if last:
            terms.append(f"au:{last[-1]}")
    q = " AND ".join(terms)
    return "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": 0, "max_results": 5})


def openalex_url(paper: dict) -> str:
    y = int(paper.get("year") or 0)
    params = {"search": norm_title(paper["title"]), "per_page": 5,
              "select": "id,doi,display_name,publication_year,authorships,open_access,locations,primary_location"}
    if y:
        params["filter"] = f"publication_year:{y - 1}-{y + 1}"
    if os.getenv("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    if os.getenv("OPENALEX_API_KEY"):                   # optional free key: its own daily budget
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    return "https://api.openalex.org/works?" + urllib.parse.urlencode(params)


# ---------------------------------------------------------------- network
class Net:
    """HTTP GET with per-host pauses and a cache for API answers. get_fn is replaced in tests."""

    def __init__(self, cache_dir: str, get_fn=None):
        self.cache_dir = cache_dir
        self.last: dict[str, float] = {}
        self.get_fn = get_fn or self._httpx_get
        self._client = None

    retry_after: str | None = None

    def _httpx_get(self, url: str) -> tuple[int, bytes, str]:
        import httpx
        if self._client is None:
            self._client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60)
        r = self._client.get(url)
        ra = r.headers.get("retry-after", "")
        self.retry_after = ra if ra.isdigit() else None
        return r.status_code, r.content, r.headers.get("content-type", "")

    def _wait(self, kind: str):
        gap = GAP[kind] - (time.time() - self.last.get(kind, 0.0))
        if gap > 0 and self.get_fn == self._httpx_get:
            time.sleep(gap)
        self.last[kind] = time.time()

    def get(self, url: str, kind: str, cache: bool = True) -> tuple[int, bytes, str]:
        path = os.path.join(self.cache_dir, hashlib.sha1(url.encode()).hexdigest()[:20] + ".bin")
        if cache and os.path.isfile(path):
            with open(path, "rb") as f:
                return 200, f.read(), "cached"
        for attempt in (1, 2):
            self._wait(kind)
            try:
                status, body, ctype = self.get_fn(url)
            except Exception as e:  # noqa: BLE001 - one failed request must not stop the run
                return 0, str(e).encode()[:300], "error"
            if status not in (429, 503) or attempt == 2:
                break
            if self.get_fn == self._httpx_get:          # "too many requests": wait as asked, then once more
                time.sleep(min(60.0, float(self.retry_after or 20)))
        if cache and status == 200:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(body)
        return status, body, ctype


def save_pdf(net: Net, url: str, path: str, kind: str) -> tuple[bool, str]:
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return True, "already downloaded"
    status, body, ctype = net.get(url, kind, cache=False)
    if status != 200:
        return False, f"HTTP {status}" if status else f"network error: {body.decode(errors='ignore')[:120]}"
    if not body.startswith(b"%PDF"):
        return False, f"not a PDF (got {ctype or 'unknown type'}; the site may need a browser)"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(body)
    return True, f"{len(body) / 1e6:.1f} MB"


# ---------------------------------------------------------------- one paper
def resolve_arxiv(net: Net, paper: dict, oa: dict | None = None) -> dict:
    """The arXiv version: the hint from the list, else the arXiv copy OpenAlex knows, else a search."""
    tried = []
    oa = oa or {}
    hints = [(h, how) for h, how in (((paper.get("arxiv_hint") or "").strip(), "hint"), (oa.get("arxiv_id", ""), "openalex"))
             if h]
    for hint, how in hints:
        st, body, _ = net.get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": hint}), "arxiv")
        if st != 200:
            tried.append(f"{how} {hint}: lookup failed (HTTP {st})")
            continue
        for c in parse_arxiv_atom(body):
            verdict = accept(paper, c["title"], c["authors"])
            sim = round(similarity(paper["title"], c["title"]), 3)
            if verdict == "no" and paper.get("authors") and surname_in(paper["authors"][0], c["authors"]) \
                    and same_publication(paper, c, oa.get("doi", "")):
                return {**c, "status": "ok", "how": f"{how} (same DOI or journal reference)", "similarity": sim}
            if verdict != "no":
                return {**c, "status": verdict, "how": how, "similarity": sim}
            tried.append(f"{how} {hint} is '{c['title']}' (does not match)")
    st, body, _ = net.get(arxiv_search_url(paper), "arxiv")
    if st != 200:
        return {"status": "error", "note": f"arXiv search failed (HTTP {st})", "tried": tried}
    best, best_v = None, "no"
    for c in parse_arxiv_atom(body):
        v = accept(paper, c["title"], c["authors"])
        if v == "ok" or (v == "review" and best_v == "no"):
            best, best_v = c, v
            if v == "ok":
                break
    if best is None:
        return {"status": "none", "note": "no arXiv version found", "tried": tried}
    return {**best, "status": best_v, "how": "search", "similarity": round(similarity(paper["title"], best["title"]), 3),
            "tried": tried}


def resolve_openalex(net: Net, paper: dict) -> dict:
    st, body, _ = net.get(openalex_url(paper), "openalex")
    if st != 200:
        return {"status": "error", "note": f"OpenAlex failed (HTTP {st})"}
    try:
        cands = parse_openalex(json.loads(body))
    except ValueError:
        return {"status": "error", "note": "OpenAlex answer was not JSON"}
    for c in cands:
        v = accept(paper, c["title"], c["authors"])
        if v != "no":
            return {**c, "status": v, "similarity": round(similarity(paper["title"], c["title"]), 3)}
    return {"status": "none", "note": "not found in OpenAlex"}


def process(net: Net, paper: dict, journal: bool = True) -> dict:
    key = paper["key"]
    rec = {"key": key, "title": paper["title"], "year": paper.get("year"), "venue": paper.get("venue", ""),
           "topic": paper.get("topic", ""), "files": {}, "problems": []}
    oa = resolve_openalex(net, paper)
    rec["openalex"] = oa
    ax = resolve_arxiv(net, paper, oa if oa.get("status") in ("ok", "review") else None)
    rec["arxiv"] = ax
    if ax.get("status") in ("ok", "review"):
        ax["license"] = ""
        for oai in OAI_ENDPOINTS:                 # arXiv has moved its OAI-PMH service before
            st, body, _ = net.get(oai + "?" + urllib.parse.urlencode(
                {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{ax['id']}", "metadataPrefix": "arXiv"}), "arxiv")
            try:
                ax["license"] = parse_arxiv_license(body) if st == 200 else ""
            except ET.ParseError:
                ax["license"] = ""
            if ax["license"]:
                break
        path = os.path.join(PAPERS_DIR, "arxiv", f"{key}.pdf")
        ok, msg = save_pdf(net, f"https://arxiv.org/pdf/{ax['id']}", path, "arxiv")
        if ok:
            rec["files"]["arxiv"] = file_info(path)
        else:
            rec["problems"].append(f"arXiv PDF: {msg}")
        if ax["status"] == "review":
            rec["problems"].append(f"arXiv match needs a look: '{ax['title']}' (similarity {ax['similarity']})")
    if ax.get("tried"):
        rec["problems"] += [f"arXiv: {t}" for t in ax["tried"]]
    if oa.get("status") == "review":
        rec["problems"].append(f"OpenAlex match needs a look: '{oa['title']}' (similarity {oa['similarity']})")
    if journal and oa.get("journal_pdf"):
        path = os.path.join(PAPERS_DIR, "journal", f"{key}.pdf")
        ok, msg = save_pdf(net, oa["journal_pdf"], path, "publisher")
        if ok:
            rec["files"]["journal"] = file_info(path)
        else:
            rec["problems"].append(f"journal PDF: {msg}")
    manual = os.path.join(PAPERS_DIR, "manual", f"{key}.pdf")
    if os.path.isfile(manual):
        rec["files"]["manual"] = file_info(manual)
    return rec


def file_info(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    return {"path": path.replace("\\", "/"), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


# ---------------------------------------------------------------- report
def category(rec: dict) -> str:
    f = rec.get("files", {})
    oa = rec.get("openalex", {})
    if "journal" in f or "manual" in f:
        return "journal"
    if oa.get("journal_pdf"):
        return "oa_blocked"
    return "paid"


def report(records: list[dict]) -> str:
    n = len(records)
    have_ax = [r for r in records if "arxiv" in r.get("files", {})]
    have_j = [r for r in records if "journal" in r.get("files", {})]
    have_m = [r for r in records if "manual" in r.get("files", {})]
    blocked = [r for r in records if category(r) == "oa_blocked"]
    paid = [r for r in records if category(r) == "paid"]
    review = [r for r in records if r.get("problems")]
    redis = [r for r in records if redistributable((r.get("arxiv") or {}).get("license"))
             or redistributable((r.get("openalex") or {}).get("journal_license"))]
    doi = lambda r: (r.get("openalex") or {}).get("doi", "")                       # noqa: E731
    link = lambda r: f"https://doi.org/{doi(r)}" if doi(r) else ""                 # noqa: E731
    L = ["# Corpus download report\n",
         f"{n} papers in the list. arXiv PDFs: **{len(have_ax)}**. Open-access journal PDFs: **{len(have_j)}**. "
         f"Your manual downloads found: **{len(have_m)}**. Papers with no PDF at all yet: "
         f"**{sum(1 for r in records if not r.get('files'))}**.\n",
         f"Licence allows redistribution (CC BY / CC BY-SA / CC0; could go into the repository or the Space): "
         f"{', '.join(r['key'] for r in redis) or 'none'}.\n"]
    L += ["## 1. Published versions not downloaded (paywalled)\n",
          "Needed only where the last column says no: the others have an arXiv copy here, and their published "
          "version is optional. `papers/DOWNLOAD_LIST.html` lists the ones worth getting. Save each as "
          "`papers/manual/<key>.pdf`, then run `python fetch_papers.py --report`.\n",
          "| key | paper | link | arXiv version here? |", "|---|---|---|---|"]
    for r in paid:
        if "manual" in r.get("files", {}):
            continue
        L.append(f"| {r['key']} | {r['title'][:70]} ({r.get('venue', '')}) | {link(r) or 'search by title'} | "
                 f"{'yes' if 'arxiv' in r.get('files', {}) else 'no'} |")
    L += ["\n## 2. Open access, but the publisher refused the script\n",
          "Open the link in your browser and save the PDF as `papers/journal/<key>.pdf`.\n",
          "| key | paper | PDF link | licence |", "|---|---|---|---|"]
    for r in blocked:
        oa = r["openalex"]
        L.append(f"| {r['key']} | {r['title'][:70]} | {oa['journal_pdf']} | {oa.get('journal_license') or '?'} |")
    L += ["\n## 3. Needs a human look (I will check these too)\n", "| key | problem |", "|---|---|"]
    for r in review:
        for p in r["problems"]:
            L.append(f"| {r['key']} | {p.replace('|', '/')} |")
    L += ["\n## 4. Everything\n", "| key | year | arXiv | arXiv licence | journal OA | files |", "|---|---|---|---|---|---|"]
    for r in records:
        ax = r.get("arxiv") or {}
        oa = r.get("openalex") or {}
        lic = (ax.get("license") or "").replace("http://", "").replace("https://", "")
        L.append(f"| {r['key']} | {r.get('year')} | {ax.get('id', '') if ax.get('status') in ('ok', 'review') else '-'} "
                 f"| {lic or '-'} | {oa.get('oa_status') or '-'} | {', '.join(r.get('files', {})) or '-'} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- your own downloads
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)
# a short list of papers whose published version is worth having for testing the PDF reader on
# publisher layouts (one or two per journal family); every other published version is optional
LAYOUT_SAMPLES = ["wootters1998", "james2001", "horodecki2009", "horodecki1996sep", "henderson2001",
                  "bouwmeester1997", "yu2009", "fedrizzi2007", "saunders2010", "fuchs1999", "plenio2007",
                  "aolita2015", "devetak2005", "lo2005"]


def identify_pdf(path: str, records: list[dict], papers: list[dict]) -> tuple[str | None, str]:
    """Which paper of the list is this PDF? By a DOI printed in it, else by its title."""
    try:
        import pymupdf
        with pymupdf.open(path) as pdf:
            text = " ".join(pdf[i].get_text() for i in range(min(2, pdf.page_count)))
            text += " " + " ".join(str(v) for v in (pdf.metadata or {}).values())
    except Exception as e:  # noqa: BLE001
        return None, f"cannot read ({type(e).__name__})"
    by_doi = {(r.get("openalex") or {}).get("doi", "").lower(): r["key"] for r in records
              if (r.get("openalex") or {}).get("doi")}
    by_doi.update({aps_doi(p.get("venue", "")).lower(): p["key"] for p in papers
                   if aps_doi(p.get("venue", "")) and aps_doi(p.get("venue", "")).lower() not in by_doi})
    for d in _DOI.findall(text):
        d = d.rstrip(".,;)]").lower()
        if d in by_doi:
            return by_doi[d], f"DOI {d}"
    # publishers name their files after the DOI: PhysRevLett.67.661.pdf, 18.761271.pdf (IEEE),
    # 1-s2.0-037596019500214N-main.pdf (Elsevier, the DOI without its punctuation)
    stem = re.sub(r"[^a-z0-9]", "", os.path.splitext(os.path.basename(path))[0].lower())
    by_name = [(k, d) for d, k in by_doi.items()
               if len(re.sub(r"[^a-z0-9]", "", d.split("/", 1)[-1])) >= 8
               and re.sub(r"[^a-z0-9]", "", d.split("/", 1)[-1]) in stem]
    if len({k for k, _ in by_name}) == 1:
        return by_name[0][0], f"file name = DOI {by_name[0][1]}"
    flat = textnorm.match_form(text)
    found = {p["key"]: flat.find(textnorm.match_form(p["title"])) for p in papers if len(p["title"]) > 12}
    hits = sorted((pos, k) for k, pos in found.items() if pos >= 0)
    if len(hits) == 1:
        return hits[0][1], "title"
    # several titles occur ("Quantum cryptography" is in many papers): the paper's own title is printed
    # near the top of the first page, the others later in the text
    if hits and hits[0][0] < 400 and hits[1][0] >= 400:
        return hits[0][1], "title at the top of the first page"
    keys = [k for _, k in hits]
    return None, "no DOI of the list and no unique title found" if not hits else f"several titles match: {keys}"


def sort_downloads(records: list[dict], papers: list[dict]) -> list[str]:
    """Rename every PDF in papers/manual/ (any file name) to <key>.pdf; returns what was done."""
    folder = os.path.join(PAPERS_DIR, "manual")
    keys = {p["key"] for p in papers}
    done = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        path = os.path.join(folder, name)
        if not name.lower().endswith(".pdf") or os.path.splitext(name)[0] in keys:
            continue
        key, how = identify_pdf(path, records, papers)
        if key is None:
            done.append(f"{name}: not recognised ({how}); rename it to <key>.pdf yourself")
            continue
        target = os.path.join(folder, f"{key}.pdf")
        if os.path.exists(target):
            done.append(f"{name}: is {key} ({how}), but {key}.pdf is already there; left as it is")
            continue
        os.rename(path, target)
        done.append(f"{name} -> {key}.pdf ({how})")
    return done


_APS = {"Phys. Rev. Lett.": "PhysRevLett", "Phys. Rev. A": "PhysRevA", "Phys. Rev. B": "PhysRevB",
        "Phys. Rev. X": "PhysRevX", "Rev. Mod. Phys.": "RevModPhys", "Phys. Rev.": "PhysRev"}


def aps_doi(venue: str) -> str:
    """APS DOIs follow from journal, volume and page: 'Phys. Rev. Lett. 67, 661' -> 10.1103/PhysRevLett.67.661."""
    for name in sorted(_APS, key=len, reverse=True):
        m = re.match(rf"^{re.escape(name)}\s+(\d+),\s*([A-Z]?\d+)", venue)
        if m:
            return f"10.1103/{_APS[name]}.{m.group(1)}.{m.group(2)}"
    return ""


def download_list(records: list[dict]) -> str:
    """papers/DOWNLOAD_LIST.html: what to download yourself, with links, in three groups."""
    import html as h
    def link(r):
        oa = r.get("openalex") or {}
        doi = oa.get("doi") or aps_doi(r.get("venue", ""))
        if doi:
            return f"https://doi.org/{doi}"
        return "https://scholar.google.com/scholar?q=" + urllib.parse.quote(f'"{r["title"]}"')
    have = lambda r: r.get("files", {})                                         # noqa: E731
    none = [r for r in records if not have(r)]
    blocked = [r for r in records if "arxiv" in have(r) and category(r) == "oa_blocked"
               and not ({"journal", "manual"} & set(have(r)))]
    samples = [r for r in records if r["key"] in LAYOUT_SAMPLES and "arxiv" in have(r)
               and not ({"journal", "manual"} & set(have(r))) and r not in blocked]
    def table(rows, note):
        out = ["<table><tr><th>#</th><th>key</th><th>paper</th><th>journal</th><th>open</th><th>note</th></tr>"]
        for i, r in enumerate(rows, start=1):
            n = note(r)
            out.append(f"<tr><td>{i}</td><td><code>{h.escape(r['key'])}</code></td><td>{h.escape(r['title'])}</td>"
                       f"<td>{h.escape(r.get('venue', ''))} ({r.get('year', '')})</td>"
                       f"<td><a href=\"{h.escape(link(r))}\" target=\"_blank\">{'DOI' if 'doi.org' in link(r) else 'search'}</a></td>"
                       f"<td>{h.escape(n)}</td></tr>")
        return "\n".join(out) + "</table>"
    def note_none(r):
        oa = r.get("openalex") or {}
        if oa.get("journal_pdf"):
            return "open access: free in your browser"
        if any("does not match" in p for p in r.get("problems", [])):
            return "the next run will probably get its arXiv copy (matching fixed); else through your institute"
        if oa.get("status") == "error" and not aps_doi(r.get("venue", "")):
            return "OpenAlex did not answer (link is a search): run fetch_papers.py again"
        return "through your institute"
    css = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}table{border-collapse:collapse;"
           "width:100%;margin:8px 0 28px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:14px;vertical-align:top}"
           "th{background:#f3f3f3;text-align:left}code{font-size:13px}h2{margin-top:28px}")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Papers to download</title>
<style>{css}</style></head><body>
<h1>Papers to download yourself</h1>
<p>Click a link, download the PDF, and save it into <code>papers\\manual\\</code> with <b>any file name</b>.
Then run <code>python fetch_papers.py --sort-downloads</code>: it recognises each PDF by its DOI or title and
renames it to <code>&lt;key&gt;.pdf</code>. The PDFs stay on your computer (papers/ is never uploaded).</p>
<h2>1. No PDF yet ({len(none)}) - needed</h2>{table(none, note_none)}
<h2>2. Open access, the publisher refused the script ({len(blocked)}) - free, optional</h2>
<p>An arXiv copy is already there; the published version adds a journal layout to the tests.</p>{table(blocked, lambda r: (r.get("openalex") or {}).get("journal_license") or "")}
<h2>3. Published versions worth having for layout tests ({len(samples)}) - optional</h2>
<p>An arXiv copy is already there; these cover the main publisher layouts (APS, Elsevier, IOP, Nature, Science,
Optica, IEEE).</p>{table(samples, lambda r: "through your institute")}
</body></html>
"""


# ---------------------------------------------------------------- main
def load_list() -> list[dict]:
    with open(LIST, encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    if os.path.isfile(EXTRA):
        with open(EXTRA, encoding="utf-8") as f:
            papers += json.load(f)
    return papers


def add_arxiv(net: Net, aid: str, key: str) -> dict:
    st, body, _ = net.get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": aid}), "arxiv")
    found = parse_arxiv_atom(body) if st == 200 else []
    if not found:
        raise SystemExit(f"arXiv has no paper {aid} (HTTP {st}).")
    c = found[0]
    surname = c["authors"][0].split()[-1] if c["authors"] else "paper"
    key = key or re.sub(r"[^a-z0-9]", "", textnorm.fold_accents(surname).lower()) + c["published"][:4]
    entry = {"key": key, "topic": "own", "title": c["title"], "authors": [a.split()[-1] for a in c["authors"]],
             "year": int(c["published"][:4]), "venue": c["journal_ref"], "arxiv_hint": c["id"], "qi_tools": []}
    extra = []
    if os.path.isfile(EXTRA):
        with open(EXTRA, encoding="utf-8") as f:
            extra = json.load(f)
    if any(e["key"] == key for e in extra) or any(p["key"] == key for p in load_list()):
        raise SystemExit(f"The key {key} is already used; choose another with --key.")
    extra.append(entry)
    with open(EXTRA, "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=1, ensure_ascii=False)
    return entry


def main(argv=None, net: Net | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="keys to process")
    ap.add_argument("--no-journal", action="store_true", help="arXiv only, no journal PDFs")
    ap.add_argument("--report", action="store_true", help="only rewrite REPORT.md from resolved.json")
    ap.add_argument("--add-arxiv", default="", help="add an arXiv paper to corpus/extra_papers.json")
    ap.add_argument("--key", default="", help="key for --add-arxiv (default: surname + year)")
    ap.add_argument("--sort-downloads", action="store_true",
                    help="rename PDFs you saved in papers/manual/ (any name) to <key>.pdf, then rewrite the report")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv()
    net = net or Net(os.path.join(PAPERS_DIR, "cache"))
    resolved_path = os.path.join(PAPERS_DIR, "resolved.json")
    resolved = {}
    if os.path.isfile(resolved_path):
        with open(resolved_path, encoding="utf-8") as f:
            resolved = {r["key"]: r for r in json.load(f)}
    if a.add_arxiv:
        e = add_arxiv(net, a.add_arxiv, a.key)
        print(f"Added {e['key']}: {e['title']}")
        a.only = [e["key"]]
    papers = load_list()
    if a.sort_downloads:
        for line in sort_downloads([resolved[p["key"]] for p in papers if p["key"] in resolved], papers):
            print(line)
        a.report = True
    if not a.report:
        todo = [p for p in papers if a.only is None or p["key"] in a.only]
        for i, p in enumerate(todo, start=1):
            rec = process(net, p, journal=not a.no_journal)
            resolved[p["key"]] = rec
            files = ", ".join(rec["files"]) or "no PDF"
            print(f"[{i}/{len(todo)}] {p['key']}: {files}" + (f"  ({len(rec['problems'])} note(s))" if rec["problems"] else ""))
            os.makedirs(PAPERS_DIR, exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as f:        # saved after every paper: resumable
                json.dump([resolved[k] for k in (q["key"] for q in papers) if k in resolved], f, indent=1,
                          ensure_ascii=False)
    for r in resolved.values():                     # manual downloads appear without a new lookup
        manual = os.path.join(PAPERS_DIR, "manual", f"{r['key']}.pdf")
        if os.path.isfile(manual):
            r.setdefault("files", {})["manual"] = file_info(manual)
    records = [resolved[p["key"]] for p in papers if p["key"] in resolved]
    os.makedirs(PAPERS_DIR, exist_ok=True)
    with open(os.path.join(PAPERS_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report(records))
    with open(os.path.join(PAPERS_DIR, "DOWNLOAD_LIST.html"), "w", encoding="utf-8") as f:
        f.write(download_list(records))
    print(f"\nWrote {PAPERS_DIR}/REPORT.md and {PAPERS_DIR}/DOWNLOAD_LIST.html ({len(records)} papers).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
