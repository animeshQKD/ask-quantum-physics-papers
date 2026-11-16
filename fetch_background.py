"""
fetch_background.py - build the agent's background knowledge base (run it on your own computer).

    python fetch_background.py            check licences, download, read, write background/

Two parts, both allowed on a public Space:
  1. Further reading: the arXiv abstracts of the papers in corpus/papers.json (arXiv's metadata,
     abstracts included, is CC0), with title, authors and arXiv link from arXiv's own record. Papers
     without an arXiv version get title, authors, year and DOI only.
  2. Background: the review papers in corpus/background_sources.json, but only those whose published
     version is CC BY or CC0. The licence is read from OpenAlex and from Crossref (the version of
     record); a paper is used only when at least one of them says CC BY / CC0 and neither says
     otherwise (NC, ND, SA, or a publisher's own licence). "library" papers are checked against the
     arXiv licence in papers/resolved.json.

PDFs go to papers/background/ (local only, like all of papers/). A PDF a site refuses to send can be
saved there by hand as <key>.pdf; the next run uses it, after the same licence check.

  3. Links only: books and lecture notes whose licence does not allow reuse of their text, or only
     with conditions we do not take on ("links" in corpus/background_sources.json: Preskill's notes
     are copyright, Wilde's book is CC BY-NC-SA, de Wolf's notes are copyright). They are listed as
     further reading, with their arXiv abstract when there is one; their text is never used.

Written for the repo and the Space (all of it reusable, with credit):
  background/sources.json    every accepted paper: title, authors, DOI, licence and the evidence
  background/passages.jsonl  the text of the accepted papers, in passages (ids B1-p3-4, ...)
  background/reading.jsonl   further reading (parts 1 and 3)
  background/SOURCES.md      the credit list
  background/embeddings/     the passages' embeddings (float16), so the Space need not compute them
                             (skip with --no-embed; the app then embeds them at start-up)
"""
import argparse
import json
import os
import re
import sys
import urllib.parse

import fetch_papers as fp

FILES_VERSION = "2026-11-12"
LIST = os.path.join("corpus", "background_sources.json")
OUT = "background"
fp.GAP.setdefault("crossref", 1.0)
ACCEPT = {"cc-by", "cc0"}


# ---------------------------------------------------------------- licences (pure, tested offline)
def licence_name(s: str) -> str:
    """A licence URL or OpenAlex name -> 'cc-by', 'cc-by-nc', 'cc-by-nd', 'cc-by-sa', 'cc0' or 'other'."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    if "publicdomain/zero" in s or s in ("cc0", "public-domain"):
        return "cc0"
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)/", s)
    name = ("cc-" + m.group(1)) if m else s
    if name == "cc-by":
        return "cc-by"
    for part in ("nc", "nd", "sa"):
        if name.startswith("cc-by") and f"-{part}" in name:
            return f"cc-by-{part}"
    return "other"


def crossref_licences(message: dict) -> list[str]:
    """The licences Crossref gives for the version of record (text-mining licences are not about reuse)."""
    out = []
    for lic in message.get("license") or []:
        if (lic.get("content-version") or "vor") in ("vor", "unspecified"):
            out.append(lic.get("URL", ""))
    return out


def licence_verdict(openalex: str, crossref: list[str]) -> tuple[bool, str]:
    """(usable, evidence). Usable when some source says CC BY / CC0 and none says anything else
    (CC BY together with CC0 is usable: CC0 asks for less)."""
    names = [(f"OpenAlex: {openalex}", licence_name(openalex))] if openalex else []
    names += [(f"Crossref: {u}", licence_name(u)) for u in crossref if u]
    evidence = "; ".join(n for n, _ in names) or "no licence found"
    found = {n for _, n in names}
    if not found & ACCEPT:
        return False, evidence
    if found - ACCEPT:
        return False, evidence + " (conflicting licences)"
    return True, evidence


# ---------------------------------------------------------------- arXiv abstracts (pure)
def parse_abstracts(xml_bytes: bytes) -> dict:
    """arXiv id (no version) -> {title, authors, abstract, published}."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes)
    out = {}
    for e in root.findall("a:entry", fp.ATOM):
        aid_url = e.findtext("a:id", "", fp.ATOM) or ""
        if "/abs/" not in aid_url:
            continue
        aid = re.sub(r"v\d+$", "", aid_url.split("/abs/")[-1])
        out[aid] = {"title": " ".join((e.findtext("a:title", "", fp.ATOM) or "").split()),
                    "authors": [a.findtext("a:name", "", fp.ATOM) for a in e.findall("a:author", fp.ATOM)],
                    "abstract": " ".join((e.findtext("a:summary", "", fp.ATOM) or "").split()),
                    "published": (e.findtext("a:published", "", fp.ATOM) or "")[:10]}
    return out


def further_reading(net, resolved: list[dict], papers: list[dict]) -> list[dict]:
    """One record per listed paper: the arXiv abstract when there is an arXiv version."""
    ids = [r["arxiv"]["id"] for r in resolved if (r.get("arxiv") or {}).get("status") in ("ok", "review")]
    found = {}
    for i in range(0, len(ids), 20):
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(ids[i:i + 20]), "max_results": 20})
        st, body, _ = net.get(url, "arxiv")
        if st == 200:
            try:
                found.update(parse_abstracts(body))
            except Exception:  # noqa: BLE001 - a broken answer only costs abstracts
                pass
    by_key = {r["key"]: r for r in resolved}
    out = []
    for p in papers:
        r = by_key.get(p["key"], {})
        ax = r.get("arxiv") or {}
        oa = r.get("openalex") or {}
        rec = {"key": p["key"], "topic": p.get("topic", ""), "title": p["title"], "authors": p.get("authors", []),
               "year": p.get("year"), "venue": p.get("venue", ""), "doi": oa.get("doi", "") if oa.get("status") == "ok" else "",
               "arxiv_id": "", "url": "", "abstract": ""}
        if ax.get("status") in ("ok", "review") and ax.get("id") in found:
            a = found[ax["id"]]
            rec.update(arxiv_id=ax["id"], url=f"https://arxiv.org/abs/{ax['id']}", abstract=a["abstract"],
                       title=a["title"] or p["title"])
        elif rec["doi"]:
            rec["url"] = f"https://doi.org/{rec['doi']}"
        out.append(rec)
    return out


def link_reading(net, links: list[dict]) -> list[dict]:
    """Further-reading records for the link-only books and notes: their arXiv abstract when they have
    an arXiv id, and always the short description written in background_sources.json ("about")."""
    ids = [x["arxiv"] for x in links if x.get("arxiv")]
    found = {}
    if ids:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(ids), "max_results": len(ids)})
        st, body, _ = net.get(url, "arxiv")
        if st == 200:
            try:
                found = parse_abstracts(body)
            except Exception:  # noqa: BLE001
                found = {}
    out = []
    for x in links:
        a = found.get(x.get("arxiv", ""), {})
        out.append({"key": x["key"], "topic": x.get("topic", ""), "title": x["title"], "authors": x.get("authors", []),
                    "year": x.get("year"), "venue": x.get("venue", ""), "doi": "", "arxiv_id": x.get("arxiv", ""),
                    "url": x.get("url") or (f"https://arxiv.org/abs/{x['arxiv']}" if x.get("arxiv") else ""),
                    "abstract": a.get("abstract", ""), "about": x.get("about", ""), "note": x.get("note", "")})
    return out


# ---------------------------------------------------------------- one review paper
def resolve_source(net, src: dict) -> dict:
    """OpenAlex (by DOI, else by title) and Crossref for one review paper: the matching record, its
    open PDF link and the licence verdict."""
    rec = {"key": src["key"], "topic": src["topic"], "title": src["title"], "authors": src.get("authors", []),
           "year": src.get("year"), "venue": src.get("venue", ""), "doi": "", "pdf_url": "", "landing": "",
           "usable": False, "evidence": "", "problems": []}
    cand = None
    if src.get("doi"):
        url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(src["doi"])
        st, body, _ = net.get(url, "openalex")
        if st == 200:
            try:
                cands = fp.parse_openalex({"results": [json.loads(body)]})
            except ValueError:
                cands = []
            if cands and fp.accept(src, cands[0]["title"], cands[0]["authors"]) == "ok":
                cand = cands[0]
            elif cands:
                rec["problems"].append(f"the DOI {src['doi']} is '{cands[0]['title']}' (does not match)")
    if cand is None:
        oa = fp.resolve_openalex(net, src)
        if oa.get("status") == "ok":
            cand = oa
        else:
            rec["problems"].append(f"not found in OpenAlex ({oa.get('status')}: {oa.get('note', oa.get('title', ''))})")
            return rec
    rec.update(doi=cand.get("doi", ""), pdf_url=cand.get("journal_pdf", ""), landing=cand.get("landing", ""),
               authors=cand.get("authors") or rec["authors"], title=cand.get("title") or rec["title"])
    crossref = []
    if rec["doi"]:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(rec["doi"])
        if os.getenv("OPENALEX_MAILTO"):
            url += "?" + urllib.parse.urlencode({"mailto": os.environ["OPENALEX_MAILTO"]})
        st, body, _ = net.get(url, "crossref")
        if st == 200:
            try:
                crossref = crossref_licences(json.loads(body).get("message", {}))
            except ValueError:
                rec["problems"].append("Crossref answer was not JSON")
        else:
            rec["problems"].append(f"Crossref lookup failed (HTTP {st})")
    rec["usable"], rec["evidence"] = licence_verdict(cand.get("journal_license", ""), crossref)
    return rec


def library_source(src: dict, resolved: dict) -> dict:
    """A paper already downloaded from arXiv: usable when its arXiv licence is CC BY / CC0."""
    r = resolved.get(src["key"], {})
    ax = r.get("arxiv") or {}
    lic = ax.get("license", "")
    usable = licence_name(lic) in ACCEPT
    return {"key": src["key"], "topic": src["topic"], "title": ax.get("title") or r.get("title", ""),
            "authors": ax.get("authors", []), "year": r.get("year"), "venue": r.get("venue", ""),
            "doi": ((r.get("openalex") or {}).get("doi") or ""), "arxiv_id": ax.get("id", ""),
            "pdf_url": f"https://arxiv.org/pdf/{ax['id']}" if ax.get("id") else "", "landing": "",
            "usable": usable, "evidence": f"arXiv: {lic or 'no licence stated'}", "problems": [],
            "library": True}


# ---------------------------------------------------------------- reading the accepted papers
def read_passages(rec: dict, path: str, alias: str) -> list[dict]:
    import ingest
    paper = ingest.read_pdf(path, alias, key=rec["key"])
    return [{"id": p.pid, "source": rec["key"], "page": p.page, "kind": p.kind, "section": p.section,
             "label": p.label, "text": p.text} for p in paper.passages]


def credit_line(rec: dict) -> str:
    """The credit of one background paper, as background.Source.credit() shows it in the app: two
    authors by name, and the arXiv version when the text and its licence are the arXiv version's."""
    import background
    src = background.Source("", rec.get("key", ""), rec.get("title", ""), rec.get("authors") or [], rec.get("year"),
                            rec.get("venue", ""), rec.get("doi", ""), rec.get("arxiv_id") or "", licence_label(rec),
                            from_arxiv=(rec.get("evidence") or "").startswith("arXiv:"))
    return src.credit()


def licence_label(rec: dict) -> str:
    """'CC BY 4.0' from the licence URL in the evidence; a licence printed in the article itself
    ("licence" in background_sources.json) comes first (Quanta 2017 prints CC BY 3.0, Crossref says 4.0)."""
    if rec.get("licence_printed"):
        return rec["licence_printed"]
    for part in (rec.get("evidence") or "").split("; "):
        url = part.split(": ", 1)[-1].lower()
        m = re.search(r"creativecommons\.org/licenses/(by[a-z\-]*)/(\d\.\d)", url)
        if m:
            return f"CC {m.group(1).upper()} {m.group(2)}"
        if "publicdomain/zero" in url:
            return "CC0 1.0"
    return "CC BY" if "cc-by" in licences_in(rec.get("evidence", "")) else "CC0"


def licences_in(evidence: str) -> set:
    """The licence names in an evidence line ("OpenAlex: cc-by; Crossref: http://creativecommons...")."""
    return {licence_name(part.split(": ", 1)[-1]) for part in evidence.split("; ") if ": " in part}


def sources_md(accepted: list[dict], n_abstracts: int) -> str:
    L = ["# Background sources", "",
         "The agent's background knowledge comes from these papers, used under their Creative Commons",
         "licences. The text was extracted from the version linked in each line (the journal's PDF, or",
         "the arXiv version where the licence is the arXiv version's) and split into passages; nothing",
         "else was changed. The licence evidence for each paper is in sources.json.", ""]
    for i, r in enumerate(accepted, start=1):
        L.append(f"{i}. [B{i}] {credit_line(r)}")
    L += ["", f"Further reading: the arXiv abstracts of {n_abstracts} papers and notes (arXiv metadata, CC0 1.0), "
          "with links. Books and lecture notes listed there are linked only; their text is not used."]
    return "\n".join(L) + "\n"


def report(records: list[dict], n_abstracts: int, n_total: int) -> str:
    L = ["# Background knowledge base", "",
         f"Further reading: {n_abstracts} of {n_total} papers and notes have an arXiv abstract.", "",
         "| key | used | licence | licence evidence | PDF | problems |", "|---|---|---|---|---|---|"]
    for r in records:
        pdf = r.get("pdf_status", "")
        lic = licence_label(r) if r.get("usable") else ""
        L.append(f"| {r['key']} | {'yes' if r.get('used') else 'no'} | {lic} | {r.get('evidence', '')} | {pdf} | "
                 f"{'; '.join(r.get('problems', []))} |")
    manual = [r for r in records if r.get("usable") and not r.get("used")]
    if manual:
        L += ["", "## Save these yourself (licence is fine, the download was refused)", ""]
        for r in manual:
            link = f"https://doi.org/{r['doi']}" if r.get("doi") else r.get("landing", "")
            L.append(f"- {r['key']}: open {link}, download the PDF and save it as "
                     f"papers/background/{r['key']}.pdf, then run this script again")
    return "\n".join(L) + "\n"


def main(argv=None, net=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-embed", action="store_true", help="do not compute the embeddings (the app will)")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    papers_dir = fp.PAPERS_DIR
    net = net or fp.Net(os.path.join(papers_dir, "cache"))
    with open(LIST, encoding="utf-8") as f:
        listing = json.load(f)
    sources, links = listing["sources"], listing.get("links", [])
    resolved_list = []
    path = os.path.join(papers_dir, "resolved.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            resolved_list = json.load(f)
    resolved = {r["key"]: r for r in resolved_list}
    papers = fp.load_list()

    reading = further_reading(net, resolved_list, papers) + link_reading(net, links)
    records = []
    for src in sources:
        rec = library_source(src, resolved) if src.get("library") else resolve_source(net, src)
        if src.get("licence_printed"):
            rec["licence_printed"] = src["licence_printed"]
        if src.get("authors_printed"):                 # names as the article prints them (OpenAlex: "Akshata Shenoy H.")
            rec["authors"] = src["authors_printed"]
        print(f"  {rec['key']}: {'CC BY/CC0' if rec['usable'] else 'not usable'} ({rec['evidence']})")
        records.append(rec)

    os.makedirs(OUT, exist_ok=True)
    passages, accepted = [], []
    for rec in records:
        if not rec["usable"]:
            continue
        if rec.get("library"):
            pdf = os.path.join(papers_dir, "arxiv", f"{rec['key']}.pdf")
            ok, msg = os.path.isfile(pdf), "in papers/arxiv/"
        else:
            pdf = os.path.join(papers_dir, "background", f"{rec['key']}.pdf")
            if os.path.isfile(pdf):
                ok, msg = True, "found in papers/background/"
            elif rec["pdf_url"]:
                ok, msg = fp.save_pdf(net, rec["pdf_url"], pdf, "publisher")
            else:
                ok, msg = False, "no open PDF link"
        rec["pdf_status"] = msg
        if not ok:
            continue
        alias = f"B{len(accepted) + 1}"
        try:
            got = read_passages(rec, pdf, alias)
        except Exception as e:  # noqa: BLE001 - one unreadable PDF must not stop the build
            rec["problems"].append(f"could not read the PDF: {type(e).__name__}: {e}")
            continue
        rec["used"], rec["alias"], rec["passages"] = True, alias, len(got)
        passages += got
        accepted.append(rec)

    with open(os.path.join(OUT, "sources.json"), "w", encoding="utf-8") as f:
        json.dump([{**{k: r.get(k) for k in ("alias", "key", "topic", "title", "authors", "year", "venue", "doi",
                                             "arxiv_id", "evidence", "passages")}, "licence": licence_label(r)}
                   for r in accepted], f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "passages.jsonl"), "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "reading.jsonl"), "w", encoding="utf-8") as f:
        for r in reading:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    old_file = os.path.join(OUT, "abstracts.jsonl")          # its name before 11 Nov 2026
    if os.path.isfile(old_file):
        os.remove(old_file)
    n_abs = sum(1 for r in reading if r["abstract"])
    with open(os.path.join(OUT, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write(sources_md(accepted, n_abs))
    with open(os.path.join(papers_dir, "BACKGROUND_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report(records, n_abs, len(reading)))
    print(f"\nBackground: {len(accepted)} papers, {len(passages)} passages; further reading: {len(reading)} "
          f"papers and notes, {n_abs} with an abstract.\nWrote {OUT}/ and {papers_dir}/BACKGROUND_REPORT.md")
    if not a.no_embed:
        n = embed(OUT)
        print(f"Embeddings of {n} searchable passages and further-reading entries: {OUT}/{EMB_DIR}/")
    return 0


EMB_DIR = "embeddings"


def embed(folder: str, embeddings=None) -> int:
    """Compute the embeddings the app uses (index.DEFAULT_MODEL) and keep only the current files."""
    import shutil
    import background
    import index
    cache = os.path.join(folder, EMB_DIR)
    shutil.rmtree(cache, ignore_errors=True)            # files of an older model or older passages
    bg = background.load(folder)
    if bg is None:
        return 0
    bg.build_index(embeddings or index.FastEmbedEmbeddings(index.DEFAULT_MODEL))
    return len(bg.index.passages) + len(bg.reading_index.passages)


if __name__ == "__main__":
    sys.exit(main())
