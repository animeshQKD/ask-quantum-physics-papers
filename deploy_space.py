"""
deploy_space.py - publish the web app (app.py) as a Hugging Face Space.

    python deploy_space.py --repo YOUR_NAME/ask-quantum-physics-papers --dry-run   # show what would happen
    python deploy_space.py --repo YOUR_NAME/ask-quantum-physics-papers             # upload the app
    python deploy_space.py --repo YOUR_NAME/ask-quantum-physics-papers --settings  # + Space secrets/variables

Safety:
  * Only the files in APP_FILES / DATA_FILES and the background knowledge base (background/: the
    passages of CC BY / CC0 review papers, their credits, further-reading links and precomputed
    embeddings; a paper with any other licence stops the upload) are uploaded (a whitelist). .env, logs/, papers/ (the
    local corpus, which may not be redistributed), data/sessions/ and the tests can never reach the Space.
  * Every uploaded file is scanned for API keys first; if one is found, nothing is uploaded.
  * --settings sends the keys from your local .env to the Space as SECRETS (encrypted, never in a
    file, not visible to visitors). OPENAI_API_KEY is never sent: the public app uses free models only.
  * The README (Space card) and requirements.txt are generated from THIS Python environment, so the
    Space runs the same Gradio, Python and package versions that passed the tests here.

Uploading needs a Hugging Face token with WRITE access: run `hf auth login` once (or set
HF_DEPLOY_TOKEN). The inference token (HF_TOKEN) is NEVER used for uploading, even when it is set in the
shell: huggingface_hub would otherwise prefer it silently over `hf auth login`, and an inference token
usually cannot write (2026-11-04: "403 Forbidden ... pass create_pr=1" right after a fresh login).
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from importlib import metadata

APP_FILES = ["app.py", "agent.py", "background.py", "checker.py", "mathview.py", "ingest.py", "layout.py", "index.py",
             "citations.py", "textnorm.py", "domain_physics.py", "llm.py", "session.py", "figures.py", "references.py"]
DATA_FILES: list[str] = []           # example papers: only ones whose licence allows redistribution (CC-BY)
# the background knowledge base built by fetch_background.py (background.py reads it); every paper in it
# must be CC BY or CC0 (checked again here), and its credit list goes with it
BACKGROUND_DIR = "background"
BACKGROUND_FILES = ["sources.json", "passages.jsonl", "reading.jsonl", "SOURCES.md"]

# packages the app imports, pinned to the versions tested here (gradio itself comes from the Space's
# sdk_version). NOT pinned: pydantic - the Space installs gradio[oauth,mcp], whose 'mcp' extra limits
# pydantic (e.g. <= 2.12.5 for gradio 6.27), so a pin copied from this computer can make the build fail.
PACKAGES = ["python-dotenv", "langgraph", "langchain-core", "langchain-openai", "langchain-groq",
            "langchain-google-genai", "pymupdf", "fastembed", "qdrant-client", "langchain-qdrant"]
UNPINNED = ["pydantic"]
# what the Space's build adds to requirements.txt (from its build log)
SPACE_EXTRAS = ['gradio[oauth,mcp]=={gradio}', "uvicorn>=0.14.0", "websockets>=10.4", "spaces"]

SECRET_KEYS = ["HF_TOKEN", "GROQ_API_KEY", "GOOGLE_API_KEY"]        # never OPENAI_API_KEY
VARIABLES = {                                                        # public settings (not secret)
    "USE_HF": "1",
    "HF_MODEL": None,                 # None: copied from your .env if set there
    "HF_FALLBACK_MODEL": None,
    "GROQ_MODEL": None,
    "GROQ_FALLBACK_MODEL": None,
    "GEMINI_MODEL": None,
    "PROVIDER_ORDER": "hf,hf_backup,groq,groq_backup,gemini",
    "DAILY_HF_CALLS": "150",
    "QUESTIONS_PER_HOUR": "40",
    "DOMAIN": "none",
    "EMBED_MODEL": None,
    "MAX_PDF_MB": "20",
    "MAX_PDF_PAGES": "100",
}

# a key starts a word: "ask-quantum-physics-papers" in the GitHub link is no OpenAI key "sk-..." (16 Nov 2026)
KEY_PATTERNS = [r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}", r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}",
                r"(?<![A-Za-z0-9])gsk_[A-Za-z0-9]{20,}", r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}"]


def find_keys(paths: list[str]) -> list[str]:
    """Files that contain something that looks like an API key."""
    bad = []
    for p in paths:
        if p.endswith(".npy"):            # embeddings (numbers only) written by fetch_background.py
            continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if any(re.search(k, text) for k in KEY_PATTERNS):
            bad.append(p)
    return bad


def open_licence(label: str) -> bool:
    """'CC BY 4.0', 'CC BY', 'CC0 1.0': yes; anything with NC, ND, SA, or no licence: no."""
    return re.fullmatch(r"CC BY(?: \d\.\d)?|CC0(?: 1\.0)?", (label or "").strip()) is not None


def background_files(folder: str | None = None) -> tuple[list[str], int]:
    """The background files to upload and the number of papers in it; ([], 0) when it was not built.
    Stops (nothing is uploaded) if a paper in it is not CC BY / CC0."""
    import json
    folder = folder or BACKGROUND_DIR
    path = os.path.join(folder, "sources.json")
    if not os.path.isfile(path):
        return [], 0
    with open(path, encoding="utf-8") as f:
        sources = json.load(f)
    if sources and not any("licence" in s for s in sources):
        raise SystemExit(f"STOP: {path} was written by an older fetch_background.py (no licence field); "
                         "run fetch_background.py again. Nothing was uploaded.")
    bad = [f"{s.get('key')} ({s.get('licence') or 'no licence'})" for s in sources if not open_licence(s.get("licence"))]
    if bad:
        raise SystemExit("STOP: background papers without a CC BY / CC0 licence; nothing was uploaded: " + ", ".join(bad))
    missing = [n for n in BACKGROUND_FILES if not os.path.isfile(os.path.join(folder, n))]
    if missing:          # the credit list is what the CC BY licences ask for: no background without it
        raise SystemExit(f"STOP: {folder}/ lacks {', '.join(missing)}; run fetch_background.py again. "
                         "Nothing was uploaded.")
    files = [os.path.join(folder, n) for n in BACKGROUND_FILES]
    emb = os.path.join(folder, "embeddings")
    if os.path.isdir(emb):
        files += sorted(os.path.join(emb, n) for n in os.listdir(emb) if n.endswith(".npy"))
    return files, len(sources)


def requirements() -> str:
    lines = []
    for pkg in PACKAGES:
        try:
            lines.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(pkg)
    lines += UNPINNED
    return "\n".join(lines) + "\n"


def check_install(req_file: str) -> str | None:
    """Ask pip whether the Space's build would succeed: the Space's exact install command (requirements.txt
    plus gradio[oauth,mcp] and the extras its build adds), resolved WITHOUT installing anything. It resolves
    for this computer: version conflicts (like pydantic vs gradio's 'mcp' extra) do not depend on the
    platform, while resolving for Linux from Windows is unreliable (pip would apply this computer's
    'Windows only' package conditions). Returns None if it resolves, else pip's explanation."""
    import subprocess
    import gradio
    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--disable-pip-version-check",
           "-r", req_file] + [x.format(gradio=gradio.__version__) for x in SPACE_EXTRAS]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"could not run pip: {e}"
    if out.returncode == 0:
        return None
    lines = [x for x in ((out.stdout or "") + "\n" + (out.stderr or "")).splitlines() if x.strip()]
    start = next((i for i, x in enumerate(lines) if x.startswith("ERROR")), max(0, len(lines) - 25))
    return "\n".join(lines[start:start + 30])


EMBED_DEFAULT = "snowflake/snowflake-arctic-embed-s"      # = index.DEFAULT_MODEL (a test checks it)
HUB_IDS = {"snowflake/snowflake-arctic-embed-s": "Snowflake/snowflake-arctic-embed-s"}   # fastembed name -> Hub id
SHORT_DESCRIPTION = "Cited, checked answers from quantum physics papers"   # <= 60 characters
TAGS = ["research-papers", "pdf", "citations", "retrieval", "hybrid-search", "qdrant", "embeddings",
        "physics", "quantum-information", "rag", "agent", "langgraph"]


def model_ids(env: dict) -> list[str]:
    """Hub model ids the Space uses (HF_MODEL, HF_FALLBACK_MODEL without the ':provider' suffix). Listing
    them puts the Space on each model's page under 'Spaces using this model'."""
    out = []
    for k in ("HF_MODEL", "HF_FALLBACK_MODEL"):
        m = (env.get(k) or "").split(":")[0].strip()
        if m.count("/") == 1 and m not in out:
            out.append(m)
    return out


def space_host(repo: str) -> str:
    """The Space's own web address: 'Animesh1986/my_app' -> 'animesh1986-my-app.hf.space'."""
    return re.sub(r"[^a-z0-9-]", "-", repo.lower().replace("/", "-")) + ".hf.space"


def readme(title: str, env: dict | None = None, thumbnail: str = "", github: str = "", repo: str = "",
           n_background: int = 0) -> str:
    """The Space card. Every sentence in it must be true of the version being deployed."""
    import gradio
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    # the Hub models the deployed version uses: the embedding model and the Hugging Face language models
    emb = (env or {}).get("EMBED_MODEL") or VARIABLES.get("EMBED_MODEL") or EMBED_DEFAULT
    models = [HUB_IDS.get(emb, emb)] + [m for m in model_ids(env or {}) if m != emb]
    yaml_list = lambda key, xs: (f"{key}:\n" + "".join(f"  - {x}\n" for x in xs)) if xs else ""   # noqa: E731
    limit = os.getenv("MAX_PAPERS_PER_SESSION", "5")
    return f"""---
title: {title}
emoji: 📄
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "{py}"
app_file: app.py
pinned: true
short_description: {SHORT_DESCRIPTION}
{yaml_list("tags", TAGS)}{yaml_list("models", models)}{f"thumbnail: {thumbnail}" + chr(10) if thumbnail else ""}---

# {title}

*Upload quantum physics papers (arXiv quant-ph preprints or journal PDFs) and ask questions. Every sentence of the answer cites the passage it rests on, and a checker verifies it.*

Upload a paper (PDF) and ask questions about it. The app splits the paper into passages (paragraphs,
equations, figure captions, reference entries), each with a stable id such as `P1-p4-2` (paper 1,
page 4, passage 2). Search is hybrid: BM25 keyword search plus sentence embeddings in a Qdrant vector
store, fused by reciprocal rank fusion.

A LangGraph agent answers a question from the best passages: a language model writes the answer and
ends every sentence with the ids of the passages it rests on. A deterministic checker (no model) then
verifies each sentence: the cited passages exist and were retrieved for this question, and every
quote and every number occurs in a cited passage. The model may revise its answer once after the
check; a sentence that still fails is shown with a warning and its reason, never removed silently.
The language models come from Hugging Face Inference, the free tier of Groq and Google Gemini, tried
in turn; the paid OpenAI API is never used.
{background_paragraph(n_background)}
**Limits of this public demo:** PDFs up to {VARIABLES['MAX_PDF_MB']} MB and {VARIABLES['MAX_PDF_PAGES']} pages,
{limit} papers per visitor. Uploaded papers are deleted after two hours without activity, or at once
with *Remove my papers*, and are never shared with other visitors. Do not upload confidential documents.

Built by Dr. Animesh Sinha Roy with PyMuPDF, fastembed, Qdrant, LangChain and Gradio.{chr(10) + chr(10) + "**Source code and tests:** " + github if github else ""}
"""


def background_paragraph(n: int) -> str:
    if not n:
        return ""
    return f"""
Besides the uploaded papers, the agent searches the passages of {n} open-access review and tutorial
papers (CC BY; the credits are in `background/SOURCES.md`). The model is told to use them only for
definitions and context the uploaded papers do not give; their sentences appear in a separate
**Background** part of the answer, cited and checked like the rest; a background sentence that speaks of "your paper" gets a warning. Under the
answer, **Further reading** lists up to three related papers or lecture notes, found by search over
their arXiv abstracts (CC0) or a short description: titles and links only, never used as evidence.
"""


def stage(folder: str, title: str, env: dict | None = None, thumbnail: str = "", github: str = "",
          repo: str = "", background_dir: str | None = None) -> list[str]:
    """Copy the whitelisted files, the background knowledge base and the generated README and requirements
    into `folder`."""
    missing = [p for p in APP_FILES + DATA_FILES if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Missing file(s): {', '.join(missing)}. Run this script in the project folder.")
    for p in APP_FILES + DATA_FILES:
        os.makedirs(os.path.join(folder, os.path.dirname(p)), exist_ok=True)
        shutil.copy(p, os.path.join(folder, p))
    background_dir = background_dir or BACKGROUND_DIR
    bg_files, n_background = background_files(background_dir)
    for p in bg_files:
        dest = os.path.join(folder, "background", os.path.relpath(p, background_dir))   # where background.py looks
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(p, dest)
    if not bg_files:
        print(f"Note: no {background_dir}/sources.json (run fetch_background.py): the Space will have no background "
              "knowledge.")
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme(title, env, thumbnail, github, repo, n_background))
    with open(os.path.join(folder, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(requirements())
    return sorted(os.path.relpath(os.path.join(r, n), folder).replace("\\", "/")
                  for r, _, names in os.walk(folder) for n in names)


def local_env() -> dict:
    from dotenv import dotenv_values
    return {k: v for k, v in dotenv_values(".env").items() if v} if os.path.isfile(".env") else {}


def deploy_token() -> tuple[str | None, str]:
    """The token for uploading and where it comes from: HF_DEPLOY_TOKEN, else the `hf auth login` token.
    HF_TOKEN in the environment is ignored here (it is the inference token, see the docstring)."""
    if os.getenv("HF_DEPLOY_TOKEN"):
        return os.environ["HF_DEPLOY_TOKEN"], "HF_DEPLOY_TOKEN"
    from huggingface_hub import get_token
    hidden = {k: os.environ.pop(k) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if k in os.environ}
    try:
        return get_token(), "hf auth login"
    finally:
        os.environ.update(hidden)


NOT_ALLOWED = ("Hugging Face refused the upload (403): the token from {src} cannot write to {repo}.\n"
               "Create a token of type Write (huggingface.co -> Settings -> Access Tokens), run\n"
               "`hf auth login`, choose 'paste a token', paste it at the hidden prompt, and run this again.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Space id, e.g. YOUR_NAME/ask-quantum-physics-papers")
    ap.add_argument("--title", default="Ask Your Quantum Physics Research Papers")
    ap.add_argument("--private", action="store_true", help="create the Space as private (first deploy)")
    ap.add_argument("--settings", action="store_true", help="also set the Space secrets and variables from .env")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded and set; change nothing")
    ap.add_argument("--thumbnail", default="", help="URL of an image shown when the Space link is shared")
    ap.add_argument("--github", default="", help="URL of the GitHub repository, linked from the Space card")
    ap.add_argument("--no-check", action="store_true", help="skip the pip check of the Space's install")
    a = ap.parse_args(argv)

    folder = tempfile.mkdtemp(prefix="pra_space_")
    try:
        env = local_env()
        files = stage(folder, a.title, env, a.thumbnail, a.github, a.repo)
        bad = find_keys([os.path.join(folder, f) for f in files])
        if bad:
            raise SystemExit("STOP: these files contain something that looks like an API key; nothing was "
                             f"uploaded: {', '.join(os.path.relpath(b, folder) for b in bad)}")
        print(f"Files for {a.repo} ({len(files)}):")
        for f in files:
            print("  ", f)
        print("\nrequirements.txt:\n" + "".join("   " + x + "\n" for x in requirements().splitlines()))
        if not a.no_check:
            print("Checking that the Space can install these packages (pip resolves them, installs nothing; "
                  "1-3 min) ...")
            problem = check_install(os.path.join(folder, "requirements.txt"))
            if problem:
                raise SystemExit("STOP: the Space's build would fail; nothing was uploaded. pip says:\n" + problem)
            print("   OK: the Space's install resolves.\n")

        print("Space card (README.md) header:")
        with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
            print("".join("   " + x for x in f.read().split("\n---\n")[0].splitlines(True)) + "\n")
        secrets = {k: env[k] for k in SECRET_KEYS if env.get(k)}
        variables = {k: (v if v is not None else env.get(k)) for k, v in VARIABLES.items()}
        variables = {k: v for k, v in variables.items() if v}
        if a.settings:
            print("Space secrets (values hidden):", ", ".join(secrets) or "none found in .env")
            print("Space variables:", ", ".join(f"{k}={v}" for k, v in variables.items()))
            if "OPENAI_API_KEY" in env:
                print("OPENAI_API_KEY is NOT sent (the public app uses free models only).")
        if a.dry_run:
            print("\nDry run: nothing was uploaded.")
            return 0

        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
        token, src = deploy_token()
        if not token:
            raise SystemExit("No Hugging Face login found; nothing was uploaded. Run `hf auth login` "
                             "with a Write token (choose 'paste a token').")
        api = HfApi(token=token)
        me = api.whoami()
        role = me.get("auth", {}).get("accessToken", {}).get("role", "unknown")
        print(f"Uploading as {me.get('name')} (token from {src}, type: {role}).")
        owner = a.repo.split("/")[0]
        if me.get("name") != owner and owner not in [o.get("name") for o in me.get("orgs", [])]:
            raise SystemExit(f"STOP: you are logged in as {me.get('name')}, but the Space belongs to {owner}; "
                             "nothing was uploaded.")
        try:
            api.create_repo(a.repo, repo_type="space", space_sdk="gradio", private=a.private, exist_ok=True)
            api.upload_folder(folder_path=folder, repo_id=a.repo, repo_type="space",
                              commit_message="Deploy Ask Your Quantum Physics Research Papers",
                              delete_patterns=["*.py", "data/*"])  # files removed from the whitelist go too
            if a.settings:
                for k, v in secrets.items():
                    api.add_space_secret(a.repo, k, v)
                for k, v in variables.items():
                    api.add_space_variable(a.repo, k, v)
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) == 403:
                raise SystemExit(NOT_ALLOWED.format(src=src, repo=a.repo)) from None
            raise
        print(f"\nDone: https://huggingface.co/spaces/{a.repo} (it builds for a few minutes).")
        return 0
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
