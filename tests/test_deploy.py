"""
The deploy script and the Space-only safety switches (adapted from the Quantum Information Agent).
No network: nothing is uploaded.
"""
import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import deploy_space
import llm

KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ALLOW_OPENAI", "USE_OLLAMA", "PROVIDER_ORDER",
        "PAID_FIRST", "HF_TOKEN", "USE_HF", "HF_MODEL", "HF_FALLBACK_MODEL", "MAX_HF_CALLS", "DAILY_HF_CALLS",
        "PROVIDERS_ONLY", "EXTRA_BASE_URL", "EXTRA_API_KEY", "EXTRA_MODEL"]


@pytest.fixture(autouse=True)
def no_project_background(monkeypatch, tmp_path):
    """The tests never read the project's own background/ (built by fetch_background.py on your
    computer): each test gets an empty folder unless it builds its own."""
    monkeypatch.setattr(deploy_space, "BACKGROUND_DIR", str(tmp_path / "project-background"))


@pytest.fixture
def space(monkeypatch, tmp_path):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "ON_SPACE", True)
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    llm.USAGE["hf_calls"] = 0
    yield monkeypatch
    llm.USAGE["hf_calls"] = 0


def test_space_never_uses_paid_openai_or_ollama(space):
    for k, v in (("OPENAI_API_KEY", "sk-test"), ("ALLOW_OPENAI", "1"), ("PAID_FIRST", "1"), ("GROQ_API_KEY", "x")):
        space.setenv(k, v)
    assert [n for n, _ in llm.build_providers([])] == ["Groq", "Groq backup (gpt-oss-20b)"]
    space.setenv("USE_OLLAMA", "1")
    assert "Ollama (local)" in [n for n, _ in llm.build_providers([])]


def test_daily_cap_survives_a_restart_and_resets_next_day(space, tmp_path):
    space.setenv("DAILY_HF_CALLS", "2")
    replies = iter([AIMessage(content=str(i)) for i in range(5)])
    capped = llm._hf_with_cap(GenericFakeChatModel(messages=replies))
    assert capped.invoke("a").content == "0" and capped.invoke("b").content == "1"
    with pytest.raises(RuntimeError, match="daily cap"):
        capped.invoke("c")
    llm.USAGE["hf_calls"] = 0
    with pytest.raises(RuntimeError, match="daily cap"):
        llm._hf_with_cap(GenericFakeChatModel(messages=replies)).invoke("d")
    (tmp_path / "hf_usage.json").write_text('{"date": "2000-01-01", "calls": 999}')
    assert llm.hf_calls_today() == 0
    (tmp_path / "hf_usage.json").write_text("not json")
    assert llm.hf_calls_today() == 0


def test_deploy_uploads_only_the_whitelist(tmp_path):
    files = deploy_space.stage(str(tmp_path), "Ask Your Quantum Physics Research Papers", background_dir=str(tmp_path / "no-background"))
    assert set(files) == set(deploy_space.APP_FILES) | {"README.md", "requirements.txt"}
    assert not any(f.startswith(("papers", "logs", "data", "tests", "corpus", ".env", "bakeoff", "results")) for f in files)
    req = (tmp_path / "requirements.txt").read_text()
    for pkg in ("pymupdf==", "fastembed==", "qdrant-client==", "langchain-qdrant==", "langgraph=="):
        assert pkg in req
    assert "gradio" not in req and "pytest" not in req and "pypdfium2" not in req and "pdfplumber" not in req
    assert deploy_space.find_keys([str(tmp_path / f) for f in files]) == []


def test_every_module_the_app_imports_is_uploaded():
    import ast
    tree = ast.parse(open("app.py", encoding="utf-8").read())
    local = {os.path.splitext(f)[0] for f in os.listdir(".") if f.endswith(".py")}
    imported = {n.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    needed = {m + ".py" for m in imported & local}
    for f in list(needed):                                   # one level of indirection (index -> ingest)
        sub = ast.parse(open(f, encoding="utf-8").read())
        needed |= {n.name + ".py" for node in ast.walk(sub) if isinstance(node, ast.Import) for n in node.names
                   if n.name in local}
    assert needed <= set(deploy_space.APP_FILES), needed - set(deploy_space.APP_FILES)


def test_deploy_stops_on_a_key_in_a_file(tmp_path):
    p = tmp_path / "leak.py"
    p.write_text('TOKEN = "hf_' + "a" * 34 + '"\n')
    q = tmp_path / "ok.py"
    q.write_text('os.environ["HF_TOKEN"]  # read from the environment\n')
    assert deploy_space.find_keys([str(p), str(q)]) == [str(p)]


def test_deploy_never_sends_the_openai_key():
    assert "OPENAI_API_KEY" not in deploy_space.SECRET_KEYS
    assert deploy_space.VARIABLES["PROVIDER_ORDER"].split(",")[0] == "hf"


def test_gitignore_keeps_secrets_papers_and_run_files_out():
    lines = open(".gitignore", encoding="utf-8").read().splitlines()
    for line in (".env", ".env.*", "!.env.example", "logs/", "data/sessions/", "papers/", "*.pdf", "bakeoff/",
                 "results/", "*.zip", "__pycache__/"):
        assert line in lines, line


def test_space_card(tmp_path):
    env = {"HF_MODEL": "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra", "HF_FALLBACK_MODEL": "openai/gpt-oss-120b:deepinfra"}
    deploy_space.stage(str(tmp_path), "Ask Your Quantum Physics Research Papers", env, "https://example.com/card.png",
                       "https://github.com/me/paper-research-agent")
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    head = card.split("\n---\n")[0]
    import yaml
    meta = yaml.safe_load(head.lstrip("-\n"))
    assert meta["sdk"] == "gradio" and meta["app_file"] == "app.py" and len(meta["short_description"]) <= 60
    assert meta["models"] == ["Snowflake/snowflake-arctic-embed-s", "deepseek-ai/DeepSeek-V4-Flash-0731",
                              "openai/gpt-oss-120b"]                   # embedding model + the HF language models
    import index
    assert deploy_space.EMBED_DEFAULT == index.DEFAULT_MODEL
    assert {"agent", "rag", "langgraph"} <= set(meta["tags"])
    assert "qdrant" in meta["tags"] and meta["thumbnail"] == "https://example.com/card.png"
    assert "**Source code and tests:** https://github.com/me/paper-research-agent" in card
    # the card describes what agent.py and checker.py do, and does not promise more
    assert meta["title"] == "Ask Your Quantum Physics Research Papers"
    assert "\n# Ask Your Quantum Physics Research Papers\n" in card and "Paper Research Agent" not in card
    assert "deterministic checker" in card and "never removed silently" in card
    assert "OpenAI API is never used" in card and "not in this version yet" not in card


def test_pydantic_is_not_pinned():
    req = deploy_space.requirements().splitlines()
    assert "pydantic" in req and not any(x.startswith("pydantic==") for x in req)


def test_pip_check_runs_the_spaces_install(monkeypatch, tmp_path):
    import subprocess
    import types
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=1, stdout="", stderr="ERROR: ResolutionImpossible")
    monkeypatch.setattr(subprocess, "run", fake_run)
    req = tmp_path / "requirements.txt"
    req.write_text(deploy_space.requirements())
    assert "ResolutionImpossible" in deploy_space.check_install(str(req))
    cmd = " ".join(seen["cmd"])
    assert "--dry-run" in cmd and "gradio[oauth,mcp]==" in cmd and "--platform" not in cmd


def test_upload_never_uses_the_inference_token_from_the_shell(monkeypatch):
    import huggingface_hub
    seen = {}
    monkeypatch.setenv("HF_TOKEN", "inference-token")
    monkeypatch.delenv("HF_DEPLOY_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token",
                        lambda: seen.setdefault("env", os.environ.get("HF_TOKEN")) or "login-token")
    assert deploy_space.deploy_token() == ("login-token", "hf auth login")
    assert seen["env"] is None and os.environ["HF_TOKEN"] == "inference-token"


class FakeApi:
    def __init__(self, name="me", status=None):
        self.name, self.status, self.uploaded, self.private = name, status, False, None

    def __call__(self, token=None):
        return self

    def whoami(self):
        return {"name": self.name, "orgs": [], "auth": {"accessToken": {"role": "write"}}}

    def create_repo(self, *a, private=None, **k):
        self.private = private
        if self.status:
            import httpx
            from huggingface_hub.errors import HfHubHTTPError
            r = httpx.Response(self.status, request=httpx.Request("POST", "https://huggingface.co"))
            raise HfHubHTTPError("403 Forbidden", response=r)

    def upload_folder(self, *a, **k):
        self.uploaded = True


def _deploy(monkeypatch, fake, *extra):
    import huggingface_hub
    monkeypatch.setattr(deploy_space, "check_install", lambda path: None)
    monkeypatch.setattr(deploy_space, "deploy_token", lambda: ("tok", "hf auth login"))
    monkeypatch.setattr(huggingface_hub, "HfApi", fake)
    return deploy_space.main(["--repo", "me/x", *extra])


def test_a_403_is_explained_in_plain_words(monkeypatch):
    with pytest.raises(SystemExit, match="(?s)cannot write to me/x.*paste a token"):
        _deploy(monkeypatch, FakeApi(status=403))


def test_wrong_account_uploads_nothing(monkeypatch):
    fake = FakeApi(name="someone-else")
    with pytest.raises(SystemExit, match="logged in as someone-else"):
        _deploy(monkeypatch, fake)
    assert not fake.uploaded


def test_private_first_deploy(monkeypatch, capsys):
    fake = FakeApi()
    assert _deploy(monkeypatch, fake, "--private") == 0 and fake.uploaded and fake.private is True
    assert "Uploading as me (token from hf auth login, type: write)" in capsys.readouterr().out


def _background(folder, licence="CC BY 4.0"):
    import json
    os.makedirs(os.path.join(folder, "embeddings"))
    with open(os.path.join(folder, "sources.json"), "w", encoding="utf-8") as f:
        json.dump([{"alias": "B1", "key": "a2020", "licence": "CC BY 4.0"}, {"alias": "B2", "key": "b2021", "licence": licence}], f)
    for name in ("passages.jsonl", "reading.jsonl", "SOURCES.md", "embeddings/a2020-123.npy", "notes.txt"):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write("x")
    return folder


def test_deploy_ships_the_background_and_says_so(tmp_path):
    bg = _background(str(tmp_path / "bg"))
    files = deploy_space.stage(str(tmp_path / "out"), "Ask Your Quantum Physics Research Papers", background_dir=bg)
    assert {f for f in files if f.startswith("background/")} == {
        "background/sources.json", "background/passages.jsonl", "background/reading.jsonl", "background/SOURCES.md",
        "background/embeddings/a2020-123.npy"}                       # not notes.txt: a whitelist
    card = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    assert "searches the passages of 2 open-access review" in card and "Further reading" in card
    none = deploy_space.stage(str(tmp_path / "out2"), "Ask Your Quantum Physics Research Papers", background_dir=str(tmp_path / "x"))
    assert "Background" not in (tmp_path / "out2" / "README.md").read_text(encoding="utf-8")
    assert not any(f.startswith("background/") for f in none)


def test_deploy_stops_on_a_background_paper_that_is_not_cc_by(tmp_path):
    for bad in ("CC BY-NC-SA 3.0", "", "CC BY-ND 4.0", "other"):
        bg = _background(str(tmp_path / f"bg{len(bad)}{bad[:5]}"), licence=bad)
        with pytest.raises(SystemExit, match="STOP: background papers without a CC BY"):
            deploy_space.stage(str(tmp_path / "o"), "x", background_dir=bg)
    assert deploy_space.open_licence("CC BY") and deploy_space.open_licence("CC0 1.0")
    old = str(tmp_path / "old")                     # sources.json of update 11: no licence field
    os.makedirs(old)
    with open(os.path.join(old, "sources.json"), "w", encoding="utf-8") as f:
        f.write('[{"alias": "B1", "key": "a2020", "evidence": "OpenAlex: cc-by"}]')
    with pytest.raises(SystemExit, match="older fetch_background.py"):
        deploy_space.stage(str(tmp_path / "o3"), "x", background_dir=old)
    bg = _background(str(tmp_path / "nocredits"))
    os.remove(os.path.join(bg, "SOURCES.md"))
    with pytest.raises(SystemExit, match="lacks SOURCES.md"):
        deploy_space.stage(str(tmp_path / "o2"), "x", background_dir=bg)


def test_a_link_with_ask_in_it_is_no_key(tmp_path):
    """16 Nov 2026: the Space card's GitHub link ".../ask-quantum-physics-papers" stopped the upload as an
    OpenAI key "sk-quantum-physics-papers"."""
    card = tmp_path / "README.md"
    card.write_text("**Source code and tests:** https://github.com/animeshQKD/ask-quantum-physics-papers\n")
    assert deploy_space.find_keys([str(card)]) == []
    leak = tmp_path / "leak.py"
    leak.write_text('key = "sk-proj-' + "a1B2c3D4e5F6g7H8i9J0k1L2" + '"\n')
    assert deploy_space.find_keys([str(leak)]) == [str(leak)]
