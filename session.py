"""
session.py - per-visitor state, so that several people can use the web app at the same time.
Copied from the Quantum Information Agent; the session now holds papers instead of quantum states.

Every browser tab of app.py gets its own Session: its own uploaded papers, their search index and
its own upload folder (data/sessions/<id>/). The command line and the tests use one default
session (folder data/).

How it works: the current session is kept in a context variable. app.py runs each request inside
    with session.use(session_id): ...
and the code reads and writes the session that is current at that moment. LangGraph copies the
context into its worker threads, so the agent's tools (Phase 1) will see the right session too.

Sessions that are idle for SESSION_TTL_S seconds are deleted together with their uploaded files (a
sweeper thread checks every 10 minutes); uploads are never shared with other visitors and never kept.
"""
FILES_VERSION = "2026-11-13a"

import contextlib
import contextvars
import os
import re
import shutil
import threading
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field

ON_SPACE = bool(os.getenv("SPACE_ID"))                       # set by Hugging Face on a Space
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", "7200"))      # 2 hours without a question
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))
MAX_PAPERS = int(os.getenv("MAX_PAPERS_PER_SESSION", "5"))     # uploaded papers per visitor
SESSIONS_DIR = "sessions"                                      # under data/
DEFAULT_ID = "default"


@dataclass
class Session:
    id: str
    papers: dict = field(default_factory=dict)       # alias ("P1") -> ingest.Paper
    index: object = None                             # index.Index over this session's papers
    question: dict = field(default_factory=lambda: {"text": ""})   # the user's latest question
    calls: list = field(default_factory=list)        # MCP clients: their tool calls + results
    last_provider: str | None = None                 # which model answered this user's last call
    asked: list = field(default_factory=list)        # times of this user's recent questions (rate limit)
    turns: list = field(default_factory=list)        # this user's conversation in the Ask tab (app.py)
    last_used: float = field(default_factory=time.time)

    @property
    def is_default(self) -> bool:
        return self.id == DEFAULT_ID

    @property
    def data_dir(self) -> str:
        """Where this session's own files go (uploaded PDFs)."""
        return "data" if self.is_default else f"data/{SESSIONS_DIR}/{self.id}"

    def clear(self):
        self.papers.clear()
        self.index = None
        self.question["text"] = ""
        self.calls.clear()
        self.turns.clear()


# CPU-heavy work (PDF parsing, embedding) is serialized, so that several visitors cannot exhaust the
# memory of a free Space together. Only that work holds the lock, not the (slow) language-model calls.
# Re-entrant: a tool may call another tool.
COMPUTE_LOCK = threading.RLock()

_DEFAULT = Session(DEFAULT_ID)
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("pra_session", default=_DEFAULT)
_SESSIONS: dict[str, Session] = {}
_LOCK = threading.Lock()
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def current() -> Session:
    return _CURRENT.get()


def get(session_id: str) -> Session:
    """The session with this id (created on first use). Idle sessions are cleaned up here."""
    if session_id == DEFAULT_ID:
        return _DEFAULT
    if not _SAFE_ID.match(session_id or ""):
        raise ValueError("invalid session id")
    with _LOCK:
        _expire_locked()
        s = _SESSIONS.get(session_id)
        if s is None:
            if len(_SESSIONS) >= MAX_SESSIONS:          # drop the longest-idle session
                oldest = min(_SESSIONS.values(), key=lambda x: x.last_used)
                _drop_locked(oldest.id)
            s = _SESSIONS[session_id] = Session(session_id)
        s.last_used = time.time()
        return s


@contextlib.contextmanager
def use(session_id: str):
    """Run the enclosed code with this session as the current one."""
    token = _CURRENT.set(get(session_id))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


def drop(session_id: str):
    with _LOCK:
        _drop_locked(session_id)


def _drop_locked(session_id: str):
    s = _SESSIONS.pop(session_id, None)
    if s is None:
        return
    s.clear()
    shutil.rmtree(s.data_dir, ignore_errors=True)
    for hook in list(ON_DROP):
        try:
            hook(session_id)
        except Exception:  # noqa: BLE001 - cleanup must never break a request
            pass


def _expire_locked():
    now = time.time()
    for sid in [k for k, s in _SESSIONS.items() if now - s.last_used > SESSION_TTL_S]:
        _drop_locked(sid)


_SWEEPER: list = []


def start_sweeper(every_s: float = 600.0):
    """Delete idle sessions (and their files) every `every_s` seconds, also when nobody makes a request.
    Without it an idle session is only deleted at the next request. Starts once per process."""
    with _LOCK:
        if _SWEEPER:
            return _SWEEPER[0]

        def run():
            while True:
                time.sleep(every_s)
                with _LOCK:
                    _expire_locked()

        t = threading.Thread(target=run, daemon=True, name="session-sweeper")
        _SWEEPER.append(t)
    t.start()
    return t


def active_ids() -> list[str]:
    with _LOCK:
        return list(_SESSIONS)


def exists(session_id: str) -> bool:
    """Is this session alive (not expired, not dropped)? Does not create it or refresh its idle time."""
    with _LOCK:
        _expire_locked()
        return session_id in _SESSIONS


def make_room(prefix: str, limit: int):
    """Before a new session with this id prefix is created: drop the longest-idle sessions with the
    prefix until fewer than `limit` remain (each kind of client gets its own share)."""
    with _LOCK:
        _expire_locked()
        mine = sorted((s for k, s in _SESSIONS.items() if k.startswith(prefix)), key=lambda s: s.last_used)
        while mine and len(mine) >= max(1, limit):
            _drop_locked(mine.pop(0).id)


ON_DROP: list = []          # callbacks(session_id), e.g. to forget the chat memory of that session


class SessionDict(MutableMapping):
    """A dict-like view of one dict of the CURRENT session (e.g. PAPERS = SessionDict("papers")): code
    written as if there were one user is per-user automatically."""

    def __init__(self, attr: str):
        self._attr = attr

    def _d(self) -> dict:
        return getattr(current(), self._attr)

    def __getitem__(self, k):
        return self._d()[k]

    def __setitem__(self, k, v):
        self._d()[k] = v

    def __delitem__(self, k):
        del self._d()[k]

    def __iter__(self):
        return iter(list(self._d()))

    def __len__(self):
        return len(self._d())

    def __contains__(self, k):
        return k in self._d()

    def __repr__(self):
        return repr(self._d())

    def clear(self):
        self._d().clear()

    def copy(self):
        return dict(self._d())


UPLOADS = "uploads"                     # what a web user sees instead of data/sessions/<id>
_SESSION_PATH = re.compile(r"data[/\\]" + SESSIONS_DIR + r"[/\\][^/\\\s'\"()]+[/\\]")


def display_path(text: str) -> str:
    """Replace the long session folder 'data/sessions/<id>/' by 'uploads/' in text shown to a user."""
    return _SESSION_PATH.sub(UPLOADS + "/", text)
