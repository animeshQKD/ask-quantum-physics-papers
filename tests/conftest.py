"""Shared test set-up: run from the project root, offline, with the stand-in embeddings."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
os.environ["EMBEDDINGS"] = "hash"          # never download a model in the tests
os.environ["BACKGROUND_DIR"] = os.path.join(ROOT, "tests", "no-background")   # tests build their own
os.environ.setdefault("DOMAIN", "physics")

import pytest  # noqa: E402

import pdf_factory  # noqa: E402


@pytest.fixture(scope="session")
def physics_pdf(tmp_path_factory):
    return pdf_factory.physics_paper(str(tmp_path_factory.mktemp("pdf") / "synthetic_physics.pdf"))


@pytest.fixture(scope="session")
def generic_pdf(tmp_path_factory):
    return pdf_factory.generic_paper(str(tmp_path_factory.mktemp("pdf") / "bird_migration.pdf"))


@pytest.fixture(scope="session")
def scanned_pdf(tmp_path_factory):
    return pdf_factory.scanned_pdf(str(tmp_path_factory.mktemp("pdf") / "scan.pdf"))


@pytest.fixture(scope="session")
def physics_paper(physics_pdf):
    import ingest
    return ingest.read_pdf(physics_pdf, "P1")


@pytest.fixture(scope="session")
def generic_paper(generic_pdf):
    import ingest
    return ingest.read_pdf(generic_pdf, "P2")
