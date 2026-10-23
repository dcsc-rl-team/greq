"""Validate linked and self-contained dropdown galleries."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from scripts.figures.gallery import Config, build


def page(root, name, title):
    path = root / name / "index.html"
    path.parent.mkdir(parents=True)
    path.write_text("<!doctype html><html></html>")
    (path.parent / "data").mkdir()
    (path.parent / "data/manifest.json").write_text(json.dumps({"title": title}))
    (path.parent / "viz").mkdir()
    (path.parent / "viz/ui.css").write_text(":root {}")
    return path


def test_gallery_reuses_existing_pages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("runs/examples")
    a = page(root, "first page", "Garnet — γ = 0.95")
    b = page(root, "second", "Garnet — γ = 0.999")
    output = build(Config(pages=(a, b), output=root / "index.html"))
    document = output.read_text()
    match = re.search(r'type="application/json">(.*?)</script>', document)
    assert match is not None
    entries = json.loads(match.group(1))
    assert entries[0]["url"] == "first%20page/index.html"
    assert entries[1]["label"] == "Garnet — γ = 0.999"
    assert entries[0]["sha256"] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert '<iframe id="figure-page"' in document
    assert not (root / "viz").exists()
    with pytest.raises(FileExistsError):
        build(Config(pages=(a, b), output=output))


def test_gallery_escapes_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("runs/examples")
    title = '</script><script>alert("x")</script>'
    a = page(root, "one", title)
    output = build(Config(pages=(a,), output=root / "index.html"))
    assert title not in output.read_text()
    assert "\\u003c/script\\u003e" in output.read_text()


ASSETS = Path(__file__).resolve().parents[1] / "scripts/figures/assets"


def standalone_page(root, name="one", title="Garnet — tuned — 11 seeds"):
    path = page(root, name, title)
    shutil.copytree(ASSETS / "viz", path.parent / "viz", dirs_exist_ok=True)
    for filename in ("index.html", "figures.mjs"):
        shutil.copyfile(ASSETS / filename, path.parent / filename)
    csv = b"step,value_linf\n0,10\n100,1\n"
    (path.parent / "data/run.csv").write_bytes(csv)
    (path.parent / "data/manifest.json").write_text(
        json.dumps(
            {
                "title": title,
                "runs": [
                    {
                        "file": "data/run.csv",
                        "source": str((path.parent / "data/run.csv").resolve()),
                        "sha256": hashlib.sha256(csv).hexdigest(),
                    }
                ],
            }
        )
    )
    return path


def test_standalone_embeds_shared_assets_and_verbatim_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("runs/figures")
    a = standalone_page(root, "one", "</script><script>alert(1)</script>")
    b = standalone_page(root, "two")
    output = Path("runs/share/experiments.html")
    build(Config(pages=(a, b), output=output, standalone=True))
    text = output.read_text()
    match = re.search(
        r'id="gallery-bundle" type="application/json">(.*?)</script>', text
    )
    assert match is not None
    bundle = json.loads(match[1])
    assert len(bundle["pages"]) == 2
    assert (
        bundle["pages"][0]["manifest"]["runs"][0]["source"]
        == "runs/figures/one/data/run.csv"
    )
    assert str(tmp_path) not in text
    assert (
        bundle["pages"][0]["csvs"]["data/run.csv"] == "step,value_linf\n0,10\n100,1\n"
    )
    assert "</script><script>alert(1)</script>" not in text
    assert "connect-src blob:" in text
    assert '<link rel="stylesheet" href=' not in text
    assets = bundle["assets"]
    assert "greq-offline/viz/fig.mjs" in assets["figures.mjs"]
    assert "greq-offline/viz/vendor/jspdf.umd.min.js" in assets["viz/export.mjs"]
    assert "greq-offline/viz/vendor/svg2pdf.umd.min.js" in assets["viz/export.mjs"]
    for name in ("D3-LICENSE", "jspdf.LICENSE", "svg2pdf.js.LICENSE"):
        assert assets[f"viz/vendor/{name}"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        build(Config(pages=(a, b), output=output, standalone=True))
    assert output.read_bytes() == before


@pytest.mark.parametrize("failure", ["checksum", "escape", "renderer", "data_selector"])
def test_standalone_rejects_inconsistent_inputs(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    a = standalone_page(Path("runs/pages"), "one")
    b = standalone_page(Path("runs/pages"), "two")
    if failure == "checksum":
        (a.parent / "data/run.csv").write_text("corrupted")
        message = "checksum"
    elif failure == "escape":
        manifest = a.parent / "data/manifest.json"
        data = json.loads(manifest.read_text())
        data["runs"][0]["file"] = "../outside.csv"
        manifest.write_text(json.dumps(data))
        message = "within"
    elif failure == "renderer":
        (b.parent / "viz/palette.mjs").write_text("different renderer")
        message = "same renderer"
    else:
        renderer = a.parent / "figures.mjs"
        renderer.write_text(
            renderer.read_text().replace("#figure-data", "#missing-data")
        )
        message = "rebuild pages"
    with pytest.raises(ValueError, match=message):
        build(
            Config(pages=(a, b), output=Path("runs/share/test.html"), standalone=True)
        )
    assert not Path("runs/share/test.html").exists()


def test_gallery_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("runs/examples")
    a = page(root, "one", "One")
    with pytest.raises(ValueError, match="under runs"):
        build(Config(pages=(a,), output=Path("index.html")))
    with pytest.raises(ValueError, match="distinct"):
        build(Config(pages=(a, a), output=root / "index.html"))
    with pytest.raises(ValueError, match="below"):
        build(Config(pages=(a,), output=root / "elsewhere/index.html"))
    with pytest.raises(ValueError, match="one entry"):
        build(Config(pages=(a,), labels=("a", "b"), output=root / "index.html"))
    with pytest.raises(ValueError, match="nonempty"):
        build(Config(pages=(a,), labels=("",), output=root / "index.html"))
