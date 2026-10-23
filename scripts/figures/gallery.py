"""Build linked or self-contained dropdown galleries from existing figures."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import tyro


@dataclass(frozen=True)
class Config:
    """Figure pages below the output HTML's directory."""

    pages: tuple[Path, ...]  # Existing generated figure HTMLs, in dropdown order.
    output: Path  # Unused HTML path under runs/, above all selected pages.
    labels: tuple[str, ...] = ()  # Optional labels; default each figure's title.
    standalone: bool = False  # Embed data and assets for sharing one offline HTML.


@dataclass(frozen=True)
class Page:
    """A relative figure URL, dropdown label, and source checksum."""

    url: str
    label: str
    sha256: str


_IMPORT = re.compile(
    r"(?P<prefix>\bfrom\s+|\bimport\s*\(\s*)"
    r"(?P<quote>['\"])(?P<path>\./[^'\"]+)(?P=quote)"
)


def _local(root: Path, name: str) -> Path:
    """Reject asset and CSV paths that escape the generated figure directory."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("embedded assets must stay within their figure directory")
    return path


def _standalone(pages: tuple[Path, ...]) -> dict:
    """Deduplicate a shared renderer and embed verified CSV snapshots verbatim."""
    root = pages[0].parent
    raw: dict[str, str] = {}

    def visit(name: str) -> None:
        path = _local(root, name)
        name = path.relative_to(root).as_posix()
        if name in raw:
            return
        raw[name] = path.read_text()
        if path.suffix == ".mjs":
            for match in _IMPORT.finditer(raw[name]):
                visit(str(Path(name).parent / match["path"]))

    for name in (
        "figures.mjs",
        "viz/ui.css",
        "viz/percentiles.css",
        "viz/vendor/d3.v7.min.js",
        "viz/vendor/D3-LICENSE",
        "viz/vendor/jspdf.LICENSE",
        "viz/vendor/svg2pdf.js.LICENSE",
    ):
        visit(name)
    if "#figure-data" not in raw["figures.mjs"]:
        raise ValueError("rebuild pages with the current renderer before embedding")

    assets = dict(raw)
    for name, source in raw.items():
        if not name.endswith(".mjs"):
            continue

        def rewrite(match: re.Match[str], module: str = name) -> str:
            dependency = _local(root, str(Path(module).parent / match["path"]))
            target = dependency.relative_to(root).as_posix()
            prefix, quote = match["prefix"], match["quote"]
            return f"{prefix}{quote}greq-offline/{target}{quote}"

        assets[name] = _IMPORT.sub(rewrite, source)

    template = pages[0].read_text()
    data = []
    for page in pages:
        if page.read_text() != template or any(
            _local(page.parent, name).read_text() != source
            for name, source in raw.items()
        ):
            raise ValueError("standalone pages must use the same renderer assets")
        manifest = json.loads((page.parent / "data/manifest.json").read_text())
        if not manifest.get("runs"):
            raise ValueError("standalone pages require populated manifests")
        csvs = {}
        for run in manifest["runs"]:
            content = _local(page.parent, run["file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != run["sha256"]:
                raise ValueError("embedded CSV checksum mismatch")
            csvs[run["file"]] = content.decode("utf-8")
            source = Path(run.get("source", ""))
            if source.is_absolute():
                run["source"] = (
                    source.relative_to(Path.cwd()).as_posix()
                    if source.is_relative_to(Path.cwd())
                    else source.name
                )
        data.append({"manifest": manifest, "csvs": csvs})
    return {"template": template, "assets": assets, "pages": data}


def build(config: Config) -> Path:
    """Create a gallery of linked pages or embedded data and shared assets."""
    output = config.output.resolve()
    if not output.is_relative_to(Path("runs").resolve()) or output.suffix != ".html":
        raise ValueError("gallery output must be an HTML file under runs/")
    pages = tuple(path.resolve() for path in config.pages)
    if not pages or len(set(pages)) != len(pages):
        raise ValueError("provide distinct figure HTMLs")
    if config.labels and len(config.labels) != len(pages):
        raise ValueError("labels must have one entry per page")
    entries = []
    for index, path in enumerate(pages):
        boundary = Path("runs").resolve() if config.standalone else output.parent
        if path == output or not path.is_relative_to(boundary):
            raise ValueError(
                "figure pages must be below the gallery directory "
                "or runs/ for standalone output"
            )
        if not path.is_file() or path.suffix != ".html":
            raise ValueError(f"missing figure HTML: {path}")
        manifest = json.loads((path.parent / "data/manifest.json").read_text())
        label = (
            config.labels[index]
            if config.labels
            else manifest.get("title") or path.parent.name
        )
        if not isinstance(label, str) or not label.strip():
            raise ValueError("page labels must be nonempty strings")
        entries.append(
            Page(
                url=str(index)
                if config.standalone
                else quote(path.relative_to(output.parent).as_posix()),
                label=label,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    stylesheet = pages[0].parent / "viz/ui.css"
    if not stylesheet.is_file():
        raise ValueError("first figure must contain the shared viz/ui.css")
    data = json.dumps([asdict(entry) for entry in entries], ensure_ascii=True)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e")
    template = (Path(__file__).with_name("assets") / "gallery.html").read_text()
    bundle = _standalone(pages) if config.standalone else None
    bundle_data = (
        json.dumps(bundle, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    stylesheet_href = (
        "data:,"
        if bundle is not None
        else html.escape(
            quote(stylesheet.relative_to(output.parent).as_posix()), quote=True
        )
    )
    replacements = {
        "__BUNDLE__": bundle_data,
        "__STYLESHEET__": stylesheet_href,
        "__PAGES__": data,
    }
    document = re.sub(
        r"__BUNDLE__|__STYLESHEET__|__PAGES__",
        lambda match: replacements[match[0]],
        template,
    )
    if bundle is not None:
        document = document.replace(
            '<link rel="stylesheet" href="data:,">',
            '<meta http-equiv="Content-Security-Policy" content="connect-src blob:">'
            "<style>" + stylesheet.read_text() + "</style>",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(document)
    print(output)
    return output


def main() -> None:
    """Generate a selector page from explicit figure HTML paths."""
    build(tyro.cli(Config))


if __name__ == "__main__":
    main()
