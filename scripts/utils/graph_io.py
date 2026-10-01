"""Find the .pt graph file of a slide whatever separator its name was saved with."""

from pathlib import Path


def resolve_graph_path(slide_name: str, graph_dir) -> Path:
    # names with a space were saved either without it or with a dash, depending on the stage
    graph_dir = Path(graph_dir)
    s = str(slide_name)
    candidates = [
        f"{s}.pt",
        f"{s.replace(' ', '')}.pt",
        f"{s.replace(' ', '-')}.pt",
    ]
    for candidate in candidates:
        path = graph_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No graph file found for slide {slide_name!r} in {graph_dir}. "
        f"Tried: {candidates}"
    )
