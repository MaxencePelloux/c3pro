"""Patient identifier (PATHID) from a slide name: the 10-character prefix, whatever separator follows it."""

import re

PATHID_REGEX = re.compile(r"^(\d{2}[A-Z]{2}\d{6})")


def extract_pathid(slide_name: str) -> str:
    # some slide names use a space instead of a dash before the block, so split("-") is not enough
    match = PATHID_REGEX.match(str(slide_name).strip())
    if not match:
        raise ValueError(f"Cannot extract PATHID from slide_name: {slide_name!r}")
    return match.group(1)
