"""Build (or rebuild) the search index from the PDFs in data/raw.

Equivalent to ``challansaathi build-index``; kept as a script for use without installing
the console entry point.
"""

from challansaathi.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-index"]))
