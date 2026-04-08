"""Server entrypoint required by multi-mode deployment validators."""

from main import app
import uvicorn


def main() -> None:
    """CLI entrypoint for `server` script in pyproject.toml."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
