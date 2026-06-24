"""Pytest bootstrap: make `import src.*` work, and keep the server's test DB in tmp."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ACROUTER_DB", os.path.join(tempfile.gettempdir(), "acrouter_test.sqlite"))
