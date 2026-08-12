import sys
from pathlib import Path

import streamlit as st

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from sentiment_analysis.libs.web import create_app

if __name__ == "__main__":
    create_app()
