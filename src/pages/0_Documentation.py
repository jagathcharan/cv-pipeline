import streamlit as st
from pathlib import Path
from functools import lru_cache


DOC_FILES = {
    "README": Path("README.md"),
    "Usage Guide": Path("USAGE_GUIDE.md"),
    "Architecture": Path("ARCHITECTURE.md"),
    "Developer Guide": Path("DEVELOPER_GUIDE.md"),
    "Components": Path("src/COMPONENTS.md"),
    "Pages": Path("src/pages/PAGES.md"),
}


@lru_cache(maxsize=None)
def _read_text(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return f"_Missing file: {path_str}_"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"_Failed to read {path_str}: {exc}_"


def main():
    st.set_page_config(page_title="Documentation | Traffic Detection", layout="wide")
    st.title("Project Documentation")

    st.sidebar.header("Documentation Sections")
    available_titles = [title for title, p in DOC_FILES.items() if Path(p).exists()]
    if not available_titles:
        available_titles = list(DOC_FILES.keys())

    selection = st.sidebar.radio("Select a document", available_titles, index=0)

    if selection:
        content = _read_text(str(DOC_FILES[selection]))
        st.markdown(content)

    with st.expander("Show all documents"):
        for title, path in DOC_FILES.items():
            st.markdown(f"## {title}")
            st.markdown(_read_text(str(path)))
            st.divider()


if __name__ == "__main__":
    main()

