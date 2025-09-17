# Developer Guide

This guide outlines standards and best practices for contributing to the Traffic Detection Analysis project.

---

## 1. Coding Standards

- **PEP8:**  
  All code should comply with PEP8 style guidelines.
- **Formatting:**  
  Use `black` for auto-formatting:
  ```bash
  black src/
  ```
- **Linting:**  
  Check with `pylint`:
  ```bash
  pylint src/
  ```

---

## 2. Documentation

- **Docstrings:**  
  All public classes, functions, and modules must include descriptive docstrings.
- **README:**  
  Main project overview and rationale in `README.md`.
- **Component Docs:**  
  See `src/COMPONENTS.md` and `src/pages/PAGES.md` for detailed module/page documentation.

---

## 3. Extensibility

- **Adding Analytics:**  
  Extend `analyzer.py` for new dataset stats. Add supporting functions to `analysis.py`.
- **Dashboard Pages:**  
  Add numbered `.py` files to `src/pages/` for new features.
- **Model/Pipeline:**  
  Modify training/evaluation scripts for new architectures or metrics.

---

## 4. Testing

- **Unit Tests:**  
  Add tests for new analytics or utility functions where possible.
- **Manual Testing:**  
  Run dashboard and scripts with sample datasets to verify outputs.

---

## 5. Contribution Workflow

1. Fork repository and create feature branch.
2. Make changes with code style and documentation.
3. Test new features or fixes.
4. Open pull request with clear description and related documentation updates.

---

## 6. Docker/Environment

- **Dependencies:**  
  Pin in `requirements.txt`.
- **Paths:**  
  Use `/app/data/traffic-detection-project/` for all data references (Docker).
- **One-time pipeline:**  
  The entrypoint `start.sh` runs a single clean pipeline after image rebuilds by comparing an image build ID with a marker in the mounted volume. It clears `data/outputs/`, runs training/evaluation, then serves Streamlit.
- **Reproducibility:**  
  All experiments and outputs are reproducible via container.

---

**For questions, open an issue or pull request on GitHub.**