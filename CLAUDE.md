# Action Documentation and Project Guidelines for Claude

This document outlines the strict behavioral, coding, verification, and action documentation guidelines that Claude must follow when working on the `llm-inference-emulator` project.

---

## 🛠️ Build and Test Commands

- **Test suite (local)**: Run `python3 tests/test_formula.py` directly. **Must print `all tests passed`**. Do NOT use the `pytest` command line harness, as it is not a direct runtime dependency.
- **Streamlit UI (local)**: Run `streamlit run ui/app.py` to launch the visualization dashboard.
- **CI verification**: Run `gh run list --workflow=test.yml --limit 1` and ensure the run is completely green (`✓` / `completed success`). Never push new code if the CI is red.

---

## 📝 Rules for Detailed Action Documentation

Every interaction, code change, and verification must follow these rigorous documentation standards:

### 1. Explain the "Why" and the "How"
- **Code Comments**: When introducing mathematical formulations, hardware communication models, or physical penalties, add explanatory comments directly in the source code.
- **Physical Modeling Context**: Explicitly detail the logic behind models (e.g., DeepSeek MLA representations, Megatron-LM Ring All-Reduce, PP 1F1B bubble sizes, or PCIe bandwidth/latency penalties). Cite hardware specifications and driver-level assumptions.

### 2. High-Discipline Git Commit Practices
- **Atomic Commits**: Keep your commits highly focused. Do not bundle multiple unrelated tasks or fixes into a single commit.
- **Semantic Prefixing**: Always use standard semantic commit prefixes:
  - `fix:` for bug fixes.
  - `docs:` for documentation updates.
  - `chore:` for maintenance, cleanup, and task tracking.
- **Detailed Explanations**: If there is a regression, a database truncation, or a mismatch in data length (e.g., the RTX-3090 row count changing from 60 to 33), **explain it thoroughly in the git commit message**. Detail the root-cause analysis (e.g., upstream file truncation) and why the new state is correct.

### 3. Absolute Synchronicity of Claims and Reality
- **No Ghost Artifacts**: Never write in status reports or commit messages that files like `task.md` or `implementation_plan.md` are updated in the repository if they only exist as private agent files. Only claim documentation is updated if the actual git-tracked files (e.g., `docs/WALKTHROUGH.md`, `README.md`) are modified.
- **Sync Documentation**: Whenever new physical modeling parameters or equations are introduced, immediately document them in `docs/WALKTHROUGH.md`.

### 4. Continuous Verification Loop
- Before declaring a task finished, verify that the local test script passes and prints `all tests passed`.
- Verify the GitHub Actions status after pushing to ensure the build remains 100% healthy.

---

## 🛠️ Code Style Guidelines

- **Python Styling**: Write clean, standard Python 3.11+ code.
- **Dependencies**: Keep runtime dependencies to a minimum. Do not introduce new packages to `requirements.txt` unless absolutely necessary and approved.
- **Formatting**: Preserve existing code comments and code formatting when editing files.
