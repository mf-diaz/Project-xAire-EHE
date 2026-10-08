"""
Fails (exit 1) if the tracked files of this repository contain anything that must not be public.
Run before every push:  python tools/check_repo_clean.py
Enabled as a pre-commit hook with:  git config core.hooksPath .githooks

Checks, on `git ls-files`:
  1. forbidden file names (raw or processed participant data, caches, lock files, stray modules)
  2. forbidden column names in the header of any tracked .csv (identifiers, free text)
  3. Jupyter notebooks that still carry outputs (outputs can print survey free text)
  4. any file above 25 MB (GitHub rejects 100 MB, warns at 50 MB)
  5. anything tracked under data/ (except its README), work/ or results/
"""
import csv, json, subprocess, sys, os

FORBIDDEN_NAMES = {
    "users_xaire.csv", "games_xaire.csv", "participants.csv", "games.csv", "df_users_six_players.csv",
    "df_games_six_players.csv", "successful_users_six.csv", "successful_games_six.csv",
    "empirical_successful_six_players.csv", "pollution.csv", "fastdtw.py", "fastdtw_fallback.py",
}
FORBIDDEN_SUFFIXES = (".pkl",)
FORBIDDEN_PREFIXES = ("~$",)
FORBIDDEN_COLUMNS = {"nickname", "comment", "comentari", "enquesta_final_pr11", "email", "ip", "date_register", "date_tutorial"}
MAX_BYTES = 25 * 1024 * 1024

root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
problems = []
for f in filter(None, files):
    base = os.path.basename(f)
    path = os.path.join(root, f)
    if not os.path.exists(path):
        continue
    if base in FORBIDDEN_NAMES or base.endswith(FORBIDDEN_SUFFIXES) or base.startswith(FORBIDDEN_PREFIXES):
        problems.append(f"forbidden file: {f}")
    if f.startswith(("data/", "work/", "results/")) and f != "data/README.md":
        problems.append(f"generated or downloaded file is tracked: {f}")
    if os.path.getsize(path) > MAX_BYTES:
        problems.append(f"file above 25 MB: {f}")
    if base.endswith(".csv"):
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            header = next(csv.reader(fh), [])
        bad = FORBIDDEN_COLUMNS & {h.strip().lower() for h in header}
        if bad:
            problems.append(f"forbidden column(s) {sorted(bad)} in {f}")
    if base.endswith(".ipynb"):
        nb = json.load(open(path, encoding="utf-8"))
        if any(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"):
            problems.append(f"notebook with outputs: {f}")
if problems:
    print("REPOSITORY NOT CLEAN:\n  " + "\n  ".join(problems)); sys.exit(1)
print(f"clean: {len([f for f in files if f])} tracked files checked")
