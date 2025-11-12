#!/usr/bin/env python3
"""
Scan Python files under ML/ and replace occurrences of joblib.dump/joblib.load
with model_io.save_model/model_io.load_model. Creates .bak backups before modifying.
Run from project root: python ML/scripts/migrate_to_model_io.py
"""
import re
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
py_files = list(ROOT.rglob("*.py"))

dump_re = re.compile(r"joblib\.dump\(\s*([^,]+)\s*,\s*([^\)]+)\)")
load_re = re.compile(r"joblib\.load\(\s*([^\)]+)\)")

for p in py_files:
    text = p.read_text(encoding="utf-8")
    new_text = text
    if "joblib.dump" in text or "joblib.load" in text:
        # backup
        bak = p.with_suffix(p.suffix + ".bak")
        shutil.copyfile(p, bak)
        new_text = dump_re.sub(r"save_model(\1, \2)", new_text)
        new_text = load_re.sub(r"load_model(\1)", new_text)
        # ensure import exists
        if "from .model_io import save_model, load_model" not in new_text and "from model_io import save_model, load_model" not in new_text:
            # add import near top (after other imports)
            new_text = new_text.replace("\n\n", "\n\nfrom .model_io import save_model, load_model\n\n", 1)
        p.write_text(new_text, encoding="utf-8")
        print(f"Patched {p} (backup -> {bak})")
print("Migration script completed. Review .bak files and run tests.")