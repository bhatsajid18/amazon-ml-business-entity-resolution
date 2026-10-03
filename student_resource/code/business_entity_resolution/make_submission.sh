#!/usr/bin/env bash
# Build the final submission zip in the layout the challenge requires:
#
#   <team_name>_submission.zip
#   ├── output/matching_results.tsv, output/candidate_pairs.tsv
#   ├── code/business_entity_resolution/  (src/, README.md, requirements.txt, sbatch/, ...)
#   └── Documentation_template.md          (the filled-in methodology write-up)
#
# Usage (from code/business_entity_resolution/):
#   bash make_submission.sh <team_name> <run folder under output/history/>
#   bash make_submission.sh MyTeam labse2c_20260927_133000
# The zip is written to student_resource/<team_name>_submission.zip.
set -eo pipefail
TEAM="${1:?team name}"
RUN="${2:?run folder under output/history, e.g. labse2_20260927_051906}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CODE_DIR/../.." && pwd)"
HIST="$ROOT/output/history/$RUN"
DOC="$ROOT/Documentation_template.md"
STAGE="$ROOT/submission_build/${TEAM}_submission"
ZIP="$ROOT/${TEAM}_submission.zip"

[[ -f "$HIST/matching_results.tsv" && -f "$HIST/candidate_pairs.tsv" ]] || { echo "missing TSVs in $HIST"; exit 1; }
[[ -f "$DOC" ]] || { echo "missing $DOC"; exit 1; }
if grep -q -E "\[Your Team Name\]|<<TODO" "$DOC"; then
  echo "ERROR: $DOC still has placeholders:"; grep -n -E "\[Your Team Name\]|<<TODO" "$DOC"; exit 1
fi

echo "== validating $RUN with the official validator (--check-ids)"
( cd "$ROOT" && python3 utils/validate_submission.py --matching "$HIST/matching_results.tsv" \
    --candidate "$HIST/candidate_pairs.tsv" --test-dir dataset/test --check-ids )

rm -rf "$STAGE" && mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp "$HIST/matching_results.tsv" "$HIST/candidate_pairs.tsv" "$STAGE/output/"
cp "$DOC" "$STAGE/"
DEST="$STAGE/code/business_entity_resolution"
cp -r "$CODE_DIR/src" "$CODE_DIR/sbatch" "$DEST/"
for f in README.md requirements.txt requirements.lock.txt setup_env.sh run_all.sh make_submission.sh; do
  [[ -f "$CODE_DIR/$f" ]] && cp "$CODE_DIR/$f" "$DEST/"
done
[[ -f "$HIST/decision.json" ]] && cp "$HIST/decision.json" "$DEST/final_decision.json"
find "$DEST" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -f "$DEST/sbatch/.conda_base"

echo "== zipping"
rm -f "$ZIP"
( cd "$STAGE" && python3 -m zipfile -c "$ZIP" output code Documentation_template.md )
python3 - "$ZIP" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
need = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
        "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt"]
missing = [n for n in need if n not in names]
print(f"{len(names)} files, {sum(i.file_size for i in z.infolist()) / 2**20:.0f} MB uncompressed")
print("MISSING: " + ", ".join(missing) if missing else "layout OK")
sys.exit(1 if missing else 0)
EOF
ls -lh "$ZIP"
