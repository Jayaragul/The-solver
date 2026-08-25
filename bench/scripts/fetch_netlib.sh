#!/usr/bin/env bash
# Fetch and expand the Netlib LP test set.
#
# Netlib ships these instances in a compressed MPS dialect; netlib's own
# emps.c is the reference expander, so it is compiled here rather than
# reimplemented.  The reference optimal values are parsed out of netlib's
# PROBLEM SUMMARY TABLE and written to optima.txt, which the benchmark
# harness treats as ground truth.
set -u

DATA="${1:-$HOME/sk/data/netlib}"
BASE="https://netlib.org/lp/data"
mkdir -p "$DATA/mps"
cd "$DATA" || exit 1

if [ ! -f readme ]; then
    curl -sS --max-time 120 -O "$BASE/readme" || { echo "cannot fetch readme"; exit 1; }
fi

# --- reference optima, from the summary table ------------------------------
awk '
  /^Name +Rows +Cols/ { intable=1; next }
  intable && /^[A-Z0-9]+ +[0-9]/ {
      name=tolower($1)
      val=""
      for (i=NF; i>=1; i--) if ($i ~ /^-?[0-9.]+[Ee][+-][0-9]+$/) { val=$i; break }
      if (val != "") print name, val
  }
' readme | sort -u > optima.txt
echo "reference optima: $(wc -l < optima.txt)"

# --- expander --------------------------------------------------------------
if [ ! -x ./emps ]; then
    [ -f emps.c ] || curl -sS --max-time 120 -O "$BASE/emps.c"
    cc -O2 -w -o emps emps.c || { echo "emps build failed"; exit 1; }
fi

# --- instances -------------------------------------------------------------
ok=0; skip=0; fail=0
while read -r name val; do
    out="mps/$name.mps"
    [ -s "$out" ] && { ok=$((ok+1)); continue; }
    if [ ! -s "raw/$name" ]; then
        mkdir -p raw
        curl -sS --max-time 180 -o "raw/$name" "$BASE/$name" || true
    fi
    # netlib returns an HTML error page for names held in sub-archives
    if [ ! -s "raw/$name" ] || head -c 200 "raw/$name" | grep -qi '<html\|not found'; then
        rm -f "raw/$name"; skip=$((skip+1)); continue
    fi
    if ./emps "raw/$name" > "$out" 2>/dev/null && [ -s "$out" ]; then
        ok=$((ok+1))
    else
        rm -f "$out"; fail=$((fail+1))
    fi
done < optima.txt

echo "expanded=$ok unavailable=$skip failed=$fail"
ls mps | wc -l
