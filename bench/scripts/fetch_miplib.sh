#!/usr/bin/env bash
# Fetch a frozen subset of MIPLIB instances for MILP benchmarking.
#
# The subset is chosen for tractability rather than difficulty: these are the
# small and medium members of the collection, so a run finishes in minutes and
# regressions are visible.  Names are fixed in this file so the benchmark set
# cannot drift between runs.
set -u

DATA="${1:-$HOME/sk/data/miplib}"
# The classic MIPLIB instances used by this frozen, tractable subset are
# maintained in the official MIPLIB archive.  The newer MIPLIB 2017 site
# serves bulk ZIP files rather than this per-instance URL layout.
BASE="https://miplib2010.zib.de/miplib3/miplib3"
mkdir -p "$DATA/mps"
cd "$DATA" || exit 1

INSTANCES="
flugpl egout enigma bell5 khb05250 lseu misc03 mod008 p0033 p0201 p0548
stein27 stein45 vpm1 vpm2 gt2 rgn dcmulti fixnet6 l152lav blend2 dsbmip
gen-ip054 gen-ip002 markshare_4_0 neos5 pk1 mas76 mas74 timtab1 noswot
qnet1 qnet1_o set1ch rout fiber gesa2 gesa3 harp2 mod010 mod011 p2756 qiu
"

ok=0; skip=0
for name in $INSTANCES; do
    out="mps/$name.mps"
    [ -s "$out" ] && { ok=$((ok+1)); continue; }
    if [ ! -s "raw/$name.mps.gz" ]; then
        mkdir -p raw
        curl -sS -f --max-time 180 -o "raw/$name.mps.gz" "$BASE/$name.mps.gz" 2>/dev/null || true
    fi
    if [ ! -s "raw/$name.mps.gz" ]; then
        rm -f "raw/$name.mps.gz"; skip=$((skip+1)); continue
    fi
    if gunzip -c "raw/$name.mps.gz" > "$out" 2>/dev/null && [ -s "$out" ]; then
        ok=$((ok+1))
    else
        rm -f "$out"; skip=$((skip+1))
    fi
done

echo "miplib: available=$ok unavailable=$skip"
ls -1 mps 2>/dev/null | wc -l
