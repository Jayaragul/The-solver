#!/usr/bin/env bash
# Fetch a frozen, tractable MIPLIB 2017 subset.
#
# The five instances already vendored under data/miplib2017_small are hard
# enough that HiGHS times out on four of them at 60s, so the set cannot show
# whether a change helped or hurt.  These additions are drawn from the easy end
# of the collection: every one has a published =opt= (or =inf=) value in
# miplib2017-v36.solu, and a competent solver closes them in seconds.  That is
# what makes a regression visible.
#
# Names are fixed in this file, never globbed from the server, so the benchmark
# set cannot drift between runs.
set -u

DEST="${1:-data/miplib2017_small}"
BASE="https://miplib.zib.de/WebData/instances"

# Solvable quickly, spanning pure binary, general integer and mixed models.
INSTANCES="flugpl gt2 blend2 khb05250 neos5 gr4x6 mad p0201 dcmulti bppc4-08 22433 23588 noswot enlight4"

mkdir -p "$DEST" || exit 1
got=0
missing=""

for name in $INSTANCES; do
    target="$DEST/$name.mps"
    if [ -s "$target" ]; then
        echo "have    $name"
        got=$((got + 1))
        continue
    fi
    tmp="$DEST/.$name.mps.gz"
    if ! curl -sfL --max-time 120 -o "$tmp" "$BASE/$name.mps.gz"; then
        echo "MISS    $name (download failed)"
        missing="$missing $name"
        rm -f "$tmp"
        continue
    fi
    # A soft 404 returns an HTML page with status 200, so check the magic
    # bytes rather than trusting the exit code.
    if ! gzip -t "$tmp" 2>/dev/null; then
        echo "MISS    $name (not gzip: soft 404)"
        missing="$missing $name"
        rm -f "$tmp"
        continue
    fi
    gunzip -c "$tmp" > "$target" && rm -f "$tmp"
    if [ -s "$target" ]; then
        echo "fetched $name"
        got=$((got + 1))
    else
        echo "MISS    $name (empty after decompress)"
        missing="$missing $name"
        rm -f "$target"
    fi
done

echo
echo "$got instance(s) available in $DEST"
if [ -n "$missing" ]; then
    echo "not retrieved:$missing"
fi
