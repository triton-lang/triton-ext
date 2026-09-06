#!/bin/sh
# Compile emitted MSL with the real Metal compiler.
#
# Skips when there is no Metal toolchain. A toolchain that is present and
# rejects the output is a failure.
#
# TOOLCHAINS is required on Xcode 27: without it `xcrun metal` reports the
# toolchain as missing even when it is installed.
set -e

generator=$1
out=$2
: "${AGPU_METAL_TOOLCHAIN:=com.apple.dt.toolchain.Metal.32023.917}"

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: no metal compiler on this machine"
  exit 0
fi

"$generator" > "$out.metal"

if ! TOOLCHAINS="$AGPU_METAL_TOOLCHAIN" xcrun metal -c "$out.metal" \
     -o "$out.air" 2> "$out.err"; then
  # A toolchain that is selected but not downloaded is an absent one.
  if grep -q "missing Metal Toolchain" "$out.err"; then
    echo "SKIP: metal toolchain not installed"
    cat "$out.err"
    exit 0
  fi
  echo "the emitted module does not compile:"
  cat "$out.err"
  exit 1
fi

# Most warnings are not defects: the probe has no epilogue consuming every
# register, so -Wunused-variable fires on a correct emission. These are the
# warnings that do mean something is wrong.
defects='uninitialized|out of bounds|undefined behavior'
defects="$defects|incompatible pointer|shift count"
defects="$defects|implicit conversion (from|loses)"

if grep -Eq "warning: .*($defects)" "$out.err"; then
  echo "the emitted module compiles, with warnings that indicate a defect:"
  cat "$out.err"
  exit 1
fi

echo "emitted module compiles: $out.air"
