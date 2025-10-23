#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 2 \) ] ; then
	echo "usage: ${0} <input-ba> [<params>]"
	exit 1
fi

A=$1
B=$2
shift
shift
params=("$@")

ABSOLUTE_SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${ABSOLUTE_SCRIPT_PATH}")

bait_exe="java -jar ${SCRIPT_DIR}/BAIT/bait.jar"
bait_str="bait"

TMP=$(mktemp)
${bait_exe} -a "$A" -b "$B" "${params[@]}" > "${TMP}"
ret=$?

# Prefer to detect explicit inclusion result strings in the tool output
# and print a normalized `result: true` / `result: false`. 
if grep -qF "Inclusion holds: true" "${TMP}" 2>/dev/null; then
	echo "${bait_str}-result: true"
elif grep -qF "Inclusion holds: false" "${TMP}" 2>/dev/null; then
	echo "${bait_str}-result: false"
else
	echo "${bait_str}-result: $(cat "${TMP}")"
fi

rm "${TMP}"
exit 0