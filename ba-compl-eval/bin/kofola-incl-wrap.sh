#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 1 \) ] ; then
	echo "usage: ${0} <input-ba> [<params>]"
	exit 1
fi

ABSOLUTE_SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${ABSOLUTE_SCRIPT_PATH}")

A=$1
B=$2
shift
shift
params=("$@")


kofola_exe="${SCRIPT_DIR}/kofola/build/src/kofola"
# capture the full version string (may contain spaces)
kofola_version_string="$("${kofola_exe}" --version 2>/dev/null)"
# extract the last whitespace-separated token (the git hash)
kofola_git_hash=$(awk '{print $NF}' <<< "${kofola_version_string}")
kofola_str=${kofola_git_hash:0:7}

TMP=$(mktemp)

set -o pipefail

"${kofola_exe}" --inclusion ${A} ${B} "${params[@]}" > "${TMP}"
ret=$?

# Inspect the output and print the requested short result line
if grep -q "Inclusion does not hold!" "${TMP}"; then
	echo "${kofola_str}-result: false"
elif grep -q "Inclusion holds!" "${TMP}"; then
	echo "${kofola_str}-result: true"
fi

rm "${TMP}"

exit ${ret}