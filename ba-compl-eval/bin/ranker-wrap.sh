#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 1 \) ] ; then
	echo "usage: ${0} <input-ba> [<params>]"
	exit 1
fi

ABSOLUTE_SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${ABSOLUTE_SCRIPT_PATH}")

INPUT=$1
shift
# preserve argument boundaries and spacing
params=("$@")

ranker_exe="${SCRIPT_DIR}/ranker/build/ranker"
ranker_str="ranker"

# for the backoff
export SPOTEXE="/usr/local/bin/autfilt"
TMP=$(mktemp)
cat "${INPUT}" | "${SPOTEXE}" --split-edges > "${TMP}" 
"${ranker_exe}" "${params[@]}" "${TMP}"  | grep "^States:" | sed "s/^States/${ranker_str}-states/" || exit 1

#cat "${TMP}" | grep "^States:" | sed "s/^States/${ranker_str}-states/"

rm -f "${TMP}"

exit ${ret}
