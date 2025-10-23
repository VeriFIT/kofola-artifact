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

cola_exe="${SCRIPT_DIR}/COLA/cola"
cola_str="cola"

TMP=$(mktemp)
"${cola_exe}" "${params[@]}" "${INPUT}" > "${TMP}" || exit 1

cat "${TMP}" | grep "^States:" | sed "s/^States/${cola_str}-states/"
echo "${cola_out}"

rm -f "${TMP}"

exit ${ret}
