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

forklift_exe="java -jar ${SCRIPT_DIR}/FORKLIFT/forklift.jar"
forklift_str="forklift"

TMP=$(mktemp)
${forklift_exe} "$A" "$B" "${params[@]}" > "${TMP}"
ret=$?

# print result flag based on exit code while preserving the original exit code
if [ "${ret}" -eq 0 ]; then
	echo "${forklift_str}-result: true"
elif [ "${ret}" -eq 1 ]; then
	echo "${forklift_str}-result: false"
else 
	echo "${forklift_str}-result: `cat ${TMP}`"
fi

rm ${TMP}
exit 0