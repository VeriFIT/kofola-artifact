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

rabit_exe="java -jar ${SCRIPT_DIR}/rabit/RABIT.jar"
# Parse version from the first line of --help output, e.g., "RABIT v2.5.1." -> "2.5.1"
# We capture stderr as some Java apps print help to stderr
rabit_help_first_line=$(${rabit_exe} --help 2>&1 | head -n1)
# Try to extract the version following a leading 'v'
rabit_version=$(echo "${rabit_help_first_line}" | sed -n 's/.*v\([0-9][0-9.]*\).*/\1/p')
# Remove a possible trailing dot (e.g., "2.5.1." -> "2.5.1")
rabit_version=${rabit_version%.}
# Fallback: extract the first x.y[.z] number pattern if the above failed
if [ -z "${rabit_version}" ]; then
	rabit_version=$(echo "${rabit_help_first_line}" | grep -oE '[0-9]+(\.[0-9]+)*' | head -n1)
fi
# Use parsed version if available, else default label
rabit_str=${rabit_version:-rabit}

TMP=$(mktemp)
${rabit_exe} "$A" "$B" "${params[@]}" > "${TMP}"
ret=$?

# read temporary output and interpret known messages
result_text=$(cat "${TMP}")

# If the tool prints an explicit inclusion/exclusion message prefer that over exit code
if echo "${result_text}" | grep -qF "Not included."; then
	echo "${rabit_str}-result: false"
elif echo "${result_text}" | grep -qF "Included"; then
	echo "${rabit_str}-result: true"
else
	echo "${rabit_str}-result: ${result_text}"
fi

rm ${TMP}
exit 0