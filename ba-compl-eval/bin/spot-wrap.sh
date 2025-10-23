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

autfilt_exe="autfilt"
# capture the full version output (may contain multiple lines)
autfilt_version_output="$("${autfilt_exe}" --version 2>/dev/null)"
# extract the first line
autfilt_first_line=$(awk 'NR==1{print; exit}' <<< "${autfilt_version_output}")
# extract the version token from the first line (last whitespace-separated field)
autfilt_version=$(awk '{print $NF}' <<< "${autfilt_first_line}")
autfilt_str=${autfilt_version}

TMP=$(mktemp)
"${autfilt_exe}" "${params[@]}" "${INPUT}" > "${TMP}" || exit 1

cat "${TMP}" | grep "^States:" | sed "s/^States/${autfilt_str}-states/"
echo "${autfilt_out}"

rm -f "${TMP}"

exit ${ret}
