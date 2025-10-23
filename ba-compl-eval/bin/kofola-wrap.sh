#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 1 \) ] ; then
	echo "usage: ${0} <input-ba> [params]"
	exit 1
fi

ABSOLUTE_SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${ABSOLUTE_SCRIPT_PATH}")

INPUT=$1
shift
# preserve argument boundaries and spacing
params=("$@")

# Remove any parameter that contains --high or --check (kofola doesn't accept them)
kofola_params=()
has_high=0
has_check=0
for p in "${params[@]}"; do
    if [[ "$p" == *--high* ]]; then
        has_high=1
        # skip this parameter when invoking kofola
        continue
    fi
    if [[ "$p" == *--check* ]]; then
        has_check=1
        # skip this parameter when invoking kofola
        continue
    fi
    kofola_params+=("$p")
done

kofola_exe="${SCRIPT_DIR}/kofola/build/src/kofola"
# capture the full version string (may contain spaces)
kofola_version_string="$("${kofola_exe}" --version 2>/dev/null)"
# extract the last whitespace-separated token (the git hash)
kofola_git_hash=$(awk '{print $NF}' <<< "${kofola_version_string}")
kofola_str=${kofola_git_hash:0:7}

TMP=$(mktemp)

# make sure pipeline failures are detected
set -o pipefail

if [ "$has_high" -eq 1 ]; then
    "${kofola_exe}" "${kofola_params[@]}" "${INPUT}" | autfilt --high > "${TMP}"
else
    "${kofola_exe}" "${kofola_params[@]}" "${INPUT}" > "${TMP}"
fi

# capture return code
ret=$?

# prefix the States header with the short git hash and print the full output
cat "${TMP}" | grep "^States:" | sed "s/^States/$kofola_str-states/"

# if --check is specified, check correctness using autcross
if [ "$has_check" -eq 1 ]; then
    TIMEOUT=100
    AUTCROSS_CMD="autcross"
    CHECK_TMP=$(mktemp)
    
    # Use autcross to compare kofola output with autfilt --complement
    cat "${INPUT}" | timeout ${TIMEOUT} ${AUTCROSS_CMD} "a=%H; cat ${TMP} > %O" 'autfilt --complement %H > %O' > "${CHECK_TMP}" 2>&1

    check_ret=$?
    if [ ${check_ret} -eq 0 ]; then
        echo "check: True"
    elif [ ${check_ret} -eq 124 ]; then
        echo "check: TO"
    else
        echo "check: False"
    fi
    
    rm -f "${CHECK_TMP}"
fi

rm -f "${TMP}"

exit ${ret}
