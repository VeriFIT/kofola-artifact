#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 1 \) ] ; then
	echo "usage: ${0} <input-ba> [<params>]"
	exit 1
fi

INPUT=$1
shift
params="$*"

TMP="$(mktemp).gff"
./util/ba2gff.py ${INPUT} > ${TMP} || exit $?

set -o pipefail
out=$(./bin/goal/gc complement ${params} ${TMP} | grep -i "<state sid" | wc -l)
ret=$?
rm ${TMP}

echo "States: ${out}"

exit ${ret}
