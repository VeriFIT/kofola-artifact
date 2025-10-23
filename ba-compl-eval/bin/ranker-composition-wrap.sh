#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -ne 1 \) ] ; then
	echo "usage: ${0} <input-ba>"
	exit 1
fi

INPUT=$1

TMP=$(mktemp)
TMP_STAT=$(mktemp)
./bin/ranker-composition --stats ${INPUT} > ${TMP} 2> ${TMP_STAT} || exit 1

set -o pipefail
autfilt_out=$(./bin/autfilt --high ${TMP} | grep "^States:" | sed 's/^States/autfilt-States/')
ret=$?
rm ${TMP}

cat ${TMP_STAT} | sed 's/^Generated states/nopost-States/' | sed 's/^Generated trans/nopost-Transitions/'
echo ${autfilt_out}

rm ${TMP_STAT}

exit ${ret}
