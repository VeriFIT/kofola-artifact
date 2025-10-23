#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 1 \) ] ; then
	echo "usage: ${0} <input-ba> [<params>]"
	exit 1
fi

INPUT=$1
shift
params="$*"

# TMP=$(mktemp)
# ./util/ba2hoa.py ${INPUT} > ${TMP} || exit $?

TMP_OUT=$(mktemp)

set -o pipefail
# ./bin/seminator --complement --ba --postprocess-comp=0 ${params} ${TMP} > ${TMP_OUT}
./bin/seminator --complement --ba --postprocess-comp=0 ${params} ${INPUT} > ${TMP_OUT}
ret=$?
# rm ${TMP}

cat ${TMP_OUT} | grep '^States:'
cat ${TMP_OUT} | ./bin/autfilt --high | grep '^States:' | sed 's/^States/autfilt-States/'

rm ${TMP_OUT}

exit ${ret}
