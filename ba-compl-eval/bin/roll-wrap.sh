#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 1 \) ] ; then
	echo "usage: ${0} <input-ba> [<params>]"
	exit 1
fi

INPUT=$1
shift
params="$*"

# TMP="$(mktemp).hoa"
# ./util/ba2hoa.py ${INPUT} > ${TMP} || exit $?

TMP_OUT=$(mktemp)

set -o pipefail
# out=$(java -jar ./bin/ROLL.jar complement ${TMP} -v 0 -table -syntactic ${params} | grep '#H.S' | sed -E 's/^.*([0-9]+).*$/\1/')
# java -jar ./bin/ROLL.jar complement ${TMP} -v 0 -table -syntactic ${params} -out ${TMP_OUT} > /dev/null
java -jar ./bin/ROLL.jar complement ${INPUT} -v 0 -table -syntactic ${params} -out ${TMP_OUT} > /dev/null
ret=$?
# rm ${TMP}

cat ${TMP_OUT} | grep '^States:'
cat ${TMP_OUT} | ./bin/autfilt --high | grep '^States:' | sed 's/^States/autfilt-States/'

rm ${TMP_OUT}

exit ${ret}
