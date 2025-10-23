#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 1 \) ] ; then
	echo "usage: ${0} <input-ba> [<params>]"
	exit 1
fi

INPUT=$1
shift
params="$*"

GOAL_DIR=./bin/goal

GOAL_TMP_DIR="$(mktemp -d)"
cp -r ${GOAL_DIR} ${GOAL_TMP_DIR}

# TMP="$(mktemp).gff"
# ./util/ba2gff.py ${INPUT} > ${TMP} || exit $?
TMP=${INPUT}

TIME_TMP="$(mktemp)"
set -o pipefail
# out=$(time ${GOAL_TMP_DIR}/goal/gc complement ${params} ${TMP} ${TIME_TMP} | grep -i "<state sid" | wc -l)

GOAL_TMP="$(mktemp)"
#this was working
#out=$(/usr/bin/time -p ${GOAL_TMP_DIR}/goal/gc complement ${params} ${TMP} 2>${TIME_TMP} | grep -i "<state sid" | wc -l)
# /usr/bin/time -p ${GOAL_TMP_DIR}/goal/gc batch "load \$aut \$1; \$compl = complement --option \$3 \$aut; save -c hoaf \$compl \$2;" ${TMP} ${GOAL_TMP} "${params}" 2> ${TIME_TMP}
/usr/bin/time -p ${GOAL_TMP_DIR}/goal/gc batch "load -c hoaf \$aut \$1; \$compl = complement --option \$3 \$aut; save -c hoaf \$compl \$2;" ${INPUT} ${GOAL_TMP} "${params}" 2> ${TIME_TMP}
ret=$?
# rm ${TMP}
rm -rf ${GOAL_TMP_DIR}

./bin/autfilt --high --ba ${GOAL_TMP} | grep "States:" | sed "s/States/autfilt-States/"

cat ${GOAL_TMP} | grep "States:"
cat ${TIME_TMP} | grep "user" | sed "s/user/Time:/"
rm ${GOAL_TMP}
rm ${TIME_TMP}

exit ${ret}
