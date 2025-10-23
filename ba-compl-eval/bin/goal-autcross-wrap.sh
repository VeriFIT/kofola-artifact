#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 2 \) ] ; then
	echo "usage: ${0} <input-ba> <output-ba> [params]"
	exit 1
fi

INPUT=$1
OUTPUT=$2
shift
shift
params="$*"

./bin/goal/gc batch "load \$aut \$1; \$compl = complement --option \$3 \$aut; save -c hoaf \$compl \$2;" ${INPUT} ${OUTPUT} "${params}"
