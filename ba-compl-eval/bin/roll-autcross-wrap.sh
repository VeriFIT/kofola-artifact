#!/bin/bash

# Check the number of command-line arguments
if [ \( "$#" -lt 2 \) ] ; then
	echo "usage: ${0} <input-ba> <output-ba>"
	exit 1
fi

INPUT=$1
OUTPUT=$2

TMP=$(mktemp).hoa
cp ${INPUT} ${TMP}
java -jar ./bin/ROLL.jar complement ${TMP} -v 0 -table -syntactic -out ${OUTPUT} > /dev/null
