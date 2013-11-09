if [ $# -lt 4 ]; then
	echo "Usage: sh run_test.sh test-data test-feat-OR-None param-dir gpu'n' [outfile] [write-arpa=True/False]"
	echo "eg: sh run.sh test.txt 3gram_param.1 gpu0"
	exit 
fi
 
test=$1
testfeat=$2
param=$3
gpu=$4
outfile=$5
write_arpa=$6 
mkdir -p /var/tmp/ankurgan/$gpu
THEANO_FLAGS='base_compiledir=/var/tmp/ankurgan/'$gpu,'device='$gpu,'exception_verbosity=high' python scripts/TestNNLM.py $test $testfeat $param $outfile $write_arpa
if [ "$outfile" = "" ]
then
	python scripts/ppl.py $test.prob 
else
	python scripts/ppl.py $outfile
fi
