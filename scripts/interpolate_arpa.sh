if [ $# -lt 3 ]; then
        echo "Usage: sh interpolate_arpa.sh model-dir srilm dev-data "
        echo "eg: sh interpolate_arpa.sh 2gram_param srilm.arap dev-data"
        exit
fi

model=$1
srilm=$2
devdata=$3
SRILM=/home/speech/Tools/srilm-1.5.11/bin/i686-m64 

#cp $model/nnlm.arpa.original $model/nnlm.arpa 
echo "Pruning NNLM only"
$SRILM/ngram -order 3 -debug 2 -lm $model/nnlm.arpa -prune 1e-50 -write-lm $model/nnlm.pruned.arpa -map-unk "<UNK>"
sh ~/tools/srilm/test_srilm.sh $model/nnlm.pruned.arpa $devdata 3 > $model/nnlm.pruned.arpa.ppl
#cp $model/nnlm.pruned.arpa $model/nnlm.arpa

echo "Find best mix for SRILM and NNLM"
sh ~/tools/srilm/test_srilm.sh $srilm $devdata 3  > $model/srilm.ppl
sh ~/tools/srilm/test_srilm.sh $model/nnlm.arpa $devdata 3 > $model/nnlm.arpa.ppl

echo "Finding best mix "
$SRILM/compute-best-mix $model/srilm.ppl $model/nnlm.arpa.ppl > $model/bestmix.log 2>&1
#write script to read and find best lamnda. For now, 0.5 0.5 
lambdaSRILM=0.3
lambdaNNLM=0.7
echo "Interpolatng LMs" 
$SRILM/ngram -order 3 -debug 2 -unk -lm $srilm -lambda $lambdaSRILM -mix-lm $model/nnlm.arpa -write-lm $model/nnlm+srilm.arpa -map-unk "<UNK>"
sh ~/tools/srilm/test_srilm.sh $model/nnlm+srilm.arpa $devdata 3 > $model/nnlm+srilm.arpa.ppl

echo "Pruning combined LM"
$SRILM/ngram -order 3 -debug 2 -lm $model/nnlm+srilm.arpa -prune 1e-20 -write-lm $model/nnlm+srilm.pruned.arpa -map-unk "<UNK>" 
sh ~/tools/srilm/test_srilm.sh $model/nnlm+srilm.pruned.arpa $devdata 3 > $model/nnlm+srilm.pruned.arpa.ppl 

