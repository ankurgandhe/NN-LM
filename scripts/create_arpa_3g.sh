if [ $# -lt 4 ]; then
        echo "Usage: sh create_arpa.sh model-dir srilm bigramNNLM1-pruned bigramNNLM2-pruned bigramNNLM-unpruned gpu'n' "
        echo "eg: sh create_arpa.sh 2gram_param srilm.arap nnlm+srilm.pruned.arpa gpu0"
        exit
fi

model=$1
srilm=$2
bigramNNLM=$3
bigramNNLM2=$4
bigramNNLMunpruned=$5
gpu=$6
echo "Creating all tri-gram data.." 
#python scripts/arpa/Create3gram.py $bigramNNLM $bigramNNLM2 $model  > $model/allphrases.3g.txt.id  2> $model/allphrases.3g.txt 

echo "calculating probabilities using NNLM..." 
#sh scripts/run_test.sh $model/allphrases.3g.txt.id None $model $gpu $model/allphrases.3g.outprob 1 

echo "Combining 1gram from SRILM, 2-gram from unpruned bigramNNLM and 3-grams from NNLM.." 
tail_size=`grep -n '2-grams' $srilm | cut -d ':' -f1`
head_size=`grep -n '1-grams' $srilm | cut -d ':' -f1`
head_size=`expr $head_size`
tail_size=`expr $tail_size - 2`
tot=`expr $tail_size - $head_size`
echo "1gram:" $head_size $tail_size $tot
cat $srilm | head -$tail_size | tail -$tot > $model/allphrases.1g.lm
 
tail_size=`grep -n '3-grams' $bigramNNLMunpruned | cut -d ':' -f1`
head_size=`grep -n '2-grams' $bigramNNLMunpruned | cut -d ':' -f1`
head_size=`expr $head_size`
tail_size=`expr $tail_size - 2`
tot=`expr $tail_size - $head_size`
echo "2gram:" $head_size $tail_size $tot
cat $bigramNNLMunpruned | head -$tail_size | tail -$tot > $model/allphrases.2g.lm

echo "Creating 3gram ( might take a long time):" 
#paste -d '\t'  $model/allphrases.3g.outprob $model/allphrases.3g.txt  > $model/allphrases.3g.lm
len1g=`wc $model/allphrases.1g.lm | awk '{print $1}'`
len2g=`wc $model/allphrases.2g.lm | awk '{print $1}'`
len3g=`wc $model/allphrases.3g.lm | awk '{print $1}'`
echo "Writing LM.. ", $len1g, $len2g, $len3g
python scripts/arpa/Create_3gram_lm.py $model/allphrases.1g.lm $len1g  $model/allphrases.2g.lm $len2g  $model/allphrases.3g.lm $len3g  > $model/nnlm.3g.arpa

#sh interpolate_arpa.sh $model $srilm data/dev.clean.txt 
#echo "Find best mix for SRILM and NNLM" 
#sh ~/tools/srilm/test_srilm.sh $srilm data/dev.clean.txt 3  > $model/srilm.ppl
#sh ~/tools/srilm/test_srilm.sh $model/nnlm.arpa data/dev.clean.txt 3 > $model/nnlm.arpa.ppl

#$SRILM/compute-best-mix $model/srilm.ppl $model/nnlm.arpa.ppl > $model/bestmix.log 2>&1 


