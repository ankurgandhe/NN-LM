if [ $# -ne 2 ]; then
	echo "Usage: sh run.sh config-file gpu'n'"
	echo "eg: sh run.sh nnLM_config.ini gpu0"
	exit 
fi
 
vars=$1	
gpu=$2
mkdir /var/tmp/ankurgan/$gpu
THEANO_FLAGS='base_compiledir=/var/tmp/ankurgan/'$gpu,'device='$gpu python scripts/mlp_test3.py $vars
python scripts/ppl.py $vars 
