if [ $# -lt 2 ]; then
	echo "Usage: sh run.sh gpu'n' config-file config-file-2 ..."
	echo "eg: sh run.sh gpu0 nnLM_config.ini"
	exit 
fi
 
echo "RUN" > /tmp/stoppage 	
config_file=$2
gpu=$1
mkdir /var/tmp/ankurgan/$gpu
THEANO_FLAGS='base_compiledir=/var/tmp/ankurgan/'$gpu,'device='$gpu python scripts/TrainNNLM.py $config_file $3 $4 $5 
