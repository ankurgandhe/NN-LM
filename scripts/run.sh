if [ $# -ne 2 ]; then
	echo "Usage: sh run.sh config-file gpu'n'"
	echo "eg: sh run.sh nnLM_config.ini gpu0"
	exit 
fi
 
echo "RUN" > stoppage 	
config_file=$1
gpu=$2
mkdir /var/tmp/ankurgan/$gpu
THEANO_FLAGS='base_compiledir=/var/tmp/ankurgan/'$gpu,'device='$gpu python scripts/TrainNNLM.py $config_file
