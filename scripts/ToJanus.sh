if [ $# -ne 1 ]; then
	echo "Usage : sh ToJanus.sh param-dir " 
	exit
fi 

paramdir=$1
mv $paramdir/hW0.mat $paramdir/hW.mat 
mv $paramdir/hB0.mat $paramdir/hB.mat
sed 's/ /\n/g' $paramdir/vocab.nnid > $paramdir/vocab.nnid.janus 
