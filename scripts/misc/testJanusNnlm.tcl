LingKS lm NNLM
#lm load /data/ASR1/ankurgan/BABEL/Vietnamese-Full/NNLM/2gram.param.9/janus.nnlm.bin
lm load /data/ASR1/ankurgan/BABEL/Vietnamese-Full/NNLM/3gram.param.opt4.1/janus.nnlm.bin
#LingKS lm NGramLM 
#lm load /data/ASR1/ankurgan/BABEL/Vietnamese-Full/NNLM/nnlm2g+srilm2g.arpa.v9 

set fp [open "/data/ASR1/ankurgan/BABEL/Vietnamese-Limited/LM/data/train.clean.txt" r]
fconfigure $fp -encoding utf-8

set file_data [read $fp]
set data [split $file_data "\n"]
foreach line $data {
	#puts $line 
	set x [lm score $line]
	# -array 1] 
	puts $x 
     }

#set x [lm score "quen ra"]
#puts ""
#lm score "quen ra cung"

#lm score "quen cung"

#lm score "hello" 
#puts $x 

