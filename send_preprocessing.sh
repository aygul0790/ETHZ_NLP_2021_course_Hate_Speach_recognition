 #!/bin/bash

 script_path=/cluster/home/gpatoulidis/data_code/Preprocessing.py
 memory=7500
 cores=20
 time_=24

 source /cluster/home/gpatoulidis/software/pyenv/bin/activate
 export LD_LIBRARY_PATH=/lib:/usr/lib:/cluster/apps/python/3.7.1/x86_64/lib64/
 echo "python $script_path" | bsub -J georgnlp -n ${cores} -W ${time_}:00 -R "rusage[mem=${memory}]" -R "span[hosts=1]" #-o ${outputdir}/nlp.log 
