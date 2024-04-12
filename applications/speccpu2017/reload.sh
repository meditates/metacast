#!/bin/bash
NUMBER=$1
NAME=$2
INDEX=$3
l1i_size=$4
l1d_size=$5
l1i_assoc=$6
l1d_assoc=$7
l2_size=$8
l2_assoc=$9
l3_size=${10}
l3_assoc=${11}
cacheline_size=${12}
mem_type=${13}
cpu_clock=${14}
fetchWidth=${15}
fetchBufferSize=${16}
fetchQueueSize=${17}
decodeWidth=${18}
LQEntries=${19}
SQEntries=${20}
numROBEntries=${21}

SE_OUT_DIR_CHECKPOINT=/users/meditate/speccpu2017/benchspec/CPU/${NUMBER}.${NAME}/run/run_base_refrate_mytest-m64.0000/m5out/${NAME}
SE_OUT_DIR_RELOAD=/users/meditate/RELOAD/${NAME}${INDEX}  
SH_ROUTE=/users/meditate/
OUTPUT=/users/meditate/out/${NAME}${INDEX}
mkdir -p ${OUTPUT}
times=$(ls ${SE_OUT_DIR_CHECKPOINT} | grep simpoint | wc -l)
#times=1
for ((i=1;i<=times;i++))
#   对每一个 checkpoint 做 reload
do cd ${SH_ROUTE};nohup time bash sim_reload.sh ${NUMBER} ${NAME} ${INDEX} \
${l1i_size} ${l1d_size} ${l1i_assoc} ${l1d_assoc} ${l2_size} ${l2_assoc} \
${l3_size} ${l3_assoc} ${cacheline_size} ${mem_type} ${mem_size} ${cpu_clock} \
${fetchWidth} ${fetchBufferSize} ${fetchQueueSize} ${decodeWidth} ${LQEntries} \
${SQEntries} ${numROBEntries} \
$i ${SE_OUT_DIR_RELOAD} > ${OUTPUT}/$i.txt &
done

