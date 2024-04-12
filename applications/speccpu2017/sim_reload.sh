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

SE_OUT_DIR_RELOAD=/users/meditate/RELOAD/${NAME}${INDEX} 
SE_OUT_DIR_CHECKPOINT=/users/meditate/speccpu2017/benchspec/CPU/${NUMBER}.${NAME}/run/run_base_refrate_mytest-m64.0000/m5out/${NAME}
cd /users/meditate/speccpu2017/benchspec/CPU/${NUMBER}.${NAME}/run/run_base_refrate_mytest-m64.0000/

/users/meditate/gem5/build/X86/gem5.fast --outdir=${SE_OUT_DIR_RELOAD}/${22} \
/users/meditate/gem5/configs/example/spec17_config.py \
--benchmark=${NAME} \
--cpu-type=DerivO3CPU \
--caches \
--l2cache \
--l3cache \
--l1i_size=${l1i_size}kB \
--l1d_size=${l1d_size}kB \
--l1i_assoc=${l1i_assoc} \
--l1d_assoc=${l1d_assoc} \
--l2_size=${l2_size}kB \
--l2_assoc=${l2_assoc} \
--l3_size=${l3_size}MB \
--l3_assoc=${l3_assoc} \
--cacheline_size=${cacheline_size} \
--mem-type=${mem_type} \
--mem-size=16GB \
--cpu-clock=${cpu_clock}GHz \
--fetchWidth=${fetchWidth} \
--fetchBufferSize=${fetchBufferSize} \
--fetchQueueSize=${fetchQueueSize} \
--decodeWidth=${decodeWidth} \
--LQEntries=${LQEntries} \
--SQEntries=${SQEntries} \
--numROBEntries=${numROBEntries} \
--benchmark_stdout=/users/meditate/result_${NAME}/${NAME}${INDEX}_check.out \
--benchmark_stderr=/users/meditate/result_${NAME}/${NAME}${INDEX}_check.err \
--restore-simpoint-checkpoint -r ${22} --checkpoint-dir ${SE_OUT_DIR_CHECKPOINT}

