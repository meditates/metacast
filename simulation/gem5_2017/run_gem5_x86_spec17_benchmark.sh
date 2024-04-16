#!/bin/bash
#
# run_gem5_x86_spec17_benchmark.sh
# Author: Mark Gottscho Email: mgottscho@ucla.edu
# Copyright (C) 2014 Mark Gottscho
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 
############ DIRECTORY VARIABLES: MODIFY ACCORDINGLY #############
GEM5_DIR=/users/meditate/gem5                          # Install location of gem5
SPEC_DIR=/users/meditate/speccpu2017                  # Install location of your SPEC2017 benchmarks
##################################################################
 
ARGC=$# # Get number of arguments excluding arg0 (the script itself). Check for help message condition.
if [[ "$ARGC" != 2 ]]; then # Bad number of arguments.
    echo "run_gem5_x86_spec17_benchmark.sh  Copyright (C) 2014 Mark Gottscho"
   echo "This program comes with ABSOLUTELY NO WARRANTY; for details see <http://www.gnu.org/licenses/>."
   echo "This is free software, and you are welcome to redistribute it under certain conditions; see <http://www.gnu.org/licenses/> for details."
   echo ""
    echo "Author: Mark Gottscho"
    echo "mgottscho@ucla.edu"
    echo ""
    echo "This script runs a single gem5 simulation of a single SPEC CPU2006 benchmark for Alpha ISA."
    echo ""
    echo "USAGE: run_gem5_x86_spec17_benchmark.sh <BENCHMARK> <OUTPUT_DIR>"
    echo "EXAMPLE: ./run_gem5_x86_spec17_benchmark.sh bzip2 /FULL/PATH/TO/output_dir"
    echo ""
    echo "A single --help help or -h argument will bring this message back."
    exit
fi
 
# Get command line input. We will need to check these.
BENCHMARK=$1                    # Benchmark name, e.g. bzip2
OUTPUT_DIR=$2                   # Directory to place run output. Make sure this exists!
 
# Check BENCHMARK input
#################### BENCHMARK CODE MAPPING ######################
BENCHMARK_CODE="none"
 
if [[ "$BENCHMARK" == "perlbench_r" ]]; then
    BENCHMARK_CODE=500.perlbench_r
fi
if [[ "$BENCHMARK" == "gcc_r" ]]; then
    BENCHMARK_CODE=502.gcc_r
fi
if [[ "$BENCHMARK" == "bwaves_r" ]]; then
    BENCHMARK_CODE=503.bwaves_r
fi
if [[ "$BENCHMARK" == "mcf_r" ]]; then
    BENCHMARK_CODE=505.mcf_r
fi
if [[ "$BENCHMARK" == "cactuBSSN_r" ]]; then
    BENCHMARK_CODE=507.cactuBSSN_r
fi
if [[ "$BENCHMARK" == "namd_r" ]]; then
    BENCHMARK_CODE=508.namd_r
fi
if [[ "$BENCHMARK" == "parest_r" ]]; then
    BENCHMARK_CODE=510.parest_r
fi
if [[ "$BENCHMARK" == "povray_r" ]]; then
    BENCHMARK_CODE=511.povray_r
fi
if [[ "$BENCHMARK" == "lbm_r" ]]; then
    BENCHMARK_CODE=519.lbm_r
fi
if [[ "$BENCHMARK" == "omnetpp_r" ]]; then
    BENCHMARK_CODE=520.omnetpp_r 
fi
if [[ "$BENCHMARK" == "wrf_r" ]]; then
    BENCHMARK_CODE=521.wrf_r
fi
if [[ "$BENCHMARK" == "xalancbmk_r" ]]; then
    BENCHMARK_CODE=523.xalancbmk_r
fi
if [[ "$BENCHMARK" == "x264_r" ]]; then
    BENCHMARK_CODE=525.x264_r
fi
if [[ "$BENCHMARK" == "blender_r" ]]; then 
    BENCHMARK_CODE=526.blender_r 
fi
if [[ "$BENCHMARK" == "cam4_r" ]]; then
    BENCHMARK_CODE=527.cam4_r
fi
if [[ "$BENCHMARK" == "deepsjeng_r" ]]; then
    BENCHMARK_CODE=531.deepsjeng_r
fi
if [[ "$BENCHMARK" == "imagick_r" ]]; then
    BENCHMARK_CODE=538.imagick_r
fi
if [[ "$BENCHMARK" == "leela_r" ]]; then
    BENCHMARK_CODE=541.leela_r
fi
if [[ "$BENCHMARK" == "nab_r" ]]; then
    BENCHMARK_CODE=544.nab_r
fi
if [[ "$BENCHMARK" == "exchange2_r" ]]; then
    BENCHMARK_CODE=548.exchange2_r
fi
if [[ "$BENCHMARK" == "fotonik3d_r" ]]; then
    BENCHMARK_CODE=549.fotonik3d_r
fi
if [[ "$BENCHMARK" == "roms_r" ]]; then
    BENCHMARK_CODE=554.roms_r
fi
if [[ "$BENCHMARK" == "xz_r" ]]; then
    BENCHMARK_CODE=557.xz_r
fi
if [[ "$BENCHMARK" == "perlbench_s" ]]; then
    BENCHMARK_CODE=600.perlbench_s
fi
if [[ "$BENCHMARK" == "gcc_s" ]]; then
    BENCHMARK_CODE=602.gcc_s 
fi
if [[ "$BENCHMARK" == "bwaves_s" ]]; then
    BENCHMARK_CODE=603.bwaves_s
fi
if [[ "$BENCHMARK" == "mcf_s" ]]; then
    BENCHMARK_CODE=605.mcf_s
fi
if [[ "$BENCHMARK" == "cactuBSSN_s" ]]; then
    BENCHMARK_CODE=607.cactuBSSN_s
fi
if [[ "$BENCHMARK" == "lbm_s" ]]; then 
    BENCHMARK_CODE=619.lbm_s
fi
if [[ "$BENCHMARK" == "omnetpp_s" ]]; then
    BENCHMARK_CODE=620.omnetpp_s
fi
if [[ "$BENCHMARK" == "wrf_s" ]]; then
    BENCHMARK_CODE=621.wrf_s
fi
if [[ "$BENCHMARK" == "xalancbmk_s" ]]; then
    BENCHMARK_CODE=623.xalancbmk_s
fi
if [[ "$BENCHMARK" == "x264_s" ]]; then
    BENCHMARK_CODE=625.x264_s
fi
if [[ "$BENCHMARK" == "cam4_s" ]]; then
    BENCHMARK_CODE=627.cam4_s
fi
if [[ "$BENCHMARK" == "pop2_s" ]]; then
    BENCHMARK_CODE=628.pop2_s
fi
if [[ "$BENCHMARK" == "deepsjeng_s" ]]; then
    BENCHMARK_CODE=631.deepsjeng_s
fi
if [[ "$BENCHMARK" == "imagick_s" ]]; then
    BENCHMARK_CODE=638.imagick_s
fi
if [[ "$BENCHMARK" == "leela_s" ]]; then
    BENCHMARK_CODE=641.leela_s 
fi
if [[ "$BENCHMARK" == "nab_s" ]]; then
    BENCHMARK_CODE=644.nab_s
fi
if [[ "$BENCHMARK" == "exchange2_s" ]]; then
    BENCHMARK_CODE=648.exchange2_s
fi
if [[ "$BENCHMARK" == "fotonik3d_s" ]]; then
    BENCHMARK_CODE=649.fotonik3d_s
fi
if [[ "$BENCHMARK" == "roms_s" ]]; then 
    BENCHMARK_CODE=654.roms_s
fi
if [[ "$BENCHMARK" == "xz_s" ]]; then
    BENCHMARK_CODE=657.xz_s
fi
if [[ "$BENCHMARK" == "specrand_fs" ]]; then
    BENCHMARK_CODE=996.specrand_fs
fi
if [[ "$BENCHMARK" == "specrand_fr" ]]; then 
    BENCHMARK_CODE=997.specrand_fr
fi
if [[ "$BENCHMARK" == "specrand_is" ]]; then
    BENCHMARK_CODE=998.specrand_is
fi
if [[ "$BENCHMARK" == "specrand_ir" ]]; then
    BENCHMARK_CODE=999.specrand_ir
fi

# Sanity check
if [[ "$BENCHMARK_CODE" == "none" ]]; then
    echo "Input benchmark selection $BENCHMARK did not match any known SPEC CPU2017 benchmarks! Exiting."
    exit 1
fi
##################################################################
 
# Check OUTPUT_DIR existence
if [[ !(-d "$OUTPUT_DIR") ]]; then
    echo "Output directory $OUTPUT_DIR does not exist! Exiting."
    exit 1
fi
 
if [[ "${BENCHMARK:0-1:1}" == "r" ]]; then
    RUN_DIR=$SPEC_DIR/benchspec/CPU/$BENCHMARK_CODE/run/run_base_refrate_mytest-m64.0000     # Run directory for the selected SPEC benchmark
fi
if [[ "${BENCHMARK:0-1:1}" == "s" ]]; then
    RUN_DIR=$SPEC_DIR/benchspec/CPU/$BENCHMARK_CODE/run/run_base_refspeed_mytest-m64.0000     # Run directory for the selected SPEC benchmark
fi
SCRIPT_OUT=$OUTPUT_DIR/runscript.log                                                                    # File log for this script's stdout henceforth
 
################## REPORT SCRIPT CONFIGURATION ###################
 
echo "Command line:"                                | tee $SCRIPT_OUT
echo "$0 $*"                                        | tee -a $SCRIPT_OUT
echo "================= Hardcoded directories ==================" | tee -a $SCRIPT_OUT
echo "GEM5_DIR:                                     $GEM5_DIR" | tee -a $SCRIPT_OUT
echo "SPEC_DIR:                                     $SPEC_DIR" | tee -a $SCRIPT_OUT
echo "==================== Script inputs =======================" | tee -a $SCRIPT_OUT
echo "BENCHMARK:                                    $BENCHMARK" | tee -a $SCRIPT_OUT
echo "OUTPUT_DIR:                                   $OUTPUT_DIR" | tee -a $SCRIPT_OUT
echo "==========================================================" | tee -a $SCRIPT_OUT
##################################################################
 
 
#################### LAUNCH GEM5 SIMULATION ######################
echo ""
echo "Changing to SPEC benchmark runtime directory: $RUN_DIR" | tee -a $SCRIPT_OUT
cd $RUN_DIR
 
echo "" | tee -a $SCRIPT_OUT
echo "" | tee -a $SCRIPT_OUT
echo "--------- Here goes nothing! Starting gem5! ------------" | tee -a $SCRIPT_OUT
echo "" | tee -a $SCRIPT_OUT
echo "" | tee -a $SCRIPT_OUT
 
# Actually launch gem5!
$GEM5_DIR/build/X86/gem5.fast --outdir=$OUTPUT_DIR $GEM5_DIR/configs/example/spec17_config.py --benchmark=$BENCHMARK --benchmark_stdout=$OUTPUT_DIR/$BENCHMARK.out --simpoint-profile --simpoint-interval=100000000 --cpu-type=NonCachingSimpleCPU  --benchmark_stderr=$OUTPUT_DIR/$BENCHMARK.err | tee -a $SCRIPT_OUT



