# Copyright (c) 2012-2013 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Copyright (c) 2006-2008 The Regents of The University of Michigan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Simple test script
#
# "m5 test.py"

from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import os

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.params import NULL
from m5.util import addToPath, fatal, warn

addToPath('../')

from ruby import Ruby

from common import Options
from common import Simulation
from common import CacheConfig
from common import CpuConfig
from common import ObjectList
from common import MemConfig
from common.FileSystemConfig import config_filesystem
from common.Caches import *
from common.cpu2000 import *
import spec17_benchmarks

# ...snip...
parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)


# NAVIGATE TO THIS POINT

# ...snip...

parser.add_argument("-b", "--benchmark", type=str, default="", help="The SPEC benchmark to be loaded.")
parser.add_argument("--benchmark_stdout", type=str, default="", help="Absolute path for stdout redirection for the benchmark.")
parser.add_argument("--benchmark_stderr", type=str, default="", help="Absolute path for stderr redirection for the benchmark.")
parser.add_argument("--fetchWidth", type=int, default=8, help="fetch width.")
parser.add_argument("--fetchBufferSize", type=int, default=64, help="fetch buffer size.")
parser.add_argument("--fetchQueueSize", type=int, default=32, help="fetch Queue Size.")
parser.add_argument("--decodeWidth", type=int, default=8, help="decode width.")
parser.add_argument("--LQEntries", type=int, default=32, help="Number of load queue entry.")
parser.add_argument("--SQEntries", type=int, default=32, help="Number of store queue entry.")
parser.add_argument("--numROBEntries", type=int, default=192, help="Number of reorder buffer.")



def get_processes(options):
    """Interprets provided options and returns a list of processes"""

    multiprocesses = []
    outputs = []
    errouts = []

    workloads = options.benchmark.split(';')

    if options.benchmark_stdout != "":
        outputs = options.benchmark_stdout.split(';')
    elif options.output != "":
        outputs = options.output.split(';')

    if options.benchmark_stderr != "":
        errouts = options.benchmark_stderr.split(';')
    elif options.errout != "":
        errouts = options.errout.split(';')

    idx = 0
    for wrkld in workloads:

        if wrkld:
            print('Selected SPEC_CPU2017 benchmark')
            if wrkld == 'perlbench_r':
                print('--> perlbench_r')
                process = spec17_benchmarks.perlbench_r
            elif wrkld == 'perlbench_s':
                print('--> perlbench_s')
                process = spec17_benchmarks.perlbench_s
            elif wrkld == 'gcc_r':
                print('--> gcc_r')
                process = spec17_benchmarks.gcc_r
            elif wrkld == 'gcc_s':
                print('--> gcc_s')
                process = spec17_benchmarks.gcc_s
            elif wrkld == 'mcf_r':
                print('--> mcf_r')
                process = spec17_benchmarks.mcf_r
            elif wrkld == 'mcf_s':
                print('--> mcf_s')
                process = spec17_benchmarks.mcf_s
            elif wrkld == 'omnetpp_r':
                print('--> omnetpp_r')
                process = spec17_benchmarks.omnetpp_r
            elif wrkld == 'omnetpp_s':
                print('--> omnetpp_s')
                process = spec17_benchmarks.omnetpp_s
            elif wrkld == 'xalancbmk_r':
                print('--> xalancbmk_r')
                process = spec17_benchmarks.xalancbmk_r
            elif wrkld == 'xalancbmk_s':
                print('--> xalancbmk_s')
                process = spec17_benchmarks.xalancbmk_s
            elif wrkld == 'x264_r':
                print('--> x264_r')
                process = spec17_benchmarks.x264_r
            elif wrkld == 'x264_s':
                print('--> x264_s')
                process = spec17_benchmarks.x264_s
            elif wrkld == 'deepsjeng_r':
                print('--> deepsjeng_r')
                process = spec17_benchmarks.deepsjeng_r
            elif wrkld == 'deepsjeng_s':
                print('--> deepsjeng_s')
                process = spec17_benchmarks.deepsjeng_s
            elif wrkld == 'leela_r':
                print('--> leela_r')
                process = spec17_benchmarks.leela_r
            elif wrkld == 'leela_s':
                print('--> leela_s')
                process = spec17_benchmarks.leela_s
            elif wrkld == 'exchange2_r':
                print('--> exchange2_r')
                process = spec17_benchmarks.exchange2_r
            elif wrkld == 'exchange2_s':
                print('--> exchange2_s')
                process = spec17_benchmarks.exchange2_s
            elif wrkld == 'xz_r':
                print('--> xz_r')
                process = spec17_benchmarks.xz_r
            elif wrkld == 'xz_s':
                print('--> xz_s')
                process = spec17_benchmarks.xz_s
            elif wrkld == 'bwaves_r':
                print('--> bwaves_r')
                process = spec17_benchmarks.bwaves_r
            elif wrkld == 'bwaves_s':
                print('--> bwaves_s')
                process = spec17_benchmarks.bwaves_s
            elif wrkld == 'cactuBSSN_r':
                print('--> cactuBSSN_r')
                process = spec17_benchmarks.cactuBSSN_r
            elif wrkld == 'cactuBSSN_s':
                print('--> cactuBSSN_s')
                process = spec17_benchmarks.cactuBSSN_s
            elif wrkld == 'namd_r':
                print('--> namd_r')
                process = spec17_benchmarks.namd_r
            elif wrkld == 'parest_r':
                print('--> parest_r')
                process = spec17_benchmarks.parest_r
            elif wrkld == 'povray_r':
                print('--> povray_r')
                process = spec17_benchmarks.povray_r
            elif wrkld == 'lbm_r':
                print('--> lbm_r')
                process = spec17_benchmarks.lbm_r
            elif wrkld == 'lbm_s':
                print('--> lbm_s')
                process = spec17_benchmarks.lbm_s
            elif wrkld == 'wrf_r':
                print('--> wrf_r')
                process = spec17_benchmarks.wrf_r
            elif wrkld == 'wrf_s':
                print('--> wrf_s')
                process = spec17_benchmarks.wrf_s
            elif wrkld == 'blender_r':
                print('--> blender_r')
                process = spec17_benchmarks.blender_r
            elif wrkld == 'cam4_r':
                print('--> cam4_r')
                process = spec17_benchmarks.cam4_r
            elif wrkld == 'cam4_s':
                print('--> cam4_s')
                process = spec17_benchmarks.cam4_s
            elif wrkld == 'pop2_s':
                print('--> pop2_s')
                process = spec17_benchmarks.pop2_s
            elif wrkld == 'imagick_r':
                print('--> imagick_r')
                process = spec17_benchmarks.imagick_r
            elif wrkld == 'imagick_s':
                print('--> imagick_s')
                process = spec17_benchmarks.imagick_s
            elif wrkld == 'nab_r':
                print('--> nab_r')
                process = spec17_benchmarks.nab_r
            elif wrkld == 'nab_s':
                print('--> nab_s')
                process = spec17_benchmarks.nab_s
            elif wrkld == 'fotonik3d_r':
                print('--> fotonik3d_r')
                process = spec17_benchmarks.fotonik3d_r
            elif wrkld == 'fotonik3d_s':
                print('--> fotonik3d_s')
                process = spec17_benchmarks.fotonik3d_s
            elif wrkld == 'roms_r':
                print('--> roms_r')
                process = spec17_benchmarks.roms_r
            elif wrkld == 'roms_s':
                print('--> roms_s')
                process = spec17_benchmarks.roms_s
            elif wrkld == 'specrand_fs':
                print('--> specrand_fs')
                process = spec17_benchmarks.specrand_fs
            elif wrkld == 'specrand_fr':
                print('--> specrand_fr')
                process = spec17_benchmarks.specrand_fr
            elif wrkld == 'specrand_is':
                print('--> specrand_is')
                process = spec17_benchmarks.specrand_is
            elif wrkld == 'specrand_ir':
                print('--> specrand_ir')
                process = spec17_benchmarks.specrand_ir
            else:
                print("No recognized SPEC2017 benchmark selected! Exiting.")
                sys.exit(1)
  
                
            process.cwd = os.getcwd()
            process.gid = os.getgid()
            if len(outputs) > idx:
                process.output = outputs[idx]
            if len(errouts) > idx:
                process.errout = errouts[idx]

            multiprocesses.append(process)
            idx += 1

        else:
            print >> sys.stderr, "Need --benchmark switch to specify SPEC CPU2017 workload. Exiting!\n"
            sys.exit(1)

    if options.smt:
        assert(options.cpu_type == "DerivO3CPU")
        return multiprocesses, idx
    else:
        return multiprocesses, 1


#parser = optparse.OptionParser()
#Options.addCommonOptions(parser)
#Options.addSEOptions(parser)

if '--ruby' in sys.argv:
    Ruby.define_options(parser)

options = parser.parse_args()

multiprocesses, numThreads = get_processes(options)
(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(options)
CPUClass.numThreads = numThreads

# Check -- do not allow SMT with multiple CPUs
if options.smt and options.num_cpus > 1:
    fatal("You cannot use SMT with multiple CPUs!")

np = options.num_cpus
mp0_path = multiprocesses[0].executable
system = System(cpu = [CPUClass(cpu_id=i) for i in range(np)],
                mem_mode = test_mem_mode,
                mem_ranges = [AddrRange(options.mem_size)],
                cache_line_size = options.cacheline_size,
                )

if numThreads > 1:
    system.multi_thread = True

# Create a top-level voltage domain
system.voltage_domain = VoltageDomain(voltage = options.sys_voltage)

# Create a source clock for the system and set the clock period
system.clk_domain = SrcClockDomain(clock =  options.sys_clock,
                                   voltage_domain = system.voltage_domain)

# Create a CPU voltage domain
system.cpu_voltage_domain = VoltageDomain()

# Create a separate clock domain for the CPUs
system.cpu_clk_domain = SrcClockDomain(clock = options.cpu_clock,
                                       voltage_domain =
                                       system.cpu_voltage_domain)

# If elastic tracing is enabled, then configure the cpu and attach the elastic
# trace probe
#if options.elastic_trace_en:
#    CpuConfig.config_etrace(CPUClass, system.cpu, options)

# All cpus belong to a common cpu_clk_domain, therefore running at a common
# frequency.
for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

if ObjectList.is_kvm_cpu(CPUClass) or ObjectList.is_kvm_cpu(FutureClass):
    if buildEnv['TARGET_ISA'] == 'x86':
        system.kvm_vm = KvmVM()
        for process in multiprocesses:
            process.useArchPT = True
            process.kvmInSE = True
    else:
        fatal("KvmCPU can only be used in SE mode with x86")

# Sanity check
if options.simpoint_profile:
    if not ObjectList.is_noncaching_cpu(CPUClass):
        fatal("SimPoint/BPProbe should be done with an atomic cpu")
    if np > 1:
        fatal("SimPoint generation not supported with more than one CPUs")

for i in range(np):
    if options.smt:
        system.cpu[i].workload = multiprocesses
    elif len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]

    if options.simpoint_profile:
        system.cpu[i].addSimPointProbe(options.simpoint_interval)

    if options.checker:
        system.cpu[i].addCheckerCpu()

    if options.bp_type:
        bpClass = ObjectList.bp_list.get(options.bp_type)
        system.cpu[i].branchPred = bpClass()

    if options.indirect_bp_type:
        indirectBPClass = \
            ObjectList.indirect_bp_list.get(options.indirect_bp_type)
        system.cpu[i].branchPred.indirectBranchPred = indirectBPClass()

    system.cpu[i].createThreads()

if options.ruby:
    Ruby.create_system(options, False, system)
    assert(options.num_cpus == len(system.ruby._cpu_ports))

    system.ruby.clk_domain = SrcClockDomain(clock = options.ruby_clock,
                                        voltage_domain = system.voltage_domain)
    for i in range(np):
        ruby_port = system.ruby._cpu_ports[i]

        # Create the interrupt controller and connect its ports to Ruby
        # Note that the interrupt controller is always present but only
        # in x86 does it have message ports that need to be connected
        system.cpu[i].createInterruptController()

        # Connect the cpu's cache ports to Ruby
        system.cpu[i].icache_port = ruby_port.slave
        system.cpu[i].dcache_port = ruby_port.slave
        if buildEnv['TARGET_ISA'] == 'x86':
            system.cpu[i].interrupts[0].pio = ruby_port.master
            system.cpu[i].interrupts[0].int_master = ruby_port.slave
            system.cpu[i].interrupts[0].int_slave = ruby_port.master
            system.cpu[i].itb.walker.port = ruby_port.slave
            system.cpu[i].dtb.walker.port = ruby_port.slave
else:
    MemClass = Simulation.setMemClass(options)
    system.membus = SystemXBar()
    system.system_port = system.membus.slave
    CacheConfig.config_cache(options, system)
    MemConfig.config_mem(options, system)
    config_filesystem(system, options)

system.workload = SEWorkload.init_compatible(mp0_path)

if options.wait_gdb:
    for cpu in system.cpu:
        cpu.wait_for_remote_gdb = True

root = Root(full_system = False, system = system)
Simulation.run(options, root, system, FutureClass)





