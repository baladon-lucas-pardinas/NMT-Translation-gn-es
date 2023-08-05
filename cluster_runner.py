# coding=utf-8
import re
import os
import argparse

# # https://www.cluster.uy/ayuda/como_ejecutar/#trabajo-besteffort
# # NOMBRE	    # NÚCLEOS DISPONIBLES	MAX. TIEMPO	   MAX. TRABAJOS
# # normal	    560	                    5 días	       40
# # besteffort	1120	                5 días	       120
NORMAL = 'normal'; BESTEFFORT = 'besteffort'; MAX_CPUS = 'max_cpus'; MAX_TIME = 'max_time'; MAX_JOBS = 'max_jobs'

# QOS_INFO = {
#     NORMAL: {MAX_CPUS: 560, MAX_TIME: '120:00:00', MAX_JOBS: 40},
#     BESTEFFORT: {MAX_CPUS: 1120, MAX_TIME: '120:00:00', MAX_JOBS: 120},
# }

# # https://www.cluster.uy/ayuda/recursos_disponibles/
# # NOMBRE	        PROCESADORES	    # NÚCLEOS  MEM.     GPU/NODE	      DISCO
# # node[01-14][17]	Xeon Gold 6138	    40	       128 GB	NVIDIA P100	      300 GB SSD
# # node[15][16]	    Xeon Gold 6138	    40	       128 GB	NVIDIA A100	      300 GB SSD
# # node[26-28]	    Xeon Gold 6138	    40	       128 GB	   -	          300 GB SSD
# # node[18-22]	    Xeon Gold 6138	    40	       128 GB	NVIDIA P100 x 2	  300 GB SSD
# # node23	        Xeon Gold 6138	    40	       128 GB	NVIDIA P100 x 3	  300 GB SSD
# # node[24-25]	    Xeon Gold 6138	    40	       512 GB	   -	          300 GB SSD
# # node31	        AMD EPYC 7642	    96	       256 GB	   -	          150 GB SSD
CORES = 'cores'; MEMORY_GB = 'memory_gb'; GPUS_N = 'gpus'; GPU_TYPE = 'gpu_type'; P100 = 'p100'; A100 = 'a100'

# NODE_INFO = {
#     **{f'node{i}': {CORES: 40, MEMORY_GB: 128, GPUS_N: 1, GPU_TYPE: P100} for i in range(1, 15)},
#     **{f'node{i}': {CORES: 40, MEMORY_GB: 128, GPUS_N: 1, GPU_TYPE: A100} for i in range(15, 17)},
#     **{f'node{i}': {CORES: 40, MEMORY_GB: 128, GPUS_N: 0, GPU_TYPE: None} for i in range(26, 29)},
#     **{f'node{i}': {CORES: 40, MEMORY_GB: 128, GPUS_N: 2, GPU_TYPE: P100} for i in range(18, 23)},
#     **{f'node{i}': {CORES: 40, MEMORY_GB: 128, GPUS_N: 3, GPU_TYPE: P100} for i in range(23, 24)},
#     **{f'node{i}': {CORES: 40, MEMORY_GB: 512, GPUS_N: 0, GPU_TYPE: None} for i in range(24, 26)},
#     **{f'node{i}': {CORES: 96, MEMORY_GB: 256, GPUS_N: 0, GPU_TYPE: None} for i in range(31, 32)},
# }

SLURM_TEMPLATE = """\
#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time=5-0
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --output={output_filename}
#SBATCH --gres=gpu:p100:{gpus_n}

pwd
ls

SCRIPT_NAME=$1
HOME=/docker/home
CLUSTER_BIND_DIR=/clusteruy/home/${USER}/marian/marian_container
SCRIPT_PATH=${HOME}/marianmt/${SCRIPT_NAME}
export SINGULARITY_TMPDIR=${HOME}/cache
export TMPDIR=$SINGULARITY_TMPDIR
chmod +x ${CLUSTER_BIND_DIR}/marianmt/${SCRIPT_NAME}
export PYTHONPATH=${HOME}/marianmt/libs
singularity exec -H ${HOME} --nv --no-home --contain --bind ${CLUSTER_BIND_DIR}:$HOME ${CLUSTER_BIND_DIR}/marian-nmt_1.11.0_sentencepiece_cuda-11.3.0.sif $SCRIPT_PATH
"""

TIME_LIMIT_MESSAGE = "TIME LIMIT"
NORMAL_QOS = 'gpu'

def get_out_file_name(from_flag, to_flag, besteffort=False):
    filename = "run"
    if besteffort:
        filename += "_besteffort"
    filename += '{from_flag}-{to_flag}'.format(from_flag=from_flag, to_flag=to_flag)
    filename += ".out"
    return filename

def get_slurm_file_name(from_flag, to_flag, besteffort=False):
    filename = "run"
    if besteffort:
        filename += "_besteffort"
    filename += '{from_flag}-{to_flag}'.format(from_flag=from_flag, to_flag=to_flag)
    filename += ".slurm"
    return filename

def get_bash_file_name(from_flag, to_flag, besteffort=False):
    filename = "run"
    if besteffort:
        filename += "_besteffort"
    filename += '{from_flag}-{to_flag}'.format(from_flag=from_flag, to_flag=to_flag)
    filename += ".sh"
    return filename

def get_grid_partitions(total_jobs_n, jobs_n, from_flag, to_flag):
    flag_range = (to_flag - from_flag)
    flags_per_job = flag_range // jobs_n
    partitions = []
    for i in range(jobs_n):
        start = from_flag + i * flags_per_job
        end = start + flags_per_job
        partitions.append((start, end))
    partitions[-1] = (partitions[-1][0], to_flag)
    return partitions

def create_slurm_file_content(output_filename, job_name, partition, qos, gpus_n, ntasks=4, cpus_per_task=9, mem='60G', file_template=SLURM_TEMPLATE):
    params_to_replace = {'job_name': job_name, 'partition': partition, 'qos': qos, 'gpus_n': gpus_n, 'ntasks': ntasks, 'cpus_per_task': cpus_per_task, 'mem': mem, 'output_filename': output_filename}
    for param, value in params_to_replace.items():
        file_template = file_template.replace('{' + param + '}', str(value))
    return file_template

def create_bash_file_content(bash_template_dir, devices, from_flag, to_flag):
    bash_lines = []
    params_to_replace = [('^GPUS="([0-9] )*[0-9]"', 'GPUS="'+devices+'"'), ('^FROM=([0-9]+(\.[0-9]+)?)', 'FROM='+str(from_flag)), ('^TO=([0-9]+(\.[0-9]+)?)', 'TO='+str(to_flag))]
    params_to_replace = [(re.compile(regex), value) for regex, value in params_to_replace]

    with open(bash_template_dir, 'r') as f:
        bash_lines.extend(f.readlines())
    
    for idx in range(len(bash_lines)):
        for param_regex, value in params_to_replace:
            bash_lines[idx] = param_regex.sub(value, bash_lines[idx])

    bash_lines = ''.join(bash_lines)
    return bash_lines

def get_job_name(from_flag, to_flag):
    return 'M-{from_flag}-{to_flag}'.format(from_flag=from_flag, to_flag=to_flag)

def get_gpu_devices(gpus_n=1):
    return ' '.join(map(str, range(gpus_n)))

def persist_file(filedir, content):
    with open(filedir, 'w') as f:
        f.write(content)

def get_read_permissions_command(filedir):
    return 'chmod +x ' + filedir

def run_script(bash_input_template_dir, outputs_scripts_folder, flags_partition, job_name, partition, qos, gpus_n, ntasks=4, cpus_per_task=9, mem='60G', debug=False):
    gpu_devices = get_gpu_devices(gpus_n)
    slurm_filename = get_slurm_file_name(flags_partition[0], flags_partition[1], besteffort=qos==BESTEFFORT)
    bash_filename = get_bash_file_name(flags_partition[0], flags_partition[1], besteffort=qos==BESTEFFORT)
    output_filename = get_out_file_name(flags_partition[0], flags_partition[1], besteffort=qos==BESTEFFORT)
    slurm_output_filename = os.path.join(outputs_scripts_folder, slurm_filename)
    bash_output_filename = os.path.join(outputs_scripts_folder, bash_filename)
    output_filename = os.path.join(outputs_scripts_folder, output_filename)
    bash_script = create_bash_file_content(bash_input_template_dir, gpu_devices, flags_partition[0], flags_partition[1])
    slurm_script = create_slurm_file_content(output_filename, job_name, partition, qos, gpus_n, ntasks, cpus_per_task, mem)
    persist_file(slurm_output_filename, slurm_script) 
    persist_file(bash_output_filename, bash_script)
    script = ['sbatch', slurm_output_filename, bash_output_filename]
    script = ' '.join(script)

    if debug:
        print(script)
        return script

    read_permissions_command = get_read_permissions_command(bash_output_filename)
    os.system(read_permissions_command)
    os.system(script)
    return script

GPU_LOG_REGEX = re.compile('^.+Using ([1-9]) GPUs$')

def awake_jobs(grid_partitions, outputs_scripts_folder, bash_template_file, time_limit_message=TIME_LIMIT_MESSAGE, gpu_regex=GPU_LOG_REGEX, debug=False):
    slept_jobs = []
    for grid_partition in grid_partitions:
        output_file_normal = get_out_file_name(grid_partition[0], grid_partition[1], besteffort=False)
        output_file_besteffort = get_out_file_name(grid_partition[0], grid_partition[1], besteffort=True)
        output_file_normal = os.path.join(outputs_scripts_folder, output_file_normal)
        output_file_besteffort = os.path.join(outputs_scripts_folder, output_file_besteffort)

        for output_file, besteffort in zip([output_file_normal, output_file_besteffort], [False, True]):
            if not os.path.isfile(output_file):
                continue

            with open(output_file, 'r') as f:
                output_lines = f.readlines()
                last_output_line = output_lines[-1]
                if time_limit_message in last_output_line:
                    gpus_n = 1
                    print(last_output_line)
                    gpu_line = [line for line in output_lines if gpu_regex.match(line)]
                    if len(gpu_line) > 0:
                        print(gpu_line[0])
                        gpus_n = int(gpu_regex.match(gpu_line[0]).group(1))
                    slept_jobs.append((grid_partition, gpus_n, besteffort))

    if len(slept_jobs) == 0:
        print("No jobs to awake 🎉")
        return

    for grid_partition, gpus_n, besteffort in slept_jobs:
        job_name = get_job_name(*grid_partition)
        partition = BESTEFFORT if besteffort else NORMAL
        qos = BESTEFFORT if besteffort else NORMAL_QOS
        run_script(bash_template_file, outputs_scripts_folder, grid_partition, job_name, partition, qos, gpus_n, debug=debug)

    return slept_jobs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='run', required=False, choices=['run', 'awake'])
    parser.add_argument('--total_jobs_n', type=int, default=0, required=True)
    parser.add_argument('--jobs_n', type=int, default=0, required=True)
    parser.add_argument('--besteffort_rate', type=float, default=0.0, required=False)
    parser.add_argument('--from_flag', type=int, default=None, required=False)
    parser.add_argument('--to_flag', type=int, default=None, required=False)
    parser.add_argument('--normal_gpus', type=int, default=1, required=False)
    parser.add_argument('--besteffort_gpus', type=int, default=1, required=False)
    parser.add_argument('--outputs_scripts_folder', type=str, default='.', required=False)
    parser.add_argument('--bash_template_file', type=str, default='.\\scripts\\cluster\\train_gn_es_level2_s2s_grid.sh', required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return vars(args)

def check_preconditions(mode, total_jobs_n, jobs_n, besteffort_n, from_flag, to_flag=None):
    if mode not in ['run', 'awake']:
        raise ValueError('Invalid mode:', mode)
    if jobs_n == 0:
        raise Exception("The total number of jobs must be greater than zero")
    if jobs_n < besteffort_n:
        raise Exception("The total number of jobs must not be of the same size as the number of besteffort jobs")
    if total_jobs_n == 0:
        raise Exception("The total number of jobs must be greater than zero")
    if total_jobs_n < jobs_n:
        raise Exception("The total number of jobs must be greater than or equal to the number of jobs")
    if to_flag is not None and from_flag >= to_flag:
        raise Exception("The from flag must be less than the to flag")
    if to_flag is not None and to_flag - from_flag < jobs_n:
        raise Exception("The difference between the from flag and the to flag must be greater than or equal to the number of jobs")
    if to_flag is None and from_flag + jobs_n > total_jobs_n:
        raise Exception("The from flag plus the number of jobs must be less than or equal to the total number of jobs")


# Test examples local
# python cluster_runner.py --debug --from_flag 0 --to_flag 20 --total_jobs_n 20 --jobs_n 10 --besteffort_rate 0.8 --normal_gpus 1 --besteffort_gpus 1 --bash_template_file .\\scripts\\cluster\\train_gn_es_level2_s2s_grid.sh --outputs_scripts_folder ./tests/data/scripts
# python cluster_runner.py --mode awake --debug --jobs_n 12 --bash_template_file .\\scripts\\cluster\\train_gn_es_level2_s2s_grid.sh --outputs_scripts_folder ./tests/data/scripts

# Test examples cluster
# python3 cluster_runner.py --from_flag 0 --to_flag 2 --total_jobs_n 2 --jobs_n 2 --besteffort_rate 0.9 --normal_gpus 1 --besteffort_gpus 1 --bash_template_file /clusteruy/home/${USER}/marian/marian_container/scripts/train_gn_es_level3_s2s_grid.sh --outputs_scripts_folder /clusteruy/home/${USER}/marian/marian_container/scripts/lvl3/s2s
# python3 marianmt/cluster_runner.py --debug --from_flag 0 --to_flag 20 --total_jobs_n 20 --jobs_n 20 --besteffort_rate 0.9 --normal_gpus 1 --besteffort_gpus 1 --bash_template_file scripts/train_gn_es_level3_s2s_grid.sh --outputs_scripts_folder scripts/lvl3/s2s

if __name__ == '__main__':
    args = get_args()
    mode = args['mode']
    total_jobs_n = args['total_jobs_n']
    jobs_n = args['jobs_n']
    besteffort_n = args['besteffort_rate'] * jobs_
    from_flag = args['from_flag']
    to_flag = args['to_flag']
    normal_gpus = args['normal_gpus']
    besteffort_gpus = args['besteffort_gpus']
    outputs_scripts_folder = args['outputs_scripts_folder']
    bash_template_file = args['bash_template_file']
    debug = args['debug']
    partitions = get_grid_partitions(total_jobs_n, jobs_n, from_flag, to_flag)

    print('Generated partitions: ', partitions)
    check_preconditions(mode, total_jobs_n, jobs_n, besteffort_n, from_flag, to_flag)
    
    if mode == 'run':
        normal_n = jobs_n - besteffort_n
        normal_partitions = partitions[:normal_n]
        besteffort_partitions = partitions[normal_n:]

        for grid_partition in normal_partitions:
            job_name = get_job_name(*grid_partition)
            partition = NORMAL
            qos = NORMAL_QOS
            gpus_n = normal_gpus
            run_script(bash_template_file, outputs_scripts_folder, grid_partition, job_name, partition, qos, gpus_n, debug=debug)

        for grid_partition in besteffort_partitions:
            job_name = get_job_name(*grid_partition)
            partition = BESTEFFORT
            qos = BESTEFFORT
            gpus_n = besteffort_gpus
            run_script(bash_template_file, outputs_scripts_folder, grid_partition, job_name, partition, qos, gpus_n, debug=debug)
    elif mode == 'awake':
        awake_jobs(partitions, outputs_scripts_folder, bash_template_file, debug=debug)