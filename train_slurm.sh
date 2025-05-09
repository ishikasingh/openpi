# account="nvr_srl_simpler"

# srun -p interactive -A $account -t 4:00:00 --gres=gpu:8 --nodes=1 --mem=0 --exclusive  --overcommit --ntasks-per-node=1 --job-name gpu_interactive_short   --pty /bin/bash 

# ~/ssh_server_setup.sh

# --no-container-remap-root --container-image="/home/hanrongy/user_path/docker_images/nextImage_v0.sqsh" --container-mounts="/lustre:/lustre,$HOME:$HOME,/home:/home" --container-save=$ISFILES/nextImage_v0.sqsh


for ((i=1; i<=100; i++))
do
    echo "Iteration $i"
# done


export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
srun --nodes=1 -A nvr_srl_simpler --gres gpu:1  --time=4:0:0  --partition=interactive,grizzly,polar,polar3,polar4 --mincpus=224 --container-env PATH,HOME \
    --export=XLA_PYTHON_CLIENT_MEM_FRACTION uv run scripts/train.py pi0_fast_libero --exp-name=my_experiment --resume
done     
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  uv run scripts/train.py pi0_fast_libero --exp-name=my_experiment --resume

##cmd: bash run_slurm_job.sh 6