from jinja2 import Template
import yaml
import subprocess

submission = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N contact_map
#PBS -l nodes=1:ppn=40:thinnode
#PBS -l walltime=45:00:00
#PBS -l mem=170gb
#PBS -o {{cwd}}/{{dataset}}/{{record}}/cmap_out
#PBS -e {{cwd}}/{{dataset}}/{{record}}/cmap_err

source ~/.bashrc

conda activate {{env_name}}

python3 {{cwd}}/contact_map.py  --path2config {{path2config}}""")


home_folder = "yourhomedir/_2024_Cao_CALVADOSCOM/src"  # absolute path to "submit_contact_map.py"
dataset = "slabC2_SCCOM_1"
record = "hnRNPA1S@0.15@293"
env_name = "CALVADOSCOM"
cycle = 0  # only used to import simulation settings
temp = 293

config_data = dict(cwd=home_folder, dataset=dataset, record=record, cycle=cycle, temp=temp)
path2config = f"{home_folder}/{dataset}/{record}/{cycle}/contact_map.yaml"
yaml.dump(config_data, open(path2config,'w'))
cmap_dict = dict(cwd=home_folder, dataset=dataset, record=record, cycle=cycle, path2config=path2config, env_name=env_name)
open(f"{home_folder}/{dataset}/{record}/{cycle}/contact_map.pbs", 'w').write(submission.render(cmap_dict))

proc = subprocess.run(['qsub', f"{home_folder}/{dataset}/{record}/{cycle}/contact_map.pbs"],capture_output=True)
print(proc)
