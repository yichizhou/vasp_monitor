import sys
import time
import logging
import torch
import os
import argparse
import wandb
import numpy as np
import copy
import ase
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from rl4kinetics.active_learning.utils import simple_dir_watcher
from ase import Atoms
from ase.io.trajectory import Trajectory
from scipy import stats
from ase.calculators.vasp import Vasp
from tqdm import tqdm
from rl4kinetics.active_learning.utils import Event, filter_handler
parser=argparse.ArgumentParser()
parser.add_argument("--path",type=str,default=None)
parser.add_argument("--n_core",type=int,default=6)
parser.add_argument("--interleave",nargs="+",type=int,default=[0,1500])  ## interleave [low,high)
parser.add_argument("--encut",type=float,default=400)
parser.add_argument("--ediff",type=float,default=1e-4)
parser.add_argument("--ismear",type=int,default=0)
parser.add_argument("--sigma",type=float,default=0.02)
parser.add_argument("--amplitude",type=float,default=0.2)
parser.add_argument("--kspacing",type=float,default=0.2)
parser.add_argument("--gamma",type=bool,default=True)
parser.add_argument("--nelm",type=int,default=200)
parser.add_argument("--wandb",action="store_true")
parser.add_argument("--wandb_key",type=str,default="829d0c4979c647952de7550ef7d8764f91221d36")
parser.add_argument("--print_log",action="store_true")
parser.add_argument("--vc_code_dir", type=str, default="/mnt/data/vasp_watcher/vasp_monitor")
parser.add_argument("--local_dir_prefix", type=str, default="/home/yiczho/mycontainer/vasp_watcher")
parser.add_argument("--vc_dir_prefix", type=str, default="/mnt/data/vasp_watcher")
parser.add_argument("--read_dir_suffix",type=str,default="data/unlabeled")
parser.add_argument("--write_dir_suffix",type=str,default="data/labeled")
parser.add_argument("--vasp_dir",type=str,default="./vasp_run")
args=parser.parse_args()

import yaml
base_yaml = yaml.safe_load(open("amlt_vasp_watch.yaml", 'r'))
# print(base_yaml)
# print(base_yaml["jobs"])

def generate_vasp_yaml(file, base_yaml, args):
    yaml_dict = copy.deepcopy(base_yaml)
    yaml_name = f"run_vasp_{file}.yaml"

    vasp_dir_suffix = f"vasp_run_{file}"
    vasp_dir = os.path.join(args.vc_dir_prefix, vasp_dir_suffix)
    vc_read_dir = os.path.join(args.vc_dir_prefix, args.read_dir_suffix)
    vc_write_dir = os.path.join(args.vc_dir_prefix, args.write_dir_suffix)
    vasp_pp_dir = os.path.join(args.vc_code_dir, "VASP_PP")
    find_code_command = f"cd {args.vc_code_dir}"
    run_command = f"python run_vasp.py --file {file} --vasp_dir {vasp_dir} --read_dir {vc_read_dir} --write_dir {vc_write_dir}"
    delete_vasp_run = f"rm -rf {vasp_dir}"
    
    yaml_dict['jobs'][0]['command'].append(f"export VASP_PP_PATH={vasp_pp_dir}")
    yaml_dict['jobs'][0]['command'].append(find_code_command)
    yaml_dict["jobs"][0]['command'].append(run_command)
    yaml_dict["jobs"][0]['command'].append(delete_vasp_run)
    #yaml_dict["jobs"][0]['command'].append("python haha.py")

    yaml.dump(yaml_dict,open(yaml_name, "w"))

    submit_job = f"amlt run -y {yaml_name} -d run_vasp_{file}"
    os.system(submit_job)
    os.system(f"rm {yaml_name}")

def handler(file, args):
    read_path = os.path.join(args.local_dir_prefix, args.read_dir_suffix, file)
    generate_vasp_yaml(file, base_yaml, args)

class Event(LoggingEventHandler):
    def __init__(self, args, func):
        super().__init__()
        self.args = args
        self.func = func
        self.processed_files = {}
    def on_modified(self,event):
        args = self.args
        print(self.processed_files)
        read_dir = os.path.join(args.local_dir_prefix, args.read_dir_suffix)
        unprocessed_file = simple_dir_watcher(self.processed_files, read_dir, ext="xyz")
        print(unprocessed_file)
        for file in unprocessed_file:
            self.func(file, args)
            self.processed_files[file]=True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = os.path.join(args.local_dir_prefix, args.read_dir_suffix)
    print("monitoring dir:", path)
    event_handler = Event(args, handler)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()