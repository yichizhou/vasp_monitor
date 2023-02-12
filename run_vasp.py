import os
import sys
import time
import logging
import os
import argparse
import copy
import ase
import pickle
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from ase import Atoms
from ase.io.trajectory import Trajectory
from scipy import stats
from ase.calculators.vasp import Vasp
from tqdm import tqdm



def vasp_handler(file, args):
    ASE_VASP_COMMAND="mpirun -np "+str(args.n_core)+" vasp_std"
    calc = Vasp(xc='PBE',
            encut=args.encut,
            ediff=args.ediff,
            ismear=args.ismear,
            sigma=args.sigma,
            kspacing=args.kspacing,
            gamma=args.gamma,
            nelm=args.nelm,
            restart=None,            
            command=ASE_VASP_COMMAND,
            directory=args.vasp_dir,
            txt="-" if args.print_log else None)

    read_path = os.path.join(args.read_dir, file)
    atoms_list=ase.io.read(read_path, index=":", format="extxyz")
    print(f"calculating DFT of {len(atoms_list)} atoms from {read_path}")
    processed_atoms_list = []
    for atoms in tqdm(atoms_list):
        atoms.calc = calc
        try:
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            s = atoms.get_stress()
            processed_atoms_list.append(copy.deepcopy(atoms))
            bashCommand = "rm -rf "+args.vasp_dir
            os.system(bashCommand)
        except:
            pass
    bashCommand = "rm -rf "+args.vasp_dir
    os.system(bashCommand)
    ase.io.write(os.path.join(args.write_dir, file), processed_atoms_list, format="extxyz")
    
if __name__ == "__main__":
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
    parser.add_argument("--print_log",action="store_true")
    parser.add_argument("--read_dir",type=str,default="./data/unlabeled")
    parser.add_argument("--write_dir",type=str,default="./data/labeled")
    parser.add_argument("--vasp_dir",type=str,default="./vasp_run")
    parser.add_argument("--file",type=str,default=None)
    args=parser.parse_args()
    vasp_handler(args.file, args)
    