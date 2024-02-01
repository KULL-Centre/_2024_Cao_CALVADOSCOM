import os
import yaml
from utils import visualize_traj, energy_details
from rawdata import *
from argparse import ArgumentParser
from utils import load_parameters
def centerDCD(config):
    record = config["record"]
    cwd = config["cwd"]
    nframes = config["nframes"]
    dataset = config["dataset"]
    cycle = config["cycle"]
    replicas = config["replicas"]
    discard_first_nframes = config["discard_first_nframes"]
    validate = config["validate"]
    initial_type = config["initial_type"]
    print("validate:", validate)
    residues = load_parameters(cwd, dataset, cycle, initial_type)
    if not validate:
        prot = pd.read_pickle(f'{cwd}/{dataset}/allproteins.pkl').loc[record]
    else:
        prot = pd.read_pickle(f'{cwd}/{dataset}/allproteins_validate.pkl').loc[record]

    incomplete = True
    while incomplete:
        try:
            top = md.Topology()
            chain = top.add_chain()
            for resname in prot.fasta:
                residue = top.add_residue(residues.loc[resname, 'three'], chain)
                top.add_atom(residues.loc[resname, 'three'], element=md.element.carbon, residue=residue)
            for i in range(len(prot.fasta) - 1):
                top.add_bond(top.atom(i), top.atom(i + 1))
            traj = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/0.dcd", top=top)[discard_first_nframes:]
            for i in range(1, replicas):
                t = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{i}.dcd", top=top)[discard_first_nframes:]
                traj = md.join([traj, t])
            traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
            assert len(traj) == int(nframes * replicas)
            traj.center_coordinates()
            traj.xyz += traj.unitcell_lengths[0, 0] / 2
            print(f'Number of frames: {traj.n_frames}')
            traj.save_dcd(f'{cwd}/{dataset}/{record}/{cycle}/{record}.dcd')
            traj[0].save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/{record}.pdb')
        except Exception:
            os.system("rm /home/people/fancao/IDPs_multi/core*")
            os.system("sleep 1")
        else:
            incomplete = False

    visualize_traj(cwd, dataset, record, cycle)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', nargs='?', default='config.yaml', const='config.yaml', type=str)
    args = parser.parse_args()
    with open(f'{args.config}', 'r') as stream:
        config = yaml.safe_load(stream)
    centerDCD(config)
