import os
from openmm import app
from utils import *
from misc_tools import *
import ray
from argparse import ArgumentParser
import time
import yaml
print("Starting simulations........")
os.system(f"sleep {np.random.choice(60, size=1)[0]}")
parser = ArgumentParser()
parser.add_argument('--config', nargs='?', default='config_sim.yaml', const='config_simerge.yaml', type=str)
args = parser.parse_args()
config_sim = yaml.safe_load(open(f'{args.config}', 'r'))
cwd = config_sim["cwd"]
dataset = config_sim["dataset"]
cycle = config_sim["cycle"]
name = config_sim["name"]
record = config_sim["record"]
@ray.remote(num_cpus=config_sim["Threads"])  # specify num_cpus to make sure well-distributed jobs
def simulate(config):
    """ Simulate openMM Calvados

        * config is a dictionary """

    # parse config
    record = config["record"]
    overwrite = config["overwrite"]
    dataset = config["dataset"]
    Usecheckpoint = config["Usecheckpoint"]
    name, temp, ionic = config['name'], config['temp'], config['ionic']
    cutoff, steps, wfreq = config['cutoff'], config['steps'], config['wfreq']
    L = config['L']
    eps_factor = config["eps_factor"]
    pH = config["pH"]
    isIDP = config['isIDP']
    CoarseGrained = config['CoarseGrained']
    gpu_id = config['gpu_id']
    gpu = config['gpu']
    seq = config['seq']
    cycle = config["cycle"]
    replica = config["replica"]
    cwd, path2fasta = config['cwd'], config['path2fasta']
    os.system(f"uname -n>{cwd}/{dataset}/{record}/{cycle}/{replica}_uname.txt")
    use_pdb, path2pdb = config['use_pdb'], config['path2pdb']
    slab = config['slab']
    Threads = config["Threads"]
    fdomains = config['fdomains']
    initial_type = config['initial_type']
    print("use_pdb: ", use_pdb)
    print("Threads: ", Threads)
    center_of_mass = np.array([0,0,0])
    if use_pdb:
        use_hnetwork = config['use_hnetwork']
        if use_hnetwork:
            k_restraint = config['k_restraint']
            use_ssdomains = config['use_ssdomains']
            if use_ssdomains:
                ssdomains = get_ssdomains(name, fdomains)
                pae = None
            else:
                input_pae = config['input_pae']
                pae = load_pae(input_pae)
                ssdomains = None
    else:
        path2pdb = ''
        use_hnetwork = False
        pae = None
        ssdomains = None
    # load residue parameters
    residues = load_parameters(cwd, dataset, cycle, initial_type)
    # build protein dataframe
    df = pd.DataFrame(columns=['pH', 'ionic', 'temp', 'eps_factor','fasta'], dtype=object)
    df.loc[record] = dict(pH=pH, ionic=ionic, temp=temp, eps_factor=eps_factor, fasta=seq)
    prot = df.loc[record]
    print("pH:", prot.pH)
    print("ionic:", prot.ionic)
    print("temp:", prot.temp)
    # LJ and YU parameters
    lj_eps, fasta, types, MWs = genParamsLJ(residues, record, prot)
    print("lj_eps:", lj_eps * unit.kilojoules_per_mole)
    yukawa_eps, yukawa_kappa = genParamsDH(residues, record, prot, temp)

    N = len(fasta)  # number of residues

    if slab:  # overrides L from config for now
        L, Lz, margin, Nsteps = slab_dimensions(N)
        xy, n_chains = slab_xy(L, margin)
    else:
        n_chains = 1
        Lz = L

    # get input geometry
    if slab:
        if use_pdb:
            raise
        else:
            pos = []
            for x, y in xy:
                pos.append([[x, y, Lz / 2 + (i - N / 2.) * .38] for i in range(N)])
            pos = np.array(pos).reshape(n_chains * N, 3)
    else:
        if use_pdb:
            print(f'Starting from pdb structure {path2pdb}')
            pos, center_of_mass = geometry_from_pdb(cwd, dataset, name, record, cycle, replica, path2pdb, CoarseGrained=CoarseGrained, ssdomains=ssdomains)
        else:
            spiral = True
            if spiral:
                pos = xy_spiral_array(N)
            else:
                pos = [[L / 2, L / 2, L / 2 + (i - N / 2.) * .38] for i in range(N)]
            pos = np.array(pos)

    top = build_topology(fasta, n_chains=n_chains)
    md.Trajectory(pos + center_of_mass, top, 0, [L, L, Lz], [90, 90, 90]).save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/ini_beads.pdb', force_overwrite=True)
    pos = pos + np.array([L / 2, L / 2, L / 2])
    a = md.Trajectory(pos, top, 0, [L, L, Lz], [90, 90, 90])
    a.save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/top_{replica}.pdb', force_overwrite=True)
    # build openmm system
    system = openmm.System()

    # box
    a, b, c = build_box(L, L, Lz)
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # load topology into system
    pdb = app.pdbfile.PDBFile(f'{cwd}/{dataset}/{record}/{cycle}/top_{replica}.pdb')

    # print(pdb.topology)

    # particles and termini
    system = add_particles(system, residues, prot, n_chains=n_chains)
    dmap = euclidean(pos, pos)
    # interactions
    print('cutoff:', cutoff * unit.nanometer)
    hb, yu, ah = set_interactions(system, residues, prot, lj_eps, cutoff, yukawa_kappa, yukawa_eps, N,
                                  n_chains=n_chains, CoarseGrained=CoarseGrained, dismatrix=dmap, isIDP=isIDP,
                                  fdomains=fdomains)
    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    print("use_hnetwork: ", use_hnetwork)
    # harmonic network (currently unavailable for slab sim)
    if use_hnetwork:
        if slab:
            raise
        else:
            cs, yu, ah = set_harmonic_network(N, dmap, pae, yu, ah, ssdomains=ssdomains, k_restraint=k_restraint)
            print(f"k_restraint used: {k_restraint}")
            system.addForce(cs)

    # use langevin integrator
    integrator = openmm.openmm.LangevinIntegrator(temp * unit.kelvin, 0.01 / unit.picosecond, 0.01 * unit.picosecond)
    print(integrator.getFriction(), integrator.getTemperature())
    open(f'{cwd}/{dataset}/{record}/{cycle}/system_pro.xml', 'w').write(openmm.XmlSerializer.serialize(system))
    if replica==0:
        open(f'{cwd}/{dataset}/{record}/{cycle}/system_pro.xml', 'w').write(openmm.XmlSerializer.serialize(system))
    # assemble simulation
    if gpu:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # feasible
        os.system("echo $CUDA_VISIBLE_DEVICES")
        # os.system("export CUDA_VISIBLE_DEVICES=1")
        # platform = openmm.Platform.getPlatformByName("CUDA")
        simulation = app.simulation.Simulation(pdb.topology, system, integrator,
                                               openmm.Platform.getPlatformByName("CUDA"),
                                               {"DeviceIndex": f"{gpu_id}"})
    else:
        platform = openmm.Platform.getPlatformByName('CPU')
        simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(Threads=str(Threads)))

    if Usecheckpoint and os.path.isfile(f"{cwd}/{dataset}/{record}/{cycle}/checkpoint{replica}.chk"):
        # make sure steps need to be substracted
        print(f'Reading check point file checkpoint{replica}.chk')
        # load pos and vel of last frame, and chk file can be used within different nodes at least on Computerome
        simulation.loadCheckpoint(f"{cwd}/{dataset}/{record}/{cycle}/checkpoint{replica}.chk")
        # print(simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True))
        simulation.reporters.append(
            app.dcdreporter.DCDReporter(f'{cwd}/{dataset}/{record}/{cycle}/{replica}_con.dcd', wfreq,
                                        enforcePeriodicBox=False, append=False))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        # state = simulation.context.getState(getPositions=True)
        # pos2 = state.getPositions(asNumpy=True)
        # enforcePeriodicBox=False makes sure that the saved molecules are whole;
        # when openmm tries to keep the center of mass of molecule in the box, it will modify the x,y,z of every atoms separately;
        # which means only if x(or y or z) of COM is out of box, it will modify x of every atom so that x of COM is in the box;
        # Therefore, if just the x of a single atom is out of box, x might not be modified into the box;
        # and the center of simulation box is (L/2, L/2, L/2)
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{cwd}/{dataset}/{record}/{cycle}/{replica}.dcd', wfreq,
                                                                enforcePeriodicBox=False, append=False))

    simulation.reporters.append(
        app.statedatareporter.StateDataReporter(f'{cwd}/{dataset}/{record}/{cycle}/statedata_{replica}.log', int(wfreq),
                                                step=True, speed=True, elapsedTime=True, separator='\t', progress=True,
                                                remainingTime=True, totalSteps=steps))
    simulation.reporters.append(
        app.checkpointreporter.CheckpointReporter(file=f"{cwd}/{dataset}/{record}/{cycle}/checkpoint{replica}.chk",
                                                  reportInterval=wfreq))


    starttime = time.time()  # begin timer
    simulation.step(steps)
    endtime = time.time()  # end timer
    target_seconds = endtime - starttime  # total used time
    print(
        f"{name} total simulations used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")

incomplete = True
while incomplete:
    try:
        ray.init(num_cpus=int(config_sim["Threads"]*len(config_sim["replicas_list4MD"])), include_dashboard=False)
    except Exception:
        os.system("sleep 1")
    else:
        incomplete = False

for replica in config_sim["replicas_list4MD"]:
    config_sim["replica"] = replica
    yaml.dump(config_sim, open(f"{cwd}/{dataset}/{record}/{cycle}/config_{replica}.yaml", 'w'))
# pool of tasks, if the allocated nodes do not allow them to be ran at the same time, they will be sequentially finished depending on requested resources
print(config_sim["replicas_list4MD"])
incomplete = True
while incomplete:
    try:
        ray.get([simulate.remote(yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_{replica}.yaml", 'r'))) for replica in config_sim["replicas_list4MD"]])
    except Exception:
        os.system("sleep 1")
    else:
        incomplete = False