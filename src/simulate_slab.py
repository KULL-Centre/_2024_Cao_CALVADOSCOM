from openmm import app
from utils_slab import *
from argparse import ArgumentParser
import time
from misc_tools import *
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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
    pH = config["pH"]
    eps_factor = config["eps_factor"]
    Lz = config['Lz']
    k_eq = config['k_eq']
    isIDP = config['isIDP']
    compact_ini = config['compact_ini']
    CoarseGrained = config['CoarseGrained']
    gpu_id = config['gpu_id']
    gpu = config['gpu']
    seq = config['seq']
    cycle = config["cycle"]
    replica = config["replica"]
    cwd, path2fasta = config['cwd'], config['path2fasta']
    os.system(f"uname -n>{cwd}/{dataset}/{record}/{cycle}/{replica}_uname.txt")
    use_pdb, path2pdb = config['use_pdb'], config['path2pdb']
    slab, runtime = config['slab'], config['runtime']
    Threads = config["Threads"]
    fdomains = config['fdomains']
    initial_type = config['initial_type']
    chains = config['chains']
    equilibrium = config['equilibrium']
    """if cycle != 0:
        os.system(f"cp {cwd}/{dataset}/{record}/{cycle - 1}/checkpoint{cycle - 1}.chk {cwd}/{dataset}/{record}/{cycle}/checkpoint{cycle - 1}.chk")"""
    """if not equilibrium:
        pdb = app.pdbfile.PDBFile(f'{cwd}/{dataset}/{record}/{cycle}/top_{replica}.pdb')
        system = openmm.XmlSerializer.deserialize(open(f'{cwd}/{dataset}/{record}/{cycle}/system_equi.xml', 'r').read())
        for force_idx, force in enumerate(system.getForces()):
            if force.getName()=="rcent":
                system.removeForce(force_idx)"""
    print("use_pdb: ", use_pdb)
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
    df = pd.DataFrame(columns=['pH', 'ionic', 'temp', 'eps_factor', 'fasta'], dtype=object)
    df.loc[record] = dict(pH=pH, ionic=ionic, temp=temp, eps_factor=eps_factor, fasta=seq)
    prot = df.loc[record]
    # LJ and YU parameters
    lj_eps, fasta, types, MWs = genParamsLJ(residues, name, prot)
    yukawa_eps, yukawa_kappa = genParamsDH(residues, name, prot, temp)

    N = len(fasta)  # number of residues

    if slab:  # overrides L from config for now
        n_chains = chains
    else:
        n_chains = 1
        Lz = L

    # get input geometry
    if slab:
        if use_pdb:
            print(f'Starting from pdb structure {path2pdb}')
            ini_pos, pos, center_of_mass = geometry_from_pdb(cwd, dataset, record, name, cycle, replica, path2pdb, compact_ini, CoarseGrained=CoarseGrained, ssdomains=ssdomains)  # nm
            ini_pos, center_of_mass = place_chains(ini_pos+center_of_mass, chains, L, isIDP)
        else:
            ini_pos = np.array(xy_spiral_array(N))
            ini_pos, center_of_mass = place_chains(ini_pos, chains, L, isIDP)
            pos = ini_pos
    else:
        if use_pdb:
            print(f'Starting from pdb structure {path2pdb}')
            ini_pos, pos, center_of_mass = geometry_from_pdb(cwd, dataset, record, name, cycle, replica, path2pdb, compact_ini, CoarseGrained=CoarseGrained, ssdomains=ssdomains)
        else:
            spiral = True
            if spiral:
                ini_pos = xy_spiral_array(N)
            else:
                ini_pos = [[L / 2, L / 2, L / 2 + (i - N / 2.) * .38] for i in range(N)]
            ini_pos = np.array(ini_pos)
            pos = ini_pos

    top = build_topology(fasta, n_chains=chains)
    print("Lz: ", Lz)
    md.Trajectory(ini_pos+center_of_mass, top, 0, [L, L, Lz], [90, 90, 90]).save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/ini_beads.pdb', force_overwrite=True)
    a = md.Trajectory(ini_pos+np.array([L/2, L/2, Lz/2]), top, 0, [L, L, Lz], [90, 90, 90])
    a.save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/top_{replica}.pdb', force_overwrite=True)
    print("building system...")
    # build openmm system
    system = openmm.System()
    # box
    a, b, c = build_box(L, L, Lz)
    system.setDefaultPeriodicBoxVectors(a, b, c)
    # load topology into system
    pdb = app.pdbfile.PDBFile(f'{cwd}/{dataset}/{record}/{cycle}/top_{replica}.pdb')
    # particles and termini
    system = add_particles(system, residues, prot, n_chains=n_chains)
    if slab:
        dmap = euclidean(pos[:N], pos[:N])
    else:
        dmap = euclidean(pos, pos)
    np.save((f"{cwd}/{dataset}/{record}/{cycle}/dmap.npy"), dmap)
    # interactions
    print("set_interactions....")
    hb, yu, ah = set_interactions(system, residues, prot, lj_eps, cutoff, yukawa_kappa, yukawa_eps, N,
                                  n_chains=n_chains, CoarseGrained=CoarseGrained, dismatrix=dmap, isIDP=isIDP,
                                  fdomains=fdomains)
    print("set_interactions.... done")
    print("use_hnetwork: ", use_hnetwork)
    # harmonic network (currently unavailable for slab sim)
    if use_hnetwork:
        cs, yu, ah = set_harmonic_network(N, dmap, pae, yu, ah, n_chains=n_chains, ssdomains=ssdomains, k_restraint=k_restraint)
        print(f"k_restraint used: {k_restraint}")
        system.addForce(cs)
    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    if equilibrium:
        rcent = force2center([L, L, Lz], k_eq)
        for i in range(len(ini_pos)):
            rcent.addParticle(i)
        system.addForce(rcent)
    # use langevin integrator
    integrator = openmm.openmm.LangevinIntegrator(temp * unit.kelvin, 0.01 / unit.picosecond, 0.01 * unit.picosecond)
    print(integrator.getFriction(), integrator.getTemperature())

    # assemble simulation
    if gpu:
        simulation = app.simulation.Simulation(
            pdb.topology, system, integrator,openmm.Platform.getPlatformByName("CUDA"),dict(CudaPrecision='mixed'))
    else:
        simulation = app.simulation.Simulation(
            pdb.topology, system, integrator, openmm.Platform.getPlatformByName('CPU'), dict(Threads=str(Threads)))

    if not equilibrium:
        # load pos and vel of last frame, and chk file can be used within different nodes at least on Computerome
        print(f'Reading lf.pdb')
        if cycle==0:
            simulation.context.setPositions(app.pdbfile.PDBFile(f"{cwd}/{dataset}/{record}/{cycle}/equilibrium_lf.pdb").positions)
        else:
            simulation.context.setPositions(app.pdbfile.PDBFile(f"{cwd}/{dataset}/{record}/{cycle-1}/production_lf.pdb").positions)
        simulation.reporters.append(
            app.dcdreporter.DCDReporter(f'{cwd}/{dataset}/{record}/{cycle}/production_{cycle}.dcd', wfreq,
                                        enforcePeriodicBox=False, append=False))
        simulation.reporters.append(
            app.statedatareporter.StateDataReporter(f'{cwd}/{dataset}/{record}/{cycle}/statedata_{cycle}.log',
                int(wfreq), step=True, speed=True, elapsedTime=True, separator='\t', progress=True, remainingTime=True,
                totalSteps=steps))
        # simulation.reporters.append(app.checkpointreporter.CheckpointReporter(file=f"{cwd}/{dataset}/{record}/{cycle}/checkpoint{cycle}.chk",reportInterval=wfreq))
    else:
        simulation.context.setPositions(pdb.positions)
        # simulation.minimizeEnergy()
        # state = simulation.context.getState(getPositions=True)
        # pos2 = state.getPositions(asNumpy=True)
        # enforcePeriodicBox=False makes sure that the saved molecules are whole;
        # when openmm tries to keep the center of mass of molecule in the box, it will modify the x,y,z of every atoms separately;
        # which means only if x(or y or z) of COM is out of box, it will modify x of every atom so that x of COM is in the box;
        # Therefore, if just the x of a single atom is out of box, x might not be modified into the box;
        # and the center of simulation box is (L/2, L/2, L/2)
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{cwd}/{dataset}/{record}/{cycle}/equilibrium_{cycle}.dcd', wfreq,
                                                                enforcePeriodicBox=False, append=False))

        simulation.reporters.append(
            app.statedatareporter.StateDataReporter(f'{cwd}/{dataset}/{record}/{cycle}/statedata_{cycle}_equi.log', int(wfreq),
                                                    step=True, speed=True, elapsedTime=True, separator='\t', progress=True,
                                                    remainingTime=True, totalSteps=steps))
        # simulation.reporters.append(app.checkpointreporter.CheckpointReporter(file=f"{cwd}/{dataset}/{record}/{cycle}/checkpoint{cycle}_equi.chk",reportInterval=wfreq))

    print("running simulations....")
    starttime = time.time()  # begin timer
    print("steps: ", steps)
    simulation.step(steps)
    endtime = time.time()  # end timer
    target_seconds = endtime - starttime  # total used time
    print(
        f"{record} total simulations used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")
    if equilibrium:
        simulation.saveState(f"{cwd}/{dataset}/{record}/{cycle}/final_state{cycle}_equi.xml")
    else:
        simulation.saveState(f"{cwd}/{dataset}/{record}/{cycle}/final_state{cycle}.xml")
    if not equilibrium:
        backup_traj(f"{cwd}/{dataset}/{record}/traj_pool.dcd",
            f"{cwd}/{dataset}/{record}/{cycle}/production_{cycle}.dcd",
            f"{cwd}/{dataset}/{record}/{cycle}/top_{replica}.pdb")
        center_slab(cwd, dataset, record, cycle, replica, f"{cwd}/{dataset}/{record}/traj_pool.dcd")
        calcProfiles_rc(cwd, dataset, record, cycle)

    traj_pool = MDAnalysis.Universe(f"{cwd}/{dataset}/{record}/{cycle}/{'equilibrium_' if equilibrium else 'production_'}{cycle}.dcd")
    with MDAnalysis.Writer(f"{cwd}/{dataset}/{record}/{cycle}/{'equilibrium_' if equilibrium else 'production_'}{cycle}_pbc.dcd", traj_pool.atoms.n_atoms) as W:
        while True:
            transformations.wrap(traj_pool.atoms)(traj_pool.trajectory.ts)
            W.write(traj_pool.atoms)
            try:
                traj_pool.trajectory.next()
            except Exception:
                break
    traj = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{'equilibrium_' if equilibrium else 'production_'}{cycle}_pbc.dcd", f"{cwd}/{dataset}/{record}/{cycle}/top_0.pdb")
    pdb = traj[-1]
    pdb.save_pdb(f"{cwd}/{dataset}/{record}/{cycle}/{'equilibrium_' if equilibrium else 'production_'}lf.pdb")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_simerge', nargs='?', default='config_simerge.yaml', const='config_simerge.yaml', type=str)
    args = parser.parse_args()
    simulate(yaml.safe_load(open(args.config_simerge, 'r')))
