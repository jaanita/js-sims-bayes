#a function to write module input files given design params
import math
import os
import os.path

def write_module_inputs(
    outdir = 'test/',
    #design point
    design_point_id = 0,

    #trento params
    projectile = 'Pb',
    target = 'Pb',
    sqrts = 2760,
    inel_nucleon_cross_section = 6.4,
    trento_normalization = 13.94,
    trento_reduced_thickness = 0.007,
    #trento_fluctuation_k = 0.3,
    trento_fluctuation_k = 1.2,
    trento_nucleon_width = 0.956,
    trento_nucleon_min_dist = 1.27,

    #freestream-milne Parameters
    tau_R = 1.16, # fm / c
    alpha = 0.0,

    #shear viscosity p'zation
    eta_over_s_T_kink_in_GeV = 0.155,
    eta_over_s_low_T_slope_in_GeV = 0.0,
    eta_over_s_high_T_slope_in_GeV = 1.11,
    eta_over_s_at_kink = 0.081,

    #bulk viscosity p'zation
    zeta_over_s_max = 0.052,
    zeta_over_s_width_in_GeV = 0.022,
    zeta_over_s_T_peak_in_GeV = 0.183,
    zeta_over_s_lambda_asymm = 0.0,

    #relaxation times
    shear_relax_time_factor = 5.0,
    bulk_relax_time_factor = 1.0 / 14.55,
    #bulk_relax_time_power = 2.,

    #hydro params
    T_switch = 0.151,

    ):

    #the jetscape seed sets the seed in smash, set to 0 for random (clocktime) seed
    js_seed = 0

    #formula for Energy-dependent freestreaming time: tau_fs = tau_R * (e_T / e_R) ^ alpha
    e_dep_fs_time = 1 # switch for energy dependent freestreaming time
    e_R = 4.0 # GeV / fm^3
    #this is just a dummy parameter that will be overridden by formula above
    tau_fs = 1.16

    #iS3D Parameters
    #delta-f mode will be overwritten by the run-events script
    delta_f_mode = 4 # 1: 14 moment, 2: C.E., 3: McNelis feq_mod, 4: Bernhard feq_mod
    rap_max = 2.0    # dN/dY sampled is flat for y in (-rap_max, rap_max) and zero outside
    min_num_hadrons = 50000
    max_num_samples = 2000
    set_T_switch = 0 #if on, iS3D will use Tc read in iS3D_parameters.dat as temperature

    #SMASH Parameters
    max_time_smash = 1000.0 #max run time [fm/c]

    #this chooses grid spacing based on nucleon width
    dx = 0.15 * trento_nucleon_width #[fm]
    dy = 0.15 * trento_nucleon_width #[fm]

    #set hydro time step based on grid spacing for convergence
    dtau = dx / 8.0

    #choose a grid size large enough to capture central events
    if (sqrts == 200):
        L_x = 12.0 #[fm]
        L_y = 12.0 #[fm]
    elif (sqrts == 62.4):
        L_x = 12.0 #[fm]
        L_y = 12.0 #[fm]
    elif (sqrts == 39):
        L_x = 12.0 #[fm]
        L_y = 12.0 #[fm]
    elif (sqrts == 27):
        L_x = 12.0 #[fm]
        L_y = 12.0 #[fm]
    elif (sqrts == 14.5):
        L_x = 12.0 #[fm]
        L_y = 12.0 #[fm]
    elif (sqrts == 11.5):
        L_x = 12.0 #[fm]
        L_y = 12.0 #[fm]
    elif (sqrts == 7.7):
        L_x = 12.0 #[fm]
        L_y = 12.0 #[fm]
    elif (sqrts == 2760):
        L_x = 18.0 #[fm]
        L_y = 18.0 #[fm]
    elif (sqrts == 5020):
        L_x = 22.0 #[fm]
        L_y = 22.0 #[fm]
    elif (sqrts == 5440):
        L_x = 18.0 #[fm]
        L_y = 18.0 #[fm]
    else :
        print("invalid sqrts !")
        exit()


    nx = 1.0 + (2.0 * L_x)/dx
    ny = 1.0 + (2.0 * L_y)/dy
    #get nearest ODD integer (freestream needs odd number of points!)
    nx = int( math.ceil(nx) )
    ny = int( math.ceil(ny) )
    if ( (nx % 2) == 0):
        nx += 1
    if ( (ny % 2) == 0):
        ny += 1

    #recalculate grid size
    L_x = ( (nx-1) / 2.0) * dx
    L_y = ( (nx-1) / 2.0) * dx
    #TRENTo needs slightly larger for odd number of points
    max_x = L_x + 0.5*dx #max x [fm]
    max_y = L_y + 0.5*dy #may y [fm]
    #TRENTo parameters
    cent_low = 0
    cent_high = 100

    #For Xenon, we want to use the deformed nucleus in TRENTo
    if projectile == 'Xe':
        projectile = 'Xe2'
        target = 'Xe2'

    #write appropriate input files

    design_point_outdir=os.path.join(outdir,str(design_point_id))
    os.mkdir(design_point_outdir)

    #freestream file
    fs_file = open(os.path.join(design_point_outdir, 'freestream_input'),'w')

    fs_file.write("OUTPUTFORMAT 2\n")
    fs_file.write("BARYON 0\n")
    fs_file.write("IC_ENERGY 5\n")
    fs_file.write("IC_BARYON 1\n")
    fs_file.write("ETA_WIDTH 0.5\n")
    fs_file.write("ETA_FLAT 0.5\n")
    fs_file.write("SIGMA 0.5\n")
    fs_file.write("SIGMA_B 0.5\n")
    fs_file.write("DIM_X " + str(nx) + "\n")
    fs_file.write("DIM_Y " + str(ny) + "\n")
    fs_file.write("DIM_ETA 1\n")
    fs_file.write("DIM_RAP 1\n")
    fs_file.write("DIM_PHIP 500\n")
    fs_file.write("DX " + str(dx) + "\n")
    fs_file.write("DY " + str(dy) + "\n")
    fs_file.write("DETA 0.1\n")
    fs_file.write("DRAP 0.07\n")
    fs_file.write("DTAU " + str(tau_fs) + "\n")
    fs_file.write("TAU0 0.0\n")
    fs_file.write("EOS_TYPE 1\n")
    dummy_e_c = 1.7   # used to compute energy density inside freeze-out surface
    fs_file.write("E_FREEZE " + str(dummy_e_c) + "\n")
    fs_file.write("VISCOUS_MATCHING 1 \n")
    fs_file.write("E_DEP_FS " + str(e_dep_fs_time) + "\n")
    fs_file.write("E_R " + str(e_R) + "\n")
    fs_file.write("TAU_R " + str(tau_R) + "\n")
    fs_file.write("ALPHA " + str(alpha))

    fs_file.close()

    #MUSIC file
    music_file = open(os.path.join(design_point_outdir, 'music_input'),'w')

    music_file.write("echo_level  1\n")                  # control the mount of message output to screen
    music_file.write("mode 2\n")                         # MUSIC running mode
    music_file.write("Initial_profile 42\n")             # type of initial condition
    music_file.write("initialize_with_entropy 0\n")      # init with entropy density or energy density
    music_file.write("s_factor 1.00\n")                  # normalization factor read in
    music_file.write("boost_invariant  1\n")             # whether the simulation is boost-invariant
    music_file.write("Initial_time_tau_0 " +str(tau_fs) + "\n")# starting time of the hydro
    music_file.write("Total_evolution_time_tau 30.\n")   # the maximum allowed running time
    music_file.write("Delta_Tau " + str(dtau) + "\n")      # time step to use in the evolution [fm/c]
    music_file.write("Eta_grid_size 1.0\n")              # spatial rapidity range
    music_file.write("Grid_size_in_eta 1\n")             # number of the grid points in spatial
    music_file.write("X_grid_size_in_fm " + str(max_x) + "\n")# spatial range along x direction in the
    music_file.write("Y_grid_size_in_fm " + str(max_y) + "\n")# spatial range along y direction in the
    music_file.write("Grid_size_in_y " + str(nx) + "\n")             # number of the grid points in y direction
    music_file.write("Grid_size_in_x " + str(ny) + "\n")             # number of the grid points in x direction
    music_file.write("EOS_to_use 9\n")                   # type of the equation of state
    music_file.write("reconst_type  1\n")                # 0: solve energy density for hydro eqns. 1: solve flow velocity for hydro eqns.
    music_file.write("Minmod_Theta 1.8\n")               # theta parameter in the min-mod like limiter
    music_file.write("Runge_Kutta_order 2\n")            # order of Runge_Kutta for temporal evolution

    music_file.write("Viscosity_Flag_Yes_1_No_0 1\n")    # turn on viscosity in the evolution
    music_file.write("Include_Shear_Visc_Yes_1_No_0 1\n")# include shear viscous effect

    music_file.write("T_dependent_Shear_to_S_ratio  3\n")# flag to use temperature dep. \eta/s(T)
    music_file.write("T_dependent_Bulk_to_S_ratio  3\n")# flag to use temperature dep. \zeta/s(T)

    music_file.write("eta_over_s_T_kink_in_GeV " + str(eta_over_s_T_kink_in_GeV) + "\n")
    music_file.write("eta_over_s_low_T_slope_in_GeV " + str(eta_over_s_low_T_slope_in_GeV) + "\n")
    music_file.write("eta_over_s_high_T_slope_in_GeV " + str(eta_over_s_high_T_slope_in_GeV) + "\n")
    music_file.write("eta_over_s_at_kink " + str(eta_over_s_at_kink) + "\n")

    music_file.write("zeta_over_s_max " + str(zeta_over_s_max) + "\n")
    music_file.write("zeta_over_s_width_in_GeV " + str(zeta_over_s_width_in_GeV) + "\n")
    music_file.write("zeta_over_s_T_peak_in_GeV " + str(zeta_over_s_T_peak_in_GeV) + "\n")
    music_file.write("zeta_over_s_lambda_asymm " + str(zeta_over_s_lambda_asymm) + "\n")

    music_file.write("shear_relax_time_factor " + str(shear_relax_time_factor) + "\n")
    music_file.write("bulk_relax_time_factor " + str(bulk_relax_time_factor) + "\n")
    #music_file.write("bulk_relax_time_power " + str(bulk_relax_time_power) + "\n")

    music_file.write("Include_Bulk_Visc_Yes_1_No_0 1\n") # include bulk viscous effect
    music_file.write("Include_second_order_terms 1\n")   # include second order non-linear coupling terms
    music_file.write("store_hydro_info_in_memory 1\n")   # flag to store hydro info in memory
    music_file.write("output_evolution_data 2\n")        # flag to output evolution history to file
    music_file.write("outputBinaryEvolution  1\n") 
    music_file.write("output_evolution_T_cut "+str(T_switch) +"\n") 
    music_file.write("output_evolution_every_N_eta  1\n") 
    music_file.write("output_evolution_every_N_y  1\n")
    music_file.write("output_evolution_every_N_x  1\n")
    music_file.write("output_evolution_every_N_timesteps  5\n")
    music_file.write("Do_FreezeOut_Yes_1_No_0 1\n")      # flag to find freeze-out surface
    music_file.write("Do_FreezeOut_lowtemp 0\n")         # flag to include cold corona
    music_file.write("freeze_out_method 4\n")            # method for hyper-surface finder
    music_file.write("average_surface_over_this_many_time_steps 5\n")   # the step skipped in the tau
    #music_file.write("epsilon_freeze " + str(e_c) + "\n")            # the freeze out energy density (GeV/fm^3)
    music_file.write("use_eps_for_freeze_out 0\n")       # 0: use temperature, 1: use energy density
    music_file.write("T_freeze " + str(T_switch) + "\n")                 # freeze-out temperature (GeV)
    music_file.write("EndOfData\n")

    music_file.close()

    #iS3D File
    iS3D_file = open(os.path.join(design_point_outdir, 'iS3D_parameters.dat'),'w')

    iS3D_file.write("operation = 2\n")
    iS3D_file.write("mode      = 6\n")
    iS3D_file.write("hrg_eos   = 3\n")
    iS3D_file.write("set_FO_temperature          = " + str(set_T_switch) + "\n")
    iS3D_file.write("T_switch                    = " + str(T_switch) + "\n")
    iS3D_file.write("dimension = 2\n")
    iS3D_file.write("df_mode                     = " + str(delta_f_mode) + "\n")
    iS3D_file.write("include_baryon                 = 0\n")
    iS3D_file.write("include_bulk_deltaf            = 1\n")
    iS3D_file.write("include_shear_deltaf           = 1\n")
    iS3D_file.write("include_baryondiff_deltaf      = 0\n")
    iS3D_file.write("regulate_deltaf                = 0\n")
    iS3D_file.write("outflow                        = 1\n")
    iS3D_file.write("deta_min                = 1.e-6\n")
    iS3D_file.write("group_particles            = 0\n")
    iS3D_file.write("particle_diff_tolerance    = 0.01\n")
    iS3D_file.write("mass_pion0                 = 0.138\n")
    iS3D_file.write("do_resonance_decays = 0\n")
    iS3D_file.write("lightest_particle       = 111\n")
    iS3D_file.write("oversample                  = 1\n")
    iS3D_file.write("min_num_hadrons             = " + str(min_num_hadrons) + "\n")
    iS3D_file.write("max_num_samples             = " + str(max_num_samples) + "\n")
    iS3D_file.write("fast                        = 1\n")
    iS3D_file.write("y_cut                       = " + str(rap_max) + "\n")
    iS3D_file.write("sampler_seed        = -1\n")

    #these only used for testing, are dummys
    iS3D_file.write("test_sampler = 0\n")
    iS3D_file.write("pT_lower_cut = 0.0\n")
    iS3D_file.write("pT_upper_cut = 3.0\n")
    iS3D_file.write("pT_bins = 100\n")
    iS3D_file.write("y_bins                      = 50\n")
    iS3D_file.write("eta_cut                     = 7\n")
    iS3D_file.write("eta_bins                    = 70\n")
    iS3D_file.write("tau_min = 0.0\n")
    iS3D_file.write("tau_max = 12.0\n")
    iS3D_file.write("tau_bins = 120\n")
    iS3D_file.write("r_min = 0.0\n")
    iS3D_file.write("r_max = 10.0\n")
    iS3D_file.write("r_bins = 50\n")

    iS3D_file.close()

    #the jetscape init xml file
    #note that this file can potentially override parameters set in the MUSIC input file
    js_file = open(os.path.join(design_point_outdir, 'jetscape_init.xml'),'w')

    js_file.write("<?xml version=\"1.0\"?>\n")
    js_file.write(" <jetscape>\n")
    #js_file.write("  <debug> on </debug>\n")
    #js_file.write("  <remark> off </remark>\n")
    #js_file.write("  <vlevel> 0 </vlevel>\n")
    js_file.write("<nEvents> 10 </nEvents>\n")
    js_file.write("<setReuseHydro> true </setReuseHydro>\n")
    js_file.write("<nReuseHydro> 10 </nReuseHydro>\n")
    js_file.write("<outputFilename>output_"+str(sqrts)+"/"+str(design_point_id)+"/test_out</outputFilename>\n")
    js_file.write("<JetScapeWriterAscii> on </JetScapeWriterAscii>\n")
    js_file.write("  <Random>\n")
    js_file.write("    <seed>" + str(js_seed) + "</seed>\n")
    js_file.write("  </Random>\n")

    #parameters common to TRENTo and MUSIC
    js_file.write("  <IS>\n")
    js_file.write("    <grid_max_x> " + str(max_x) + " </grid_max_x>\n")
    js_file.write("    <grid_max_y> " + str(max_y) + " </grid_max_y>\n")
    js_file.write("    <grid_max_z> 0 </grid_max_z>\n")
    js_file.write("    <grid_step_x> " + str(dx) + " </grid_step_x>\n")
    js_file.write("    <grid_step_y> " + str(dy) + " </grid_step_y>\n")
    js_file.write("    <grid_step_z> 0.5 </grid_step_z>\n")

    #TRENTo parameters
    js_file.write("    <Trento>\n")
    js_file.write("             <PhysicsInputs  projectile=\'" + str(projectile) + "\'\n")
    js_file.write("                                             target=\'" + str(target) + "\'\n")
    js_file.write("                                             sqrts=\'" + str(sqrts) + "\'\n")
    js_file.write("                                             cross-section=\'" + str(inel_nucleon_cross_section) + "\'\n")
    js_file.write("                                             normalization=\'" + str(trento_normalization) + "\'>\n")
    js_file.write("             </PhysicsInputs>\n")
    js_file.write("             <CutInputs      centrality-low=\'" + str(cent_low) + "\'\n")
    js_file.write("                                     centrality-high=\'" + str(cent_high) + "\'>\n")
    js_file.write("             </CutInputs>\n")
    js_file.write("             <TransInputs    reduced-thickness=\'" + str(trento_reduced_thickness) + "\'\n")
    js_file.write("                                             fluctuation=\'" + str(trento_fluctuation_k) + "\'\n")
    js_file.write("                                             nucleon-width=\'" + str(trento_nucleon_width) + "\'\n")
    js_file.write("                                             nucleon-min-dist=\'" + str(trento_nucleon_min_dist) + "\'>\n")
    
    js_file.write("             </TransInputs>\n")
    js_file.write("             <LongiInputs    mean-coeff=\'1.0\'\n")
    js_file.write("                                             std-coeff=\'3.0\'\n")
    js_file.write("                                             skew-coeff=\'0.0\'\n")
    js_file.write("                                             skew-type=\'1\'\n")
    js_file.write("                                             jacobian=\'0.8\'>\n")
    js_file.write("             </LongiInputs>\n")
    js_file.write("    </Trento>\n")
    #js_file.write("    <initial_profile_path>../examples/test_hydro_files</initial_profile_path>\n")
    js_file.write("  </IS>\n")

    #these are dummies, not actually read by freestream-milne
    js_file.write("  <Preequilibrium>\n")
    js_file.write("    <NullPreDynamics> </NullPreDynamics>\n")
    js_file.write("  </Preequilibrium>\n")

    #fixed params for MUSIC
    js_file.write("  <Hydro>\n")
    js_file.write("    <MUSIC>\n")
    js_file.write("      <name>MUSIC</name>\n")
    js_file.write("      <MUSIC_input_file>"+str(sqrts)+"/"+str(design_point_id)+"/music_input</MUSIC_input_file>\n")
    js_file.write("      <Perform_CooperFrye_Feezeout>1</Perform_CooperFrye_Feezeout>\n")
    js_file.write("      <freezeout_temperature>" + str(T_switch) + "</freezeout_temperature>\n") 
    js_file.write("    </MUSIC>\n")
    js_file.write("  </Hydro>\n")

    #fixed params for iSS
    js_file.write("  <SoftParticlization>\n")
    js_file.write("    <iSS>\n")
    js_file.write("      <afterburner_type>2</afterburner_type>\n")
    js_file.write("      <Perform_resonance_decays>0</Perform_resonance_decays>\n")
    js_file.write("    </iSS>\n")
    js_file.write("  </SoftParticlization>\n")
    
    #params for SMASH
    js_file.write("  <Afterburner>\n")
    js_file.write("    <SMASH>\n")
    js_file.write("      <name>SMASH</name>\n")
    js_file.write("      <SMASH_config_file>../../SummerSchool2021/Jul22_Transport/smash_config.yaml</SMASH_config_file>\n")
    js_file.write("      <SMASH_particles_file>../../SummerSchool2021/Jul22_Transport/particles.txt</SMASH_particles_file>\n")
    js_file.write("      <SMASH_decaymodes_file>../../SummerSchool2021/Jul22_Transport/decaymodes.txt</SMASH_decaymodes_file>\n")
    js_file.write("      <end_time>" + str(max_time_smash) + "</end_time>\n")
    js_file.write("      <only_decays>0</only_decays>\n")
    js_file.write("    </SMASH>\n")
    js_file.write("  </Afterburner>\n")

    js_file.write("</jetscape>\n")

    js_file.close()
