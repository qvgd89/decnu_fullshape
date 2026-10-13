import sys
# sys.path.insert(0, "/global/homes/e/elbers/folps/desilike/desilike")
# sys.path.insert(0, "/global/homes/e/elbers/decnu_clean/pyclass")
# sys.path.insert(0, "/global/homes/e/elbers/decnu_cosmodesi/cosmoprimo")
# sys.path.insert(0, "/global/homes/e/elbers/folps/folpsD")

sys.path.insert(0, "/global/homes/e/elbers/decnu_main/pyclass")
sys.path.insert(0, "/global/homes/e/elbers/decnu_main/cosmoprimo")
# sys.path.insert(0, "/global/homes/e/elbers/folps/folpsD")

from pyclass import decnuclass
import desilike, inspect
print(inspect.getfile(desilike))
import os
os.environ["FOLPS_BACKEND"] = "jax" 
import folps, inspect
print(inspect.getfile(folps))
# import desilike, inspect
# print(inspect.getfile(desilike))
from synthetic_data_tools import *


import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles,  DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.full_shape import FOLPSv2TracerPowerSpectrumMultipoles, FOLPSAXTracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles, FOLPSTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable 
# from desilike.observables import ObservableCovariance
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike import setup_logging
from desilike.parameter import Parameter, ParameterPrior
import argparse
import sys, os, shutil
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
from datetime import datetime

setup_logging("info")





# In[ ]:


######### Settings #########

#model: LCDM or HS
model = 'decnu'
synthetic_model = 'model1'


base_dir = '/global/cfs/cdirs/desi/science/cpe/elbers/neutrino_mock_challenge/chains'
#Put 'True' to resume chain. 'False' to start from 0 steps
restart_chain = False
#Biasing and EFT parametrization: 'physical' or 'standard'
prior_basis = 'physical_aap' #Prior to be used 
# prior_basis = 'standard'

damping= 'lor'  #Choose from 'lor', 'exp', 'vdg'

kr_min = 0.02
kr_max = 0.201
kr_b0_max = 0.20
kr_b2_max = 0.03

hexa = False
bispectrum= False

if bispectrum==False:
    kr_b0_max = None
    kr_b2_max = None

#No need to change these
width_EFT = 12.5
width_SN0 = 2.0
width_SN2 = 5.0

pt_model = "EFT"  #Choose b/w folpsD and EFT
# pt_model = "folpsD"  #Choose b/w folpsD and EFT

sampler = 'cobaya'

set_emulator = True

A_full_status= False

b3_coev = False
GR_criteria = 0.05  # R - 1 < GR_criteria 


# List of tracers
# tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']  # Add more tracers as needed
tracers = ['LRG1']
#tracers = args.tracers
all_tracers = {'BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'}

if set(tracers) == all_tracers:
    tracers_str = "all"
else:
    tracers_str = "_".join(tracers)



chain_name = (
    f"{base_dir}/{tracers_str}"
    f"_{'std' if prior_basis == 'standard' else 'phys'}"
    f"_kr{kr_max:.3f}"
    f"{f'_kb0{kr_b0_max:.3f}_kb2{kr_b2_max:.3f}' if bispectrum else ''}"
    f"{'_hexa' if hexa else ''}"
    f"_{pt_model}"
    f"_{model}"
    f"_{'Afull' if A_full_status else 'Ano'}"
    f"_{'b3_coev' if b3_coev else 'b3_samp'}"
    f"{f'_damping_{damping}' if damping != 'lor' else ''}"
)


# if not set_emulator:
#     chain_name += '_noemu'
print(chain_name)



######## 

# No need to change anything beyond this

#########

from desilike import ParameterCollection

def make_params(prior_basis, width_EFT, width_SN0, width_SN2,pt_model='folpsD',b3_coev=True):
    params = {}  # plain dict so values are not auto-assigned from prior midpoints

    if prior_basis in ('physical', 'physical_aap'):
        # Shared params
        params['b1p'] = {'prior': {'dist':'uniform','limits': [0.1,4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2/7*(sigma8_fid)**2, 'scale': 5}}
        if b3_coev:
            params['b3p'] = {'fixed':True}
        else:
            params['b3p'] = {'prior': {'dist': 'norm', 'loc': 23/42*(sigma8_fid)**4, 'scale': 1*(sigma8_fid)**4},'fixed':False}#TBD
        # PS-only
        params['alpha0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        if pt_model=='EFT': 
            params['X_FoG_pp'] = {'fixed':True}
            # params['X_FoG_bp'] = {'fixed':True}
        else: 
            params['X_FoG_pp'] = {'prior': {'dist':'uniform','limits': [0, 10]}} 
            # params['X_FoG_bp'] = {'prior': {'dist':'uniform','limits': [0, 15]}}

   

    else:
        # Shared params
        params['b1'] = {'prior': {'dist':'uniform','limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed':True}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 25}}#TBD
        # params['b3'] = {'fixed':True}

        # PS-only
        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        # params['alpha0'] = {'value':20,'fixed':True}
        # params['alpha2'] = {'value':-58.8,'fixed':True}
        # params['alpha4'] = {'value':0.0,'fixed':True}
        # params['sn0'] = {'value':-0.073,'fixed':True}
        # params['sn2'] = {'value':-6.38,'fixed':True}
        if pt_model=='EFT':
            params['X_FoG_p'] = {'fixed':True}
            params['X_FoG_b'] = {'fixed':True}
        else: 
            params['X_FoG_p'] = {'prior': {'dist':'uniform','limits': [0, 10]}} 
            params['X_FoG_b'] = {'prior': {'dist':'uniform','limits': [0, 15]}}
        # params['X_FoG_p'] = {'fixed':True}  # fixed in your snippet

        # BS-only
        Ppoisson=1/0.0002118763
        params['c1'] = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6*4}}
        params['c2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1*4}}
        params['Pshot'] ={'prior': {'dist': 'norm', 'loc': 0, 'scale': Ppoisson*4}}
        params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': Ppoisson*4}}
        

    return params



#Define a cosmology to get sigma_8, Omega_m and fR0
cosmo = Cosmoprimo(engine='decnuclass', kmax_pk=1)
cosmo.init.params['H0'] = dict(derived=True)
cosmo.init.params['Omega_m'] = dict(derived=True)
cosmo.init.params['sigma8_m'] = dict(derived=True) 
cosmo.init.params['rs_drag'] = dict(derived=True)
cosmo.init.params['logGamma_ncdm'] = dict(derived=False)
cosmo.init.params['ncdm_fluid_approximation'] = dict(derived=False, fixed=True, value=3)
cosmo.init.params['k_per_decade_for_bao'] = dict(derived=False, fixed=True, value=30)
cosmo.init.params['tol_perturbations_integration'] = dict(derived=False, fixed=True, value=1e-3)
#cosmo.init.params['fR0'] = dict(derived=False, latex ='f_{R0}')
fiducial = DESI() #fiducial cosmology

#Update cosmo priors
for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA', 'tau_reio', 'm_ncdm', 'logGamma_ncdm']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
        cosmo.params[param].update(fixed = True)
    if param == 'omega_b':
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})  #From simulations
    if param == 'h':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.5,0.9]})
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.05, 0.2]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [2.0, 4.0]})
    if param == 'm_ncdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.0, 1.0]}, delta = [0.5, 0.24, 0.24])
    if param == 'logGamma_ncdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [3, 6]}, ref={'dist': 'uniform', 'limits': [3, 6]}, delta=[4.5, 0.7, 0.7])

#Define tracer types and their corresponding redshifts
all_tracer_params = {
    'BGS': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.npy'
    },
    'LRG1': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.4-0.6.npy'
    },
    'LRG2': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.6-0.8.npy'
    },
    'LRG3': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.8-1.1.npy'
    },
    'ELG': {
       'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_ELG_LOPnotqso_GCcomb_z1.1-1.6.npy'
    },
    'QSO': {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_QSO_GCcomb_z0.8-2.1.npy'
    }
}  


all_tracer_redshifts = {
                        'BGS': 0.295,
                        'LRG1': 0.5,
                        'LRG2': 0.7058,
                        'LRG3': 0.9228,
                        'ELG': 1.3235,
                        'QSO': 1.4835
                        }

all_tracer_sigma8 = {
                        'BGS': 0.69376997,
                        'LRG1':0.6013900567354893, # from Peregrinus_PlanckM240
                        'LRG2': 0.53930063,
                        'LRG3': 0.50823223,
                        'ELG': 0.43197292,
                        'QSO': 0.41825647
}

tracer_params = {index: all_tracer_params[tracer] for index, tracer in enumerate(tracers)}
tracer_redshifts = {tracer: all_tracer_redshifts[tracer] for tracer in tracers}


#Iterate over each tracer and create the corresponding theory object
# Theories container: dict of dicts
theories = {}

for tracer in tracers:
    z = tracer_redshifts[tracer]
    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)
    sigma8_fid = fiducial.sigma8_z(z)
    # PS theory selection
    if pt_model == "rept_velocileptors":
        ps_theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template,
                                                                   prior_basis=prior_basis)
    else:
        ps_theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template,
                                                         prior_basis=prior_basis, A_full = A_full_status,b3_coev=b3_coev,damping=damping,sigma8_fid=sigma8_fid,h_fid=fiducial.h) 
        # ps_theory = FOLPSTracerPowerSpectrumMultipoles(template=template,
        #                                                  prior_basis=prior_basis)

    # BS theory always FOLPSv2 in your snippet
    bs_theory = FOLPSv2TracerBispectrumMultipoles(template=template,
                                                  prior_basis=prior_basis, A_full = A_full_status,
                                                  damping=damping,sigma8_fid=sigma8_fid,h_fid=fiducial.h,
                                                  )


    # Store
    theories[tracer] = {"ps": ps_theory, "bs": bs_theory}

# --- Ref values from profiler runs (Hardcode for now) ----
    # if tracer == 'LRG2':
    ref_values = {"LRG2":{'b1p':1.111, 'b2p':0.181, 'bsp':-0.182, 'b3p': 23/42*(0.6)**4},
                 "LRG1": {'b1p':1.116, 'b2p':0.072, 'bsp':-0.17, 'b3p': 23/42*(0.6)**4},
                 "LRG3": {'b1p':1.128, 'b2p':0.197, 'bsp':-0.16, 'b3p': 23/42*(0.6)**4},
                 "ELG": {'b1p':0.381, 'b2p':-0.022, 'bsp':-0.591, 'b3p': 23/42*(0.5)**4},
    
    }


    # --- Update parameters ---
    params = make_params(prior_basis, width_EFT, width_SN0, width_SN2, pt_model=pt_model,b3_coev=b3_coev)
    for name, p in params.items():
        for comp in ["ps", "bs"]:
            if name in theories[tracer][comp].params:
                    theories[tracer][comp].params[name].update(**p)  # unpack dict to avoid copying value
                    # if name in ref_values[tracer]:
                    #     print(tracer,ref_values[tracer][name])
                    #     theories[tracer][comp].params[name].update(ref={'dist': 'norm', 'loc': ref_values[tracer][name], 'scale': 1e-6})  # set to ref value for testing
    # print("b3=",params['b3p'].value)

def create_observable(comp,tracer='LRG2',z_ev=0.8,k_max=0.301,k_max_b0=None,k_max_b2=None,P4=False):
    
    # To avoid getting an error
    if k_max_b0==None:
        k_max_b0 = 0.12
    if k_max_b2==None:
        k_max_b2=0.08
        
    z_string = f"z{z_ev:.3f}"
    
    print("z_string:", z_string)

    dataset = build_pk_data_synthetic(
    tracer=tracer, z_string=z_string, model=synthetic_model,
    k_min_p = kr_min, k_max_p = kr_max,
    k_max_b0 = kr_b0_max, k_max_b2 = kr_b2_max,
    Vol = 27.0
)    
    print(dataset.keys())


    if "ps" in comp:
        from scipy.linalg import block_diag

      
        # wmatrix_pk = dataset['window'] #For now that function only constructs the PS window matrix, so we can use it for both multipoles. In the future, if we have different window matrices for each multipole, we can modify the code accordingly.
        data_p = np.concatenate([dataset['pkl0'], dataset['pkl2']])
        cov_array = dataset['cov_array']
        k_pk = dataset['k_data']
        cov_pk = dataset['cov_pk']
        #cov_pk = cov_array[:2*len(k_pk), :2*len(k_pk)]
        
        print(data_p.shape)
        print(cov_pk.shape)

       
        # cov_pk = cov_pk/hartlap
       
        ps_obs= TracerPowerSpectrumMultipolesObservable(
            data=dataset['pk_lsstypes'],
            covariance=cov_pk,
            theory=theories[tracer]["ps"],
            # window=wmatrix_pk,
            # kin = dataset['k_window'],
            # ellsin=[0,2], ells=(0,2), k=k_pk,
        )

        return ps_obs,None,None



#Create observables for each tracer
observables = {}
for tracer in tracers:
    ps_obs, bs_obs, cov_array =  create_observable(["ps"],tracer,tracer_redshifts[tracer],kr_max,kr_b0_max,kr_b2_max,hexa)
    observables[tracer] = {
        "ps": ps_obs,
    }



if set_emulator:
    for tracer in tracers:
        z=all_tracer_redshifts[tracer]
        for comp in ["ps"]:  # handle PS and BS separately
            obs = observables[tracer][comp]

            # emulator_filename = 'Emulator_test_sims_ps/ps_emu_LRG2_LRG2_0.201_folpsax.npy'
            if comp=='ps':            
                emulator_filename = f'./Emulators/Emulator_{comp}/{comp}_emu_{tracer}_z{z}_{kr_max}_Afull_{A_full_status}_{model}_{synthetic_model}.npy'
            else: 
                kr_b2_max_label= kr_b2_max if kr_b2_max>0.08 else 0.08
                emulator_filename = f'./Emulators/Emulator_{comp}/{comp}_emu_{tracer}_z{z}_{kr_max}_{kr_b0_max}_{kr_b2_max_label}_Afull_{A_full_status}_{model}_{synthetic_model}.npy'
                
            os.makedirs(os.path.dirname(emulator_filename), exist_ok=True)

            if os.path.exists(emulator_filename):
                print(f"{comp.upper()} emulator for tracer {tracer} already exists, loading it now")
                emulator = EmulatedCalculator.load(emulator_filename)
                theories[tracer][comp].init.update(pt=emulator)
                # obs.theory.init.update(pt=emulator)

            else:
                print(f"Fitting {comp.upper()} emulator for tracer {tracer}")
                # Start from the underlying PT theory
                # if comp=='ps':
                #     theory = obs.wmatrix.theory
                # else:
                #     theory = obs.theory
                theory=obs.theory

                emulator = Emulator(
                    theory.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=4)
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                emulated_pt.save(emulator_filename)
                theories[tracer][comp].init.update(pt=emulated_pt)
                # obs.theory.init.update(pt=emulated_pt)


            

print('All theories have been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED, proceeding without emulation')


print('All theories have been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED, proceeding without emulation')



#Analytic marginalization over eft and nuisance parameters
for i in (tracers): 
    if prior_basis in ('physical', 'physical_aap'):
        params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
         params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    for param in params_list:    
        theories[i]['ps'].params[param].update(derived = '.marg')
        
    # theories[i]['ps'].params['b3'].update(derived = '32/315*({b1}-1)')
    # print("b1=",theories[i]['ps'].params['b1'].value,"b3=",theories[i]['ps'].params['b3'].value)
    
        
#Rename the eft and nuisance parameters to get a parameter for each tracer (i.e. QSO_alpha0, QSO_alpha2, BGS_alpha0,...)        
# for i in range(len(theories)):    
    for param in theories[i]['ps'].init.params:
        # Update latex just to have better labels
        param.update(namespace='{}'.format(i)) 
    for param in theories[i]['ps'].all_params:
        print(param,':',theories[i]['ps'].all_params[param].prior)

        

#Create a likelihood per theory object

Likelihoods = []
for tracer in tracers:
        Likelihoods.append(ObservablesGaussianLikelihood(
        observables=[observables[tracer]['ps']],
        ))
        # Likelihoods.append(ObservablesGaussianLikelihood(
        # observables=[observables[tracer]['ps']],
        # ))
   # observables=[observables[tracer]['ps'],observables[tracer]['bs']],
   #          covariance=cov_arr
likelihood = SumLikelihood(Likelihoods)








import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--run_chains", action="store_true", help="To run a chain")
parser.add_argument("--plot_bestfit", action="store_true", help="Plot Bestfit")
parser.add_argument("--plot_chains", action="store_true", help="Plot Chains")
parser.add_argument("--test", action="store_true", help="testing")    

args = parser.parse_args()




from desilike.samples import Chain
import matplotlib.pyplot as plt
from pathlib import Path
from desilike import setup_logging
setup_logging()

def load_chain(fi, burnin=0.3):
    from desilike.samples import Chain
    # chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi]
    chains = [Chain.load(fi).remove_burnin(burnin)]
    chain = chains[0].concatenate(chains)
    print(f'chain: {chain}')
    return chain


if args.test:
    print("likelihood: ",likelihood())
    print(likelihood.varied_params)
    print("Test Successful!")
    
    

if args.plot_bestfit:
    chain_path = Path(f'{chain_name}.npy')
    chain = load_chain(chain_path)
    likelihood(**chain.choice(index='argmax', input=True))
    observables['LRG2']['ps'].plot(fn='test_bestfit.png',kw_save={'dpi':250})




if args.plot_chains:
    chain_path = Path(f'{chain_name}.npy')
    chain = load_chain(chain_path)
    samples2 = chain.to_getdist()

    import h5py

    filename = "/global/cfs/cdirs/desicollab/users/isaacmgm/Abacus_2ndGen_Fits/folpsDBaccoemu/chains/c_FolpsEFT_LRG_z0.800_Pkmax-0.201_bsfree.h5"
    with h5py.File(filename, "r") as f:
        print(list(f.keys()))  # Shows top-level datasets/groups
        print(list(f['mcmc'].keys()))
    import emcee
    import numpy as np
    
    backend = emcee.backends.HDFBackend(filename, read_only=True)
    
    # Get total chain shape: (nwalkers, nsteps, ndim)
    chain_shape = backend.get_chain().shape
    nwalkers, nsteps, ndim = chain_shape
    
    # Set burn-in as 30% of total steps
    burnin = int(0.5 * nsteps)
    
    # Get flat chain, discarding burn-in
    samples = backend.get_chain(discard=burnin, flat=True)  # shape: (n_samples, ndim)
    
    # Select first 4 params and downsample by 10 for plotting speed
    samples_subset = samples[:, :4][::10]

    from getdist import MCSamples, plots
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    

    import matplotlib as mpl
    
    planck_truths = {
        'h': 0.6736,
        'omega_cdm': 0.12,
        'omega_b': 0.02237,
        'logA': np.log(10**10 * 2.0830e-9)  # log(10^10 A_s), or set to your matching definition
    }
    # High-resolution Retina output (for notebooks)
    
    
    # Use LaTeX for all matplotlib text rendering
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 18
    
    param_names = ['h', 'omega_cdm', 'omega_b', 'logA']
    param_labels = [r'$h$', r'$\Omega_{\rm cdm}$', r'$\Omega_{\rm b}$', r'$\log(10^{10}A_s)$']
    
    gdsamples = MCSamples(samples=samples_subset, names=param_names, labels=param_labels)
    
    g = plots.get_subplot_plotter()
    
    # Increase axis font sizes
    g.settings.axes_fontsize = 20
    g.settings.lab_fontsize = 20
    
    g.settings.line_labels = False
    
    legend_labels = ['Chain 1', 'Chain 2']
    
    g.triangle_plot(
        [gdsamples,samples2],
        params=param_names,
        filled=[False, True],  # First is dotted, second is filled
        line_args=[{'ls': '--', 'color': 'black'}, {'lw': 1.2, 'color': 'steelblue'}],
        contour_colors=['black', 'steelblue'],
        contour_ls=['--', '-'], 
        markers = planck_truths# Make contours dotted for both chains
    )
    g.add_legend(legend_labels=[r"FolpsEFT_LRG_z0.800_Pkmax-0.201_bsfree","desilike FOLPSv2 kmax=0.201 (A_full = False)"],bbox_to_anchor=(0.5, 3.95),fontsize=18)
    g.export("test_chains.png")

if args.run_chains: 
    def load_chain(fi, burnin=0.3):
        from desilike.samples import Chain
        # chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi]
        chains = [Chain.load(fi).remove_burnin(burnin)]
        chain = chains[0].concatenate(chains)
        print(f'chain: {chain}')
        return chain
    
    #cov_chain_path = Path('/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/Abacus2gen_chains/base/LRG2_std_kr0.301_folpsD_Afull_b3_coev.npy')
    #cov_chain = load_chain(cov_chain_path,burnin=0.3)
    # cov_chain_path = Path('/global/cfs/cdirs/desicollab/users/prakharb/folpsD_paper_chains/Abacus2gen_chains/base/test_cov.npy')
    # cov_chain = load_chain(cov_chain_path,burnin=0.3)
    

#Run the sampler and save the chain
    from desilike.samplers import EmceeSampler, MCMCSampler
    
    if sampler == 'cobaya':
        if restart_chain is False:
            sampler = MCMCSampler(likelihood, save_fn = chain_name)
            
            sampler.run(check={'max_eigen_gr': GR_criteria})
            sampler.run(learn={'every': 40})
        else:
            sampler = MCMCSampler(likelihood ,save_fn = chain_name, 
                                  chains=f'{chain_name}.npy')
            #print(sampler.diagnostics)     # includes R-1, acceptance rate, etc.
            #print(sampler.converged)       # 
            sampler.run(check={'max_eigen_gr': GR_criteria})
        
    else:
        if restart_chain is False:
            sampler = EmceeSampler(likelihood ,save_fn = chain_name)
            sampler.run(check={'max_eigen_gr': GR_criteria})
        else:
            sampler = EmceeSampler(likelihood ,save_fn = chain_name, 
                                   chains=f'{chain_name}.npy')
            sampler.run(check={'max_eigen_gr': GR_criteria})
