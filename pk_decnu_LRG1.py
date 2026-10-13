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
os.environ["FOLPS_BACKEND"] = "jax"  # desilike now sets this itself on import; kept for clarity
import folps, inspect
print(inspect.getfile(folps))
# import desilike, inspect
# print(inspect.getfile(desilike))
from synthetic_data_tools import *


import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import DirectSpectrum2Template, FOLPSTracerSpectrum2Poles, FOLPSTracerSpectrum3Poles
from desilike.observables.galaxy_clustering import Spectrum2PolesObservable
from desilike.emulators import Emulator, Space
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.theories import CosmoprimoCosmology
from desilike.base import replace
from desilike.distributed import get_mpicomm
from cosmoprimo.fiducial import DESI
from desilike import setup_logging, build, get_params, Posterior, SumLikelihood
from desilike.parameter import Parameter, ParameterPrior
import argparse
import sys, os, shutil
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
from datetime import datetime
from pathlib import Path

setup_logging("info")

mpicomm = get_mpicomm()



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
#NOTE: in the physical bases these widths are in the previous desilike units (sigma8-scaled);
#make_params converts them to the new (sigma8_fid-scaled) parameters, see there.
width_EFT = 12.5
width_SN0 = 2.0
width_SN2 = 5.0

pt_model = "EFT"  #Choose b/w folpsD and EFT
# pt_model = "folpsD"  #Choose b/w folpsD and EFT

sampler = 'cobaya'  # 'cobaya' -> desilike's Metropolis-Hastings kernel (MH), anything else -> Emcee
nchains = 4  # independent MCMC chains (needed for the Gelman-Rubin test), distributed over the MPI ranks
mh_adaptation_steps = 10000  # MH: learn the proposal covariance during the first steps of each chain

set_emulator = True

#Chebyshev emulator of the PT calculator (replaces the old TaylorEmulatorEngine(method='finite', order=4)).
#The emulator is trained over a box declared by each cosmological parameter's `fd` spec,
#fd={'limits': [low, high]} (this replaces the old finite-difference step `delta`/`fd_eps`, which set
#the Taylor grid). Outside the box the emulated theory returns NaN, i.e. the sampler rejects the
#point: the box must comfortably cover the posterior. Check the chains do not pile up at its edges.
#By default: +/- 3 old finite-difference steps around the fiducial (the old order-4 Taylor stencil
#spanned +/- 2 steps), and the full prior range for the neutrino parameters.
emulator_limits = {
    'h': [0.6736 - 3 * 0.03, 0.6736 + 3 * 0.03],
    'omega_cdm': [0.12 - 3 * 0.007, 0.12 + 3 * 0.007],
    'omega_b': [0.02237 - 3 * 0.0015, 0.02237 + 3 * 0.0015],
    'logA': [3.0448 - 3 * 0.05, 3.0448 + 3 * 0.05],  # routed exactly by the FOLPS emulator, costs no nodes
    'm_ncdm': [0., 1.],  # old delta: center 0.5, step 0.24
    'logGamma_ncdm': [3., 6.],  # old delta: center 4.5, step 0.7
}
#Per-parameter Chebyshev level (2**level + 1 nodes along that axis; default 2) and the Smolyak
#budget, which sets the interaction terms. The node set is nested: raising the budget later reuses
#the evaluations stored in the checkpoint.
emulator_levels = {'h': 3, 'omega_cdm': 3, 'm_ncdm': 3, 'logGamma_ncdm': 3}
emulator_budget = 3
#Expand m_ncdm in sqrt(m_ncdm) (desilike's default expansion variable for the neutrino mass)
emulator_transforms = {'m_ncdm': 'sqrt'}

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
#desilike samplers now write one file per chain (samples_<i>.h5) into a directory,
#which must not have a suffix (chain_name contains dots)
chain_dir = Path(chain_name) / 'chains'


# if not set_emulator:
#     chain_name += '_noemu'
print(chain_name)



########

# No need to change anything beyond this

#########

def make_params(prior_basis, width_EFT, width_SN0, width_SN2, sigma8_fid, pt_model='folpsD', b3_coev=True, hexa=False):
    """Bias, counterterm and stochastic parameter settings, keyed by parameter basename.

    Parameter names no longer carry a 'p' suffix in the physical bases (b1p -> b1, alpha0p -> alpha0,
    X_FoG_pp -> X_FoG, ...). In the physical bases desilike now normalises them by
    A = sigma8(z) / sigma8_fid(z) instead of sigma8(z), so that, relative to the previous parameters,
    b1 = b1p / sigma8_fid, b2 = b2p / sigma8_fid**2, alpha_i = alpha_ip / sigma8_fid**2 (sn0, sn2 unchanged).
    The priors below are the previous ones rescaled accordingly. bs and b3 are now defined relative to
    their coevolution values (bK2 = bs / (A**2 sqrt(A_AP)) - 2/7 b1L, btd = b3 / (A**4 A_AP) + 23/42 b1L),
    so their priors are centred on 0, and b3_coev amounts to b3 = 0.
    """
    params = {}  # plain dict so values are not auto-assigned from prior midpoints

    if prior_basis in ('physical', 'physical_aap'):
        # Shared params
        params['b1'] = {'prior': {'dist':'uniform','limits': [0.1 / sigma8_fid, 4 / sigma8_fid]}}
        params['b2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5 / sigma8_fid**2}}
        params['bs'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5 / sigma8_fid**2}}
        if b3_coev:
            params['b3'] = {'value': 0., 'fixed':True}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}#TBD
        # PS-only
        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT / sigma8_fid**2}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT / sigma8_fid**2}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT / sigma8_fid**2}}
        params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        if pt_model=='EFT':
            params['X_FoG'] = {'value': 0., 'fixed':True}
        else:
            params['X_FoG'] = {'prior': {'dist':'uniform','limits': [0, 10]}}



    else:
        # Shared params
        params['b1'] = {'prior': {'dist':'uniform','limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed':True}  # set to its coevolution value, 32/315 (b1 - 1), below
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
        # A single X_FoG per theory now (previously X_FoG_p for the power spectrum and X_FoG_b for the bispectrum)
        if pt_model=='EFT':
            params['X_FoG'] = {'value': 0., 'fixed':True}
        else:
            params['X_FoG'] = {'prior': {'dist':'uniform','limits': [0, 10]}}
        # params['X_FoG_p'] = {'fixed':True}  # fixed in your snippet

        # BS-only
        # NOTE: Pshot and Bshot are now sn0 and snb0, in units of 1/nbar: set their priors when the bispectrum is used
        params['c1'] = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6*4}}
        params['c2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1*4}}

    if not hexa:
        # As desilike did before: alpha4 is fixed when the hexadecapole is not fitted
        params['alpha4'] = {'value': 0., 'fixed': True}

    for settings in params.values():
        # A prior means a varied parameter, as before
        if 'prior' in settings:
            settings.setdefault('fixed', False)

    return params



#Define a cosmology to get sigma_8, Omega_m and fR0
fiducial = DESI() #fiducial cosmology

#Parameters of the cosmology calculator: desilike's defaults, plus the decaying-neutrino ones
cosmo_params = CosmoprimoCosmology.propose_params(fiducial=fiducial)
cosmo_params.set(Parameter('logGamma_ncdm', value=4.5, latex=r'\log\Gamma_{\mathrm{ncdm}}'))
cosmo_params.set(Parameter('ncdm_fluid_approximation', value=3, fixed=True))
cosmo_params.set(Parameter('k_per_decade_for_bao', value=30, fixed=True))
cosmo_params.set(Parameter('tol_perturbations_integration', value=1e-3, fixed=True))
#cosmo_params.set(Parameter('fR0', latex='f_{R0}'))
for name in ['H0', 'Omega_m', 'sigma8_m', 'rs_drag']:
    cosmo_params.set(Parameter(name, derived=True))

#Cosmo priors; `fd` declares the emulator training range (see emulator_limits above)
cosmo_settings = {
    'h': {'prior': {'dist':'uniform','limits': [0.5,0.9]}},
    'omega_cdm': {'prior': {'dist':'uniform','limits': [0.05, 0.2]}},
    'omega_b': {'prior': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037}},  #From simulations
    'logA': {'prior': {'dist':'uniform','limits': [2.0, 4.0]}},
    'n_s': {'fixed': True},
    'tau_reio': {'fixed': True},
    'm_ncdm': {'prior': {'dist':'uniform','limits': [0.0, 1.0]}},
    'logGamma_ncdm': {'prior': {'dist':'uniform','limits': [3, 6]}, 'ref': {'dist': 'uniform', 'limits': [3, 6]}},
}
for name, settings in cosmo_settings.items():
    if 'prior' in settings:
        settings.setdefault('fixed', False)
    if name in emulator_limits:
        settings['fd'] = {'limits': emulator_limits[name]}


def update_cosmo_params(params):
    """Apply :data:`cosmo_settings` to the cosmological parameters found in ``params``."""
    for name, settings in cosmo_settings.items():
        if name in params:
            params[name].update(**settings)


update_cosmo_params(cosmo_params)

#kmax_pk is now a calculation parameter of the cosmology, passed through `precision`
cosmo = CosmoprimoCosmology(engine='decnuclass', fiducial=fiducial, params=cosmo_params,
                            precision={'calc_params': {'kmax_pk': 1.}})

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
sigma8_fids = {}

for tracer in tracers:
    z = tracer_redshifts[tracer]
    template = DirectSpectrum2Template(fiducial=fiducial, cosmo=cosmo, z=z)
    # The theory's amplitude normalisation: sigma8 of cb at z in the fiducial cosmology
    # (sigma8_fid and h_fid are no longer arguments: desilike reads them off the template's fiducial)
    sigma8_fids[tracer] = fiducial.get_fourier().sigma8_z(z, of='delta_cb')
    # PS theory; bias parameters are namespaced by tracer, e.g. LRG1.b1
    ps_theory = FOLPSTracerSpectrum2Poles(template=template, prior_basis=prior_basis, A_full=A_full_status,
                                          damping=damping, tracers=tracer)
    # FOLPSTracerPowerSpectrumMultipoles (FOLPS v1) and the velocileptors theories are now
    # FOLPSTracerSpectrum2Poles, LPTVelocileptorsTracerSpectrum2Poles and REPTVelocileptorsTracerSpectrum2Poles

    # Store
    theories[tracer] = {"ps": ps_theory}

    if bispectrum:
        # NOTE: as before, no bispectrum observable is built below
        theories[tracer]["bs"] = FOLPSTracerSpectrum3Poles(template=template, prior_basis=prior_basis, A_full=A_full_status,
                                                           damping=damping, tracers=tracer)

# --- Ref values from profiler runs (Hardcode for now) ----
    # if tracer == 'LRG2':
    # NOTE: in the previous (sigma8-scaled) parameter units, see make_params
    ref_values = {"LRG2":{'b1p':1.111, 'b2p':0.181, 'bsp':-0.182, 'b3p': 23/42*(0.6)**4},
                 "LRG1": {'b1p':1.116, 'b2p':0.072, 'bsp':-0.17, 'b3p': 23/42*(0.6)**4},
                 "LRG3": {'b1p':1.128, 'b2p':0.197, 'bsp':-0.16, 'b3p': 23/42*(0.6)**4},
                 "ELG": {'b1p':0.381, 'b2p':-0.022, 'bsp':-0.591, 'b3p': 23/42*(0.5)**4},

    }


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

        # NOTE: the observable sets the theory's k and ells to the data's (theory.update(k=..., ells=...)),
        # which re-creates its parameters: parameters are customized after this, below
        ps_obs= Spectrum2PolesObservable(
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


# --- Update parameters ---
for tracer in tracers:
    params = make_params(prior_basis, width_EFT, width_SN0, width_SN2, sigma8_fids[tracer], pt_model=pt_model, b3_coev=b3_coev, hexa=hexa)
    for comp in theories[tracer]:
        theory_params = get_params(theories[tracer][comp])
        for basename, p in params.items():
            name = f'{tracer}.{basename}'
            if name in theory_params:
                theory_params[name].update(**p)  # unpack dict to avoid copying value
                # if basename in ref_values[tracer]:
                #     print(tracer,ref_values[tracer][basename])
                #     theory_params[name].update(ref={'dist': 'norm', 'loc': ref_values[tracer][basename], 'scale': 1e-6})  # set to ref value for testing
        if prior_basis == 'standard' and b3_coev and f'{tracer}.b3' in theory_params:
            theory_params[f'{tracer}.b3'].update(derived='32. / 315. * (b1 - 1.)', depends={'b1': theory_params[f'{tracer}.b1']})
    # print("b3=",params['b3'].value)



if set_emulator:
    for tracer in tracers:
        z=all_tracer_redshifts[tracer]
        for comp in ["ps"]:  # handle PS and BS separately
            theory = theories[tracer][comp]
            pt = theory.pt

            # emulator_filename = 'Emulator_test_sims_ps/ps_emu_LRG2_LRG2_0.201_folpsax.npy'
            # Chebyshev emulators are written as HDF5 (.h5); emulators saved by the previous
            # desilike (.npy, Taylor) cannot be read by the new API and must be retrained
            if comp=='ps':
                emulator_filename = f'./Emulators/Emulator_{comp}/{comp}_emu_{tracer}_z{z}_{kr_max}_Afull_{A_full_status}_{model}_{synthetic_model}_cheb{emulator_budget}.h5'
            else:
                kr_b2_max_label= kr_b2_max if kr_b2_max>0.08 else 0.08
                emulator_filename = f'./Emulators/Emulator_{comp}/{comp}_emu_{tracer}_z{z}_{kr_max}_{kr_b0_max}_{kr_b2_max_label}_Afull_{A_full_status}_{model}_{synthetic_model}_cheb{emulator_budget}.h5'
            # Training checkpoint: an interrupted training resumes from it, and the nested node set
            # means a larger emulator_budget reuses the evaluations already made (same box only)
            checkpoint_filename = emulator_filename[:-len('.h5')] + '_checkpoint.npz'

            if mpicomm.rank == 0:
                os.makedirs(os.path.dirname(emulator_filename), exist_ok=True)
            mpicomm.Barrier()

            if os.path.exists(emulator_filename):
                print(f"{comp.upper()} emulator for tracer {tracer} already exists, loading it now")
                emulator = Emulator.read(emulator_filename)
                print(f"Emulator training box (expansion variables): {emulator.space.limits}")

            else:
                print(f"Fitting {comp.upper()} emulator for tracer {tracer}")
                # Emulate the underlying PT calculator, as before (bias parameters are not emulated).
                # The box comes from each parameter's `fd` limits, intersected with its prior.
                space = Space(pt, levels=emulator_levels, transforms=emulator_transforms)
                emulator = Emulator(pt, space, budget=emulator_budget)
                print(f"Training on {len(emulator.nodes())} nodes")
                emulator.train(checkpoint=checkpoint_filename, mpicomm=mpicomm)
                if mpicomm.rank == 0:
                    emulator.write(emulator_filename)
                mpicomm.Barrier()

            # Swap the emulated PT into the theory
            emulated_pt = emulator.to_calculator(calculator=pt)
            replace(theory, pt, emulated_pt)



print('All theories have been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED, proceeding without emulation')



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


all_params = get_params(likelihood)
# The emulated PT carries the cosmological parameters as they were at training time:
# re-apply the cosmology settings so that the priors set above are always the ones used
update_cosmo_params(all_params)

#Analytic marginalization over eft and nuisance parameters
for i in (tracers):
    for basename in ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']:
        param = all_params[f'{i}.{basename}']
        if param.varied:
            param.update(derived = 'marg')

    # theories[i]['ps'].params['b3'].update(derived = '32/315*({b1}-1)')
    # print("b1=",theories[i]['ps'].params['b1'].value,"b3=",theories[i]['ps'].params['b3'].value)

#Eft and nuisance parameters are namespaced per tracer (i.e. QSO.alpha0, QSO.alpha2, BGS.alpha0,...),
#through the theories' `tracers` argument
for param in all_params:
    if isinstance(param, Parameter):
        print(param,':',param.prior)

# Posterior = likelihood + priors (and the analytic marginalization over the 'marg' parameters)
posterior = Posterior(likelihood)


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--run_chains", action="store_true", help="To run a chain")
parser.add_argument("--plot_bestfit", action="store_true", help="Plot Bestfit")
parser.add_argument("--plot_chains", action="store_true", help="Plot Chains")
parser.add_argument("--test", action="store_true", help="testing")

args = parser.parse_args()




import matplotlib.pyplot as plt
from desilike.samples import MCSamples
from desilike import setup_logging
setup_logging()

def load_chain(chain_dir, burnin=0.3):
    # one file per chain, samples_<i>.h5
    chains = [MCSamples.read(fn).remove_burnin(burnin) for fn in sorted(Path(chain_dir).glob('samples_*.h5'))]
    if not chains:
        raise FileNotFoundError(f'no chains found in {chain_dir}')
    chain = MCSamples.concatenate(*chains)
    print(f'chain: {chain}')
    return chain


if args.test:
    pipe = build(posterior)
    print("logposterior: ",pipe())
    print(get_params(pipe).select(varied=True, derived=False).names())
    print("Test Successful!")



if args.plot_bestfit:
    chain = load_chain(chain_dir)
    pipe = build(posterior)
    names = get_params(pipe).select(varied=True, derived=False).names()
    pipe(**dict(zip(names, chain.argmax(params=names))))
    for tracer in tracers:
        observables[tracer]['ps'].plot(fn=f'test_bestfit_{tracer}.png',kw_save={'dpi':250})




if args.plot_chains:
    chain = load_chain(chain_dir)
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

#Run the sampler and save the chain
    from desilike import Sampler
    from desilike.samplers import MH, Emcee, AffineConditioner

    if not restart_chain:
        # The sampler resumes automatically from the files in chain_dir: remove them to start from 0 steps
        if mpicomm.rank == 0 and chain_dir.exists():
            for fn in [*chain_dir.glob('samples_*.h5'), *chain_dir.glob('rng_*.json'),
                       *chain_dir.glob('kernel_state_*.pkl'), *chain_dir.glob('checks.json')]:
                fn.unlink()
        mpicomm.Barrier()

    pipe = build(posterior)
    # GR_criteria is on R - 1; desilike's gelman_rubin threshold is on R
    if sampler == 'cobaya':
        # Metropolis-Hastings, proposal covariance learned during the first mh_adaptation_steps
        # (starting from the identity in units of each parameter's ref.std(), hence rescale=True)
        sampler = Sampler(pipe, kernel=MH(), nparallel=nchains, output_dir=chain_dir,
                          conditioner=AffineConditioner(rescale=True))
        samples = sampler.run(gelman_rubin=1. + GR_criteria, adaptation={'steps': mh_adaptation_steps})

    else:
        sampler = Sampler(pipe, kernel=Emcee(), nparallel=nchains, output_dir=chain_dir)
        samples = sampler.run(gelman_rubin=1. + GR_criteria)
