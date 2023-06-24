from reconstruction_flow import cdtools_from_cxi
from prefect.deployments import Deployment

infrastructure = {
    'working_dir': '/global/software/ptycholive-dev/cosmic_prefect/ptychocam/cdtools'
}

parameters = {
    'run_split_reconstructions': False,
    'n_modes': 1,
    'oversampling_factor': 1,
    'propagation_distance': 50e-6,
    'simulate_probe_translation': True,
    'n_init_rounds': 1,
    'n_init_iter': 50,
    'n_final_iter': 50,
    'translation_randomization': 0,
    'probe_initialization': None,
    'init_background': False,
    'probe_support_radius': None,
}

deployment = Deployment.build_from_flow(
    flow=cdtools_from_cxi,
    path='/global/software/ptycholive-dev/cosmic_prefect/cdtools',
    parameters=parameters,
    name="cdtools_from_cxi",
    version=1,
    work_queue_name="cdtools"
)

deployment.apply()
