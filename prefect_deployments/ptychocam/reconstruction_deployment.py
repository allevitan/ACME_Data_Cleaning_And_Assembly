from reconstruction_flow import ptychocam_from_cxi
from prefect.deployments import Deployment

infrastructure = {
    'working_dir': '/global/software/ptycholive-dev/cosmic_prefect/ptychocam'
}

parameters = {
    'n_gpus': 4,
    'n_iterations': 2000,
    'period_illu_refine': 1,
    'period_bg_refine': 1,
    'use_illu_mask': True,
}

deployment = Deployment.build_from_flow(
    flow=ptychocam_from_cxi,
    path='/global/software/ptycholive-dev/cosmic_prefect/ptychocam',
    parameters=parameters,
    name="ptychocam_from_cxi",
    version=1,
    work_queue_name="ptychocam"
)

deployment.apply()
