from prefect.deployments import run_deployment


if __name__ == '__main__':
    parameters = {
        'path': '/global/scratch/silvio/test/all/NS_230216033_ccdframes_0_0.cxi',
        "run_split_reconstructions": False,
        "n_modes": 1,
        "oversampling_factor": 1,
        "propagation_distance": 18.245e-6,
        "simulate_probe_translation": True,
        "n_init_rounds": 1,
        "n_init_iter": 50,
        "n_final_iter": 50,
        "translation_randomization": 0,
        "probe_initialization": None,
        "init_background": False,
        "probe_support_radius": None,
    }

    run_deployment(
        name='cdtools_from_cxi/cdtools_from_cxi',
        parameters=parameters,
        timeout=0
    )
