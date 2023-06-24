from prefect.deployments import run_deployment


if __name__ == "__main__":
    parameters = {
        'cxipath': '/global/scratch/silvio/test/all/NS_230216037_ccdframes_0_0.cxi',
    }

    run_deployment(name='ptychocam_from_cxi/ptychocam_from_cxi',
                   parameters=parameters,
                   timeout=0)
