from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc7523 import PrivateKeyJWT


default_path_client_id = "/global/software/ptycholive-dev/nersc_api_keys/nersc_clientid.txt"
default_path_private_key = "/global/software/ptycholive-dev/nersc_api_keys/nersc_priv_key.pem"


def ptycho(
        cxiname,
        path_client_id=default_path_client_id,
        path_private_key=default_path_private_key,
):
    script_string = f"""#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=als_g

~/cosmic_reconstruction_at_nersc/c_ptycho/ptycho_reconstruction.sh {cxiname}
"""

    submit_job_script_as_string(
        script_string=script_string,
        path_client_id=path_client_id,
        path_private_key=path_private_key
    )


def ptychocam(
        cxiname,
        n_iter=500,
        period_illu_refine=0,
        period_bg_refine=0,
        use_illu_mask=False,
        path_client_id=default_path_client_id,
        path_private_key=default_path_private_key,
):
    args = "-i "
    args += f"{n_iter} "

    if period_illu_refine != 0:
        args += "-r "
        args += f"{period_illu_refine} "
    if period_bg_refine != 0:
        args += "-T "
        args += f"{period_bg_refine} "
    if use_illu_mask:
        args += "-M "

    args += cxiname

    script_string = f"""#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=als_g

~/cosmic_reconstruction_at_nersc/c_ptychocam/ptychocam_reconstruction.sh {args}
"""

    submit_job_script_as_string(
        script_string=script_string,
        path_client_id=path_client_id,
        path_private_key=path_private_key
    )


def cdtools(
    cxiname,
    run_split_reconstructions=False,
    n_modes=1,
    oversampling_factor=1,
    propagation_distance=50 * 1e-6,
    simulate_probe_translation=True,
    n_init_rounds=1,
    n_init_iter=50,
    n_final_iter=50,
    translation_randomization=0,
    probe_initialization=None,
    init_background=False,
    probe_support_radius=None,
    path_client_id=default_path_client_id,
    path_private_key=default_path_private_key,
):
    args = f"{cxiname} "
    args += f"{run_split_reconstructions} "
    args += f"{n_modes} "
    args += f"{oversampling_factor} "
    args += f"{propagation_distance} "
    args += f"{simulate_probe_translation} "
    args += f"{n_init_rounds} "
    args += f"{n_init_iter} "
    args += f"{n_final_iter} "
    args += f"{translation_randomization} "
    args += f"{probe_initialization} "
    args += f"{init_background} "
    args += f"{probe_support_radius}"

    script_string = f"""#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=als_g

~/cosmic_reconstruction_at_nersc/c_cdtools/cdtools_reconstruction.sh {args}
"""

    submit_job_script_as_string(
        script_string=script_string,
        path_client_id=path_client_id,
        path_private_key=path_private_key
    )


def submit_job_script_as_string(
        script_string,
        path_client_id=default_path_client_id,
        path_private_key=default_path_private_key,
):
    token_url = "https://oidc.nersc.gov/c2id/token"

    with open(path_client_id, 'r') as f:
        client_id = f.read()

    with open(path_private_key, 'r') as f:
        private_key = f.read()

    session = OAuth2Session(
        client_id,
        private_key,
        PrivateKeyJWT(token_url),
        grant_type="client_credentials",
        token_endpoint=token_url
    )

    token = session.fetch_token()

    nersc_api_url = "https://api.nersc.gov/api/v1.2"
    api_call = "compute/jobs"
    system = "perlmutter"
    api_call_url = f"{nersc_api_url}/{api_call}/{system}"

    data = {
        "job": script_string,
        "isPath": False,
    }

    r = session.post(
        api_call_url,
        data=data
    )

    print(r.json())


def submit_job_script_on_nersc_storage(
        path_job_script_at_nersc_storage,
        args=None,
        path_client_id=default_path_client_id,
        path_private_key=default_path_private_key,
):
    token_url = "https://oidc.nersc.gov/c2id/token"

    with open(path_client_id, 'r') as f:
        client_id = f.read()

    with open(path_private_key, 'r') as f:
        private_key = f.read()

    session = OAuth2Session(
        client_id,
        private_key,
        PrivateKeyJWT(token_url),
        grant_type="client_credentials",
        token_endpoint=token_url
    )

    token = session.fetch_token()

    nersc_api_url = "https://api.nersc.gov/api/v1.2"
    api_call = "compute/jobs"
    system = "perlmutter"
    api_call_url = f"{nersc_api_url}/{api_call}/{system}"

    data = {
        "job": path_job_script_at_nersc_storage,
        "isPath": True,
    }

    if args is not None:
        data["args"] = args

    r = session.post(
        api_call_url,
        data=data
    )

    print(r.json())
