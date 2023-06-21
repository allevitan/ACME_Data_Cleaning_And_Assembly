from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc7523 import PrivateKeyJWT


def ptycho(
        cxiname,
        path_client_id="/global/software/ptycholive-dev/nersc_api_keys/nersc_clientid.txt",
        path_private_key="/global/software/ptycholive-dev/nersc_api_keys/nersc_priv_key.pem",
):
    path_job_script_at_nersc_storage = "~/projects/nersc_cosmic_reco/ptycho/job_ptycho.sh"
    submit_job(
        path_job_script_at_nersc_storage=path_job_script_at_nersc_storage,
        args=cxiname,
        path_client_id=path_client_id,
        path_private_key=path_private_key
    )


def ptychocam(
        cxiname,
        n_iter=500,
        period_illu_refine=0,
        period_bg_refine=0,
        use_illu_mask=False,
        path_client_id="/global/software/ptycholive-dev/nersc_api_keys/nersc_clientid.txt",
        path_private_key="/global/software/ptycholive-dev/nersc_api_keys/nersc_priv_key.pem",
):
    path_job_script_at_nersc_storage = "~/projects/nersc_cosmic_reco/ptychocam/job_ptychocam.sh"
    args = "-i,"
    args += f"{n_iter},"

    if period_illu_refine != 0:
        args += "-r,"
        args += f"{period_illu_refine},"
    if period_bg_refine != 0:
        args += "-T,"
        args += f"{period_bg_refine},"
    if use_illu_mask:
        args += "-M,"

    args += cxiname

    submit_job(
        path_job_script_at_nersc_storage=path_job_script_at_nersc_storage,
        args=cxiname,
        path_client_id=path_client_id,
        path_private_key=path_private_key
    )


def cdtools(
    cxiname,
    run_split_reconstructions=False,
    n_modes=1,
    oversampling_factor=1,
    simulate_probe_translation=True,
    n_init_rounds=1,
    n_init_iter=50,
    n_final_iter=50,
    translation_randomization=0,
    probe_initialization=None,
    init_background=False,
    probe_support_radius=None,
    path_client_id="/global/software/ptycholive-dev/nersc_api_keys/nersc_clientid.txt",
    path_private_key="/global/software/ptycholive-dev/nersc_api_keys/nersc_priv_key.pem",
):
    path_job_script_at_nersc_storage = "~/projects/nersc_cosmic_reco/cdtools/job_cdtools.sh"

    args = f"{cxiname},"
    args += f"{run_split_reconstructions},"
    args += f"{n_modes},"
    args += f"{oversampling_factor},"
    args += f"{simulate_probe_translation},"
    args += f"{n_init_rounds},"
    args += f"{n_init_iter},"
    args += f"{n_final_iter},"
    args += f"{translation_randomization},"
    args += f"{probe_initialization},"
    args += f"{init_background},"
    args += f"{probe_support_radius}"

    submit_job(
        path_job_script_at_nersc_storage=path_job_script_at_nersc_storage,
        args=args,
        path_client_id=path_client_id,
        path_private_key=path_private_key
    )


def submit_job(
        path_job_script_at_nersc_storage,
        args,
        path_client_id,
        path_private_key,
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
        "args": args
    }

    r = session.post(
        api_call_url,
        data=data
    )

    print(r.json())
