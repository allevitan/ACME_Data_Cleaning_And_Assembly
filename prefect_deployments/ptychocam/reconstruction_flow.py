import subprocess
from prefect import flow


@flow(name='ptychocam_from_cxi', log_prints=True)
def ptychocam_from_cxi(
    cxipath: str,
    n_gpus: int = 1,
    n_iterations: int = 2000,
    period_illu_refine: int = 0,
    period_bg_refine: int = 0,
    use_illu_mask: bool = False,
):
    cmd = f'mpirun -n {n_gpus} '\
          'python -m ptychocam.bin.ptycho '\
          f'-i {n_iterations} '\
          f'-r {period_illu_refine} '\
          f'-T {period_bg_refine} '

    if use_illu_mask:
        cmd += ' -M '

    cmd += cxipath

    subprocess.Popen(cmd, shell=True).wait()

    print("Reconstruction done.")


if __name__ == '__main__':
    cxipath = r'/global/scratch/silvio/test/all/NS_230216033_ccdframes_0_0.cxi'
    n_gpus = 4
    n_iterations = 1000
    period_illu_refine = 1
    period_bg_refine = 1
    use_illu_mask = True

    ptychocam_from_cxi(
        cxipath=cxipath,
        n_gpus=n_gpus,
        n_iterations=n_iterations,
        period_illu_refine=period_illu_refine,
        period_bg_refine=period_bg_refine,
        use_illu_mask=use_illu_mask,
    )
