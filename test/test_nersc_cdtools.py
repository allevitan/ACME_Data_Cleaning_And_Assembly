import os
import h5py
from scipy import constants
from acme_data_cleaning import nersc


if __name__ == "__main__":
    cxiname = "NS_230216033_ccdframes_0_0.cxi"
    cxipath = os.path.join('/global/scratch/silvio/test/all', cxiname)
    with h5py.File(cxipath) as f:
        energy_J = f['entry_1/instrument_1/source_1/energy'][()]
        energy_eV = energy_J / constants.e
        out_of_focus_distance_um = 15. * energy_eV / 700.
        out_of_focus_distance_m = out_of_focus_distance_um * 1e-6

    nersc.cdtools(
        cxiname=cxiname,
        propagation_distance=out_of_focus_distance_m,
    )
