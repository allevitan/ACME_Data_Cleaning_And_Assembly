import numpy as np
from scipy import constants


def create_illumination(dps, norm='backward'):
    dataAve = dps.mean(0)
    pMask = np.fft.fftshift((dataAve > 0.01 * dataAve.max()))
    probe = np.sqrt(np.fft.fftshift(dataAve)) * pMask
    probe = np.fft.ifftshift(np.fft.ifftn(probe, norm=norm))
    return probe, pMask


def create_streak_mask(probe_mask):
    probe_mask_shifted = np.fft.fftshift(probe_mask)
    mask = np.ones_like(probe_mask)
    sh = mask.shape
    indices = np.where(probe_mask_shifted > 0)
    ymin = min(indices[0])
    ymax = max(indices[0])
    xmin = 0
    xmax = sh[1] // 2
    mask[ymin:ymax,xmin:xmax] = 0
    mask = probe_mask_shifted | mask
    mask = np.array(mask, dtype=int)
    return mask


def propnf(a, z, l):
    """
    propnf(a,z,l) where z and l are in pixel units
    """

    if a.ndim != 2:
        raise RuntimeError("A 2-dimensional wave front 'w' was expected")
    # if a.shape[0] != a.shape[1]:
    #    raise RunTimeError("Only square arrays are currently supported")

    n = len(a)
    ny, nx = a.shape
    qx, qy = np.meshgrid(np.linspace(-nx / 2, nx / 2, nx), np.linspace(-ny / 2, ny / 2, ny))
    q2 = np.fft.fftshift(qx ** 2 + qy ** 2)

    if n / np.sqrt(2.) < np.abs(z) * l:
        print("Warning: %.2f < %.2f: this calculation could fail." % (n / np.sqrt(2.), np.abs(z) * l))
        print("You could enlarge your array, or try the far field method: propff()")

    return np.fft.ifftn(np.fft.fftn(a) * np.exp(-2j * np.pi * (z / l) * (np.sqrt(1 - q2 * (l / n) ** 2) - 1)))


def defocus_illumination(probe, energy_joule, px_size_nm, ref_defocus_length_um, ref_defocus_energy_eV):
    probe_defocused = probe.copy()
    energy_eV = energy_joule / constants.e
    # l = 1240. / energy_eV
    l = constants.h * constants.c / energy_joule
    l_nm = l * 1e9
    defocus = ref_defocus_length_um * energy_eV / ref_defocus_energy_eV
    probe_defocused = propnf(probe_defocused, defocus * 1000. / px_size_nm, l_nm / px_size_nm)
    return probe_defocused


def calculate_real_space_px_size(energy_joule, d_sample_detector, npx_detector, d_px_detector):
    wavelength = constants.h * constants.c / energy_joule
    d_detector = npx_detector * d_px_detector
    alpha = np.arctan(d_detector / (2 * d_sample_detector))
    px_size = wavelength / (2 * np.sin(alpha))
    return px_size


def init_illumination(dps, metadata, config):
    energy_joule = metadata['energy']

    probe, probe_mask = create_illumination(dps, norm='ortho')

    distance = metadata['detector_distance']
    d_px_detector = metadata['x_pixel_size']
    npx_detector = metadata['output_frame_width']
    px_size_m = calculate_real_space_px_size(
        energy_joule,
        distance,
        npx_detector,
        d_px_detector
    )
    px_size_nm = px_size_m * 1e9

    probe = defocus_illumination(
        probe,
        energy_joule,
        px_size_nm,
        config['ref_defocus_length_um'],
        config['ref_defocus_energy_eV']
    )

    return probe, probe_mask
