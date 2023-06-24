from prefect import flow, task
import cdtools
from cdtools.tools import image_processing as ip
import torch as t
import numpy as np
from scipy import io
from copy import deepcopy
import argparse


@flow(name='cdtools_from_cxi',
      log_prints=True)
def cdtools_from_cxi(path: str,
                     run_split_reconstructions: bool = False,
                     n_modes: int = 3,
                     oversampling_factor: int = 1,
                     propagation_distance: float = 0,
                     simulate_probe_translation: bool = False,
                     n_init_rounds: int = 3,
                     n_init_iter: int = 50,
                     n_final_iter: int = 50,
                     translation_randomization: int = 0,
                     probe_initialization: str = None,
                     init_background: bool = False,
                     probe_support_radius: int = None):
    args = {
        'n_modes': n_modes,
        'oversampling_factor': oversampling_factor,
        'propagation_distance': propagation_distance,
        'simulate_probe_translation': simulate_probe_translation,
        'n_init_rounds': n_init_rounds,
        'n_init_iter': n_init_iter,
        'n_final_iter': n_final_iter,
        'translation_randomization': translation_randomization,
        'probe_support_radius': probe_support_radius,
        'probe_initialization': probe_initialization,
        'init_background': init_background
    }

    dataset = load_cxi_file(path)
    dataset.translations *= -1  # Some issue with the saved translations

    if not run_split_reconstructions:
        print('INFO not splitting reconstructions')

        model = make_model_from_dataset(dataset, args)

        dataset.get_as(device='cuda')
        model.to(device='cuda')

        run_initial_reconstruction(model, dataset, args)
        run_final_reconstruction(model, dataset, args)

        savefile = '.'.join(path.split('.')[:-1]) + '_full.mat'
        save_results(savefile, model, dataset)

    else:
        print('INFO splitting reconstructions')

        dataset_1, dataset_2 = split_dataset(dataset)
        datasets = [dataset, dataset_1, dataset_2]
        plans = ['full', 'half_1', 'half_2']
        for plan, dataset in zip(plans, datasets):
            print('INFO working on plan %s' % plan)
            model = make_model_from_dataset(dataset, args)

            dataset.get_as(device='cuda')
            model.to(device='cuda')

            run_initial_reconstruction(model, dataset, args)
            run_final_reconstruction(model, dataset, args)

            savefile = '.'.join(path.split('.')[:-1]) + ('_%s.mat' % plan)
            save_results(savefile, model, dataset)

    return savefile


@task
def split_dataset(dataset):
    random_selection = np.random.rand(len(dataset)) > 0.5
    for i in range(len(dataset) - 2):
        if random_selection[i] and random_selection[i + 1] \
                and random_selection[i + 2]:
            random_selection[i + 1] = False
        if ~random_selection[i] and ~random_selection[i + 1] \
                and ~random_selection[i + 2]:
            random_selection[i + 1] = True

    dataset_1 = deepcopy(dataset)
    dataset_1.translations = dataset.translations[random_selection]
    dataset_1.patterns = dataset.patterns[random_selection]
    dataset_2 = deepcopy(dataset)
    dataset_2.translations = dataset.translations[~random_selection]
    dataset_2.patterns = dataset.patterns[~random_selection]
    if hasattr(dataset, 'intensities') and dataset.intensities is not None:
        dataset_1.intensities = dataset.intensities[random_selection]
        dataset_2.intensities = dataset.intensities[~random_selection]

    return dataset_1, dataset_2


@task
def load_cxi_file(path):
    return cdtools.datasets.Ptycho2DDataset.from_cxi(path)


@task
def make_model_from_dataset(dataset, args):
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset, n_modes=args['n_modes'],
        oversampling=args['oversampling_factor'],
        propagation_distance=args['propagation_distance'],
        simulate_probe_translation=args['simulate_probe_translation'],
        probe_support_radius=args['probe_support_radius'],
        translation_scale=1,
        units='um')

    if args['probe_initialization'] is not None:
        data = io.loadmat(args['probe_initialization'])  # '')
        model.probe.data = t.as_tensor(data['probe']) / model.probe_norm
        if args['init_background'] is True:
            model.background.data = t.sqrt(t.as_tensor(data['background']))

    model.translation_offsets.data = t.rand_like(model.translation_offsets.data) * args['translation_randomization']
    return model


def center_probe(probe):
    # Make sure we dont screw with the input probe
    probe = t.clone(probe)
    for i in range(4):

        # Empirically, 4 iterations is repeatable to subpixel accuracy
        probe_abs_sq = t.sum(t.abs(probe) ** 2, axis=0)

        centroid = ip.centroid(probe_abs_sq)
        for i in range(probe.shape[0]):
            probe[i] = ip.sinc_subpixel_shift(probe[i],
                                              (-centroid[0] + probe.shape[-2] / 2,
                                               -centroid[1] + probe.shape[-1] / 2))

    return probe


@task
def run_initial_reconstruction(model, dataset, args):
    model.weights.requires_grad = False

    model.translation_offsets.requires_grad = False
    for i in range(max(0, args['n_init_rounds'] - 1)):
        print("INFO starting probe optimization round " + str(i))
        for loss in model.Adam_optimize(args['n_init_iter'], dataset, lr=0.05,
                                        batch_size=10,
                                        schedule=False):
            print("INFO " + model.report())

        model.probe.data = center_probe(model.probe.data.cpu()).to(device='cuda')
        model.probe.data *= model.probe_support
        model.probe.requires_grad = False
        model.obj.data[:] = 1

        for loss in model.Adam_optimize(3, dataset, lr=0.02, batch_size=25, schedule=False):
            print("INFO " + model.report())

        model.probe.requires_grad = True
    model.translation_offsets.requires_grad = False
    print("INFO starting final probe optimization round")
    if args['translation_randomization'] != 0:
        model.translation_offsets.data *= 0

    for loss in model.Adam_optimize(args['n_init_iter'],
                                    dataset, lr=0.05, batch_size=25,
                                    schedule=False):
        print("INFO " + model.report())

    model.weights.requires_grad = True


@task
def run_final_reconstruction(model, dataset, args):
    for loss in model.Adam_optimize(args['n_final_iter'], dataset,
                                    lr=0.005, batch_size=200,
                                    schedule=True):
        print("INFO " + model.report())

    model.tidy_probes()


@task
def save_results(save_filename, model, dataset):
    results = model.save_results(dataset)
    results['wavelength'] = model.wavelength.cpu().numpy()
    results['oversampling'] = np.array(model.oversampling)
    print(save_filename)
    io.savemat(save_filename, results)
