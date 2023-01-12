ACME Data Cleaning & Assembly
-----------------------------

"Outstanding service, fair prices"

This package includes functions for preprocessing the fastCCD data as it is
emitted by the pystxmcontrol program at beamline 7.0.1.2 of the ALS. This is
essentially raw fastCCD data, as emitted by the framegrabber, along with some
basic metadata.

The key function of this code is therefore to perform background subtraction
and stitching of multiple-exposure datasets. However, some additional
transformations are implemented at the same stage, for example the shear
present in the probe translations is corrected for here.

## Installation

Once the various dependencies (listed in setup.py) are installed, PCW can be installed via:

```console
$ pip install -e .
```

The "-e" flag for developer-mode installation is recommended so updates to the git repo can be immediately included in the installed program.

## Usage

This package contains two programs, one to process saved data stored in a
.stxm file, and one to listen to the raw data being emitted by pystxmcontrol
and assemble the data live, passing along a stream of synthesized frames and
saving out a .cxi file as the data comes in. Note that the live saving of
.cxi files has not yet been implemented. TODO!!

To process a saved .stxm file, run

```console
$ process_stxm_file <stxm_file_path>
```

The available options can be found by adding the `--help` flag. To run the live data analysis process, execute the following command:

```console
$ process_live_data
```

## Configuration

The port which the live data processing code will listen to, and the port on
which it will broadcast to, can be set in an optional configuration file. This
also contains the matrix used to correct the probe position shearing.

An example configuration file is found in "example_config.json". To override the default configuration, place a derived file called "config.json" in the main code folder, "src/acme_data_cleaning".

If the package is installed normally, you will need to set the configuration file before installing, and reinstall the package whenever you update the configuration. If the package is installed in developer mode, changes to the configuration file will be automatically propagated.



