import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="acme_data_cleaning",
    version="0.1.0",
    python_requires='>3.7', # recommended minimum version for pytorch
    author="Abe Levitan",
    author_email="alevitan@mit.edu",
    description="Tool to process raw data at COSMIC into .cxi files ready to be used for reconstructions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/allevitan/ACME_Data_Cleaning",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=2.0", # 2.0 has better colormaps which are used by default
        "h5py",
        "pyzmq",
        "torch>=1.9.0",
    ],
    entry_points={
        'console_scripts': [
            'process_stxm_file=acme_data_cleaning.process_stxm_file:main',
            'find_center=acme_data_cleaning.find_center:main',
            'process_live_data=acme_data_cleaning.process_live_data:main',
            'simulate_zmq_from_stxm_file=acme_data_cleaning.simulate_zmq_from_stxm_file:main',
        ]
    },
    package_data={"acme_data_cleaning": ["*.json", "*.h5"]},
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

