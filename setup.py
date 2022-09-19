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
        "PyQt5",
        "pyzmq",
        "jax",
    ],

    package_dir={"": "src"},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

