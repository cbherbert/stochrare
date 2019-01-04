import setuptools

with open("README.md", "r") as fh:
    readme = fh.read()

setuptools.setup(
    name="stochpy",
    version="0.0.1",
    author="Corentin Herbert",
    author_email="corentin.herbert@ens-lyon.fr",
    # add url
    description="A package to simulate simple stochastic processes",
    long_description=readme,
    packages=["stochpy", "stochpy.dynamics"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
