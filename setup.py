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
    packages=["stochpy", "stochpy.dynamics", "stochpy.rare", "stochpy.io"],
    install_requires=["matplotlib", "numpy", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
