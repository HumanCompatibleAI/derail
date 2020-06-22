"""setup.py for dr_seals paper project."""

from setuptools import find_packages, setup
import src.dr_seals  # pytype: disable=import-error

setup(
    name="dr_seals",
    version=src.dr_seals.__version__,
    description=("Code for DERAIL: Diagnostic Environments for Reward And Imitation Learning"),
    author='Center for Human-Compatible AI',
    author_email='pedrofreirex@gmail.com',
    python_requires=">=3.7.0,<3.8",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "gym",
        "tensorflow>=1.15.0,<2.0.0",
        "stable-baselines[mpi] @ git+https://github.com/pedrofreire/stable-baselines.git",
        "rllab @ git+https://github.com/pedrofreire/rllab.git",
        "inverse_rl @ git+https://github.com/pedrofreire/inverse_rl.git",
        "imitation @ git+https://github.com/pedrofreire/imitation@rollout-fix",
    ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        # recommended packages for development
        "dev": ["ipdb", "jupyter", *TESTS_REQUIRE],
        "test": TESTS_REQUIRE,
    },
    url="https://github.com/HumanCompatibleAI/dr-seals-paper",
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)