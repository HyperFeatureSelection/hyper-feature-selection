from setuptools import setup, find_packages
import os
import requirements as reqs

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    requirements = [r.line for r in reqs.parse(f)]

with open(os.path.join(here, "requirements_dev.txt")) as f:
    test_requirements = [r.line for r in reqs.parse(f)]


setup(
    name="hyper_feature_selection",
    version="0.1.0",
    description="A Python Package to select optimal feature in Machine Learning Models",
    url="https://github.com/HyperFeatureSelection/hyper-feature-selection",
    license="MIT",
    install_requires=requirements,
    tests_require=test_requirements,
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
