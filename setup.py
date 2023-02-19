import os

import pkg_resources
from setuptools import setup, find_packages


def read_version(fname="causal_verifier/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


setup(
    name="causal_verifier",
    py_modules=["causal_verifier"],
    version=read_version(),
    description="Verifier for interpretability hypothesis",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.7",
    author="Pranav Gade <pranavgade20@gmail.com>",
    url="https://github.com/pranavgade20/causal-verifier",
    license="MIT",
    packages=find_packages(include=["causal_verifier"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
