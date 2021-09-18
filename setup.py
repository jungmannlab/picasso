from setuptools import setup


long_description = """
A collection of tools for painting super-resolution images (processing of Single-molecule localization microscopy (SMLM) data).
Taylored for DNA-PAINT, supercharged with Numba.

Features include:    
    - Design: Design rectangular DNA origami
    - Simulate: Simulate DNA-PAINT image acquistions
    - Localize: Localize allows performing super-resolution reconstruction of image stacks.
    - Render: Rendering of the super-resolution images and post-processing
    - Average: Particle averaging
"""

with open("requirements.txt") as requirements_file:
    requirements = [line for line in requirements_file]

setup(
    name="picassosr",
    version="0.3.6",
    author="Joerg Schnitzbauer, Maximilian T. Strauss",
    author_email=(
        "joschnitzbauer@gmail.com, straussmaximilian@gmail.com"
    ),
    url="https://github.com/jungmannlab/picasso",
    long_description = long_description,
    packages=["picasso", "picasso.gui"],
    entry_points={
        "console_scripts": ["picasso=picasso.__main__:main"],
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "picasso": [
            "gui/icons/*.ico",
            "config_template.yaml",
        ]
    },
)
