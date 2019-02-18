from setuptools import setup

setup(
    name="picasso",
    version="0.2.7",
    author="Joerg Schnitzbauer, Maximilian T. Strauss",
    author_email=(
        "joschnitzbauer (at) gmail.com, straussmaximilian (at) gmail.com"
    ),
    url="https://github.com/jungmannlab/picasso",
    packages=["picasso", "picasso.gui"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "picasso": [
            "gui/icons/*.ico",
            "config_template.yaml",
            "base_sequences.csv",
            "paint_sequences.csv",
        ]
    },
)
