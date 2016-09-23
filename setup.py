from setuptools import setup

setup(
    name = 'picasso',
    version = '0.1',
    author = 'Joerg Schnitzbauer',
    author_email = 'joschnitzbauer@gmail.com',
    url = 'https://gitlab.com/jungmannlab/picasso',
    packages = ['picasso', 'picasso.gui'],
    classifiers = ["Programming Language :: Python",
                   "Programming Language :: Python :: 3.5",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
    entry_points = {'console_scripts': ['picasso = picasso.__main__:main']},
    package_data = {'picasso': ['gui/icons/*.ico', 'config_template.yaml', 'base_sequences.csv']}
)
