from distutils.core import setup

setup(
    name = 'picassso',
    version = '0.1',
    author = 'Joerg Schnitzbauer, Maximilian Strauss',
    author_email = 'joschnitzbauer@gmail.com, mstrauss@biochem.mpg.de',
    url = 'https://gitlab.com/jungmannlab/picasso',
    packages = ['picasso', 'picasso.gui'],
    classifiers = ["Programming Language :: Python",
                   "Programming Language :: Python :: 3.5",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
    entry_points = {'console_scripts': ['picasso = picasso.main']}
)
