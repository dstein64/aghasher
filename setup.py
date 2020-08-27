import io
import os
from setuptools import setup

version_txt = os.path.join(os.path.dirname(__file__), 'aghasher', 'version.txt')
with open(version_txt, 'r') as f:
    version = f.read().strip()

with io.open('README.md', encoding='utf8') as f:
    long_description = f.read()

setup(
    author='Daniel Steinberg',
    author_email='ds@dannyadam.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    description='An implementation of Anchor Graph Hashing',
    install_requires=['numpy', 'scipy'],
    keywords=['anchor-graph-hashing', 'hashing', 'locality-sensitive-hashing', 'machine-learning'],
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    name='aghasher',
    package_data={'aghasher': ['version.txt']},
    packages=['aghasher'],
    url='https://github.com/dstein64/aghasher',
    version=version,
)
