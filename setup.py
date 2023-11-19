from distutils.core import setup

package_name = 'pyrobopath'

setup(
    name=package_name,
    version='0.1.1',
    packages=['pyrobopath'],
    package_dir={'':'src'}, 
    install_requires=[
        'itertools',
        'networkx',
    ]
)
