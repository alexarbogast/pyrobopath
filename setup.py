from distutils.core import setup

package_name = 'nav2_gps_waypoint_follower_demo'

setup(
    name=package_name,
    version='0.1.0',
    scripts=[],
    packages=['pyrobopath'],
    package_dir={'':'src'}, 
    install_requires=[
        'itertools',
        'networkx',
    ]
)

