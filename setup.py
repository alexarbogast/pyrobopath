from setuptools import setup, find_packages

package_name = "pyrobopath"

setup(
    name=package_name,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    maintainer="Alex Arbogast",
    maintainer_email="arbogastaw@gmail.com",
    install_requires=[
        "matplotlib",
        "networkx",
        "gcodeparser",
        "python-fcl",
        "numpy",
        "numpy-quaternion",
        "scipy",
        "colorama",
    ],
)
