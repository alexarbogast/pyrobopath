from setuptools import setup, find_packages

package_name = "pyrobopath"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    maintainer="Alex Arbogast",
    maintainer_email="arbogastaw@gmail.com",
    install_requires=["itertools", "networkx", "GcodeParser", "quaternion"],
)
