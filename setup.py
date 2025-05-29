from setuptools import setup, find_packages

setup(
    name="nnpixsi",
    version="0.1",
    packages=find_packages(),
    install_requires=["click","numpy", "matplotlib", "scipy", "torch", "torch_geometric"],
    description="ML based signal processing tools for LarPix ND DUNE",
    entry_points = dict(
        console_scripts = [
            'nnpixsi = nnpixsi.__main__:main',
        ]
    ),
)
