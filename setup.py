from importlib.metadata import entry_points
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evontogeny",
    version="2025.08",
    author="Gabe DuBose",
    author_email="james.g.dubose@gmail.com",
    description="Evolutionary dynamics of ontogenetic programs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabe-dubose/geomcp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    install_requires = ['pandas', 'numpy']
)