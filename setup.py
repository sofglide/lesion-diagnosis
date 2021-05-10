import re
import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


with open("src/version.py", "r") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)


setuptools.setup(
    name="src",
    version=version,
    author="Sofiane Soussi",
    author_email="sofiane.soussi@gmail.com",
    description="Skin legion diagnosis experimentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sofglide/lesion-diagnosis",
    python_requires=">=3.8",
    packages=["src"],
    include_package_data=True,
    install_requires=[
        "pandas>=0.23",
        "torch>=0.4",
        "torchvision>=0.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
