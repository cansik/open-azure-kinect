from pathlib import Path

from setuptools import find_packages, setup

NAME = "open-azure-kinect"

required_packages = find_packages(exclude=["*test.*", "*test", "playground", "examples"])

with open("requirements.txt") as f:
    required = [line for line in f.read().splitlines() if not line.startswith("-")]

# read readme
current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text()

setup(
    name=NAME,
    version="0.1.0a2",
    packages=required_packages,
    url="https://github.com/cansik/open-azure-kinect",
    license="MIT License",
    author="Florian Bruggisser",
    author_email="github@broox.ch",
    description="Open Azure Kinect recordings without the Azure Kinect SDK.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required
)
