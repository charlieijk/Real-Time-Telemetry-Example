from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="solar-regatta",
    version="0.1.0",
    description="A Python package for analyzing and visualizing solar boat race telemetry data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Charlie Cullen",
    author_email="charliewcullen@gmail.com",
    url="https://github.com/pixelclubsf/Real-Time-Telemetry-Example",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="solar boat telemetry analysis visualization VESC",
    project_urls={
        "Bug Reports": "https://github.com/pixelclubsf/Real-Time-Telemetry-Example/issues",
        "Source": "https://github.com/pixelclubsf/Real-Time-Telemetry-Example",
    },
    entry_points={
        "console_scripts": [
            "solar-regatta=solar_regatta.cli:main",
        ],
    },
)
