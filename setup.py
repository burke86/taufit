import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="burke86", # Replace with your own username
    version="0.1.0",
    author="Colin J. Burke",
    author_email="colinjb2@illinois.edu",
    description="Efficient AGN light curve modeling using celerite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/burke86/taufit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)