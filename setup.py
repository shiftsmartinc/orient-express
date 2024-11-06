from setuptools import setup, find_packages

setup(
    name="orient_express",
    version="0.1.1",
    description="A single-module library for [describe functionality briefly]",
    author="Alex Zankevich",
    author_email="alex.zankevich@shiftsmart.com",
    url="https://github.com/shiftsmartinc/orient-express",  # Replace with your repository URL
    packages=find_packages(),  # Automatically finds all packages in the project
    install_requires=["google-cloud-aiplatform", "google-cloud-storage"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with the appropriate license
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",  # Specify Python versions compatible with your library
)
