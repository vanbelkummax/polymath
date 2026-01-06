from setuptools import setup, find_packages

setup(
    name="polymath",
    version="1.0.0",
    author="Max Van Belkum",
    author_email="max.van.belkum@vanderbilt.edu",
    description="A Polymathic Research Intelligence System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vanbelkummax/polymath",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "chromadb>=0.4.0",
        "psycopg2-binary>=2.9.0",
        "neo4j>=5.0.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pymupdf>=1.22.0",
        "pdfplumber>=0.9.0",
        "requests>=2.28.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "polymath=polymath_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
