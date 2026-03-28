from setuptools import setup, find_packages

setup(
    name="llm-perf",
    version="0.1.0",
    description="LLM Training and Inference Performance Evaluator",
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "llm-perf=llm_perf.cli.main:main",
        ],
    },
    install_requires=[
        # Core dependencies (most are stdlib)
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "mypy>=0.950",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
