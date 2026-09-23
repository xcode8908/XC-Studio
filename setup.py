from setuptools import setup, find_packages

setup(
    name="x-studio",
    version="0.1.0",
    description="X-Studio: Cloud-Accelerated Local AI Video Production Studio",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "jsonschema>=4.20",
        "python-dotenv>=1.0",
        "Pillow>=10.0",
        "requests>=2.31",
        "google-genai>=1.0.0",
        "openai>=2.44.0",
    ],
    entry_points={
        "console_scripts": [
            "x-studio=scripts.run_x_studio:main",
        ],
    },
)
