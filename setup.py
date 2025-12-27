from pathlib import Path
import setuptools


def readme() -> str:
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def parse_requirements(requirements_file: str) -> list[str]:
    requirements: list[str] = []
    for line in Path(requirements_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


INSTALL_REQUIRES = parse_requirements("requirements.txt")

extras_require = {
    "ml": [
        "scikit-learn>=1.5",
    ],
    "viz": [
        "matplotlib>=3.9",
        "seaborn>=0.13",
    ],
    "ml-tf": [
        "tensorflow>=2.17; python_version < '3.13'",
    ],
}
extras_require["all"] = sorted({pkg for group in extras_require.values() for pkg in group})

setuptools.setup(
    name="fcmpy",
    version="0.0.1",
    author="Samvel Mkhitaryan, Philippe J. Giabbanelli, Maciej Wozniak, Nanne K. de Vries, Rik Crutzen",
    author_email="mkhitarian.samvel@gmail.com",
    description="Fuzzy Cognitive Maps for Behavior Change Interventions and Evaluation",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SamvelMK/FcmBci.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=INSTALL_REQUIRES,
    extras_require=extras_require,
)