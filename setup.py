"""本地 Matterport Mask R-CNN fork 的 package metadata。"""
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent


def read_requirements(path):
    requirements = []
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        requirements.append(line)
    return requirements


setup(
    name="mask-rcnn-road",
    version="0.1.0",
    description="使用 Mask R-CNN 進行道路場景 instance segmentation。",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["mrcnn"],
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.6,<3.8",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="mask r-cnn instance segmentation road pothole lane detection",
)
