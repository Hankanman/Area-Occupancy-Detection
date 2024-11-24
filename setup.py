"""Setup module for Room Occupancy Detection."""
from pathlib import Path
from setuptools import setup, find_packages

PROJECT_DIR = Path(__file__).parent.resolve()

README_FILE = PROJECT_DIR / "README.md"
LONG_DESCRIPTION = README_FILE.read_text(encoding="utf-8")

REQUIRES = [
    "homeassistant>=2024.1.0",
    "numpy>=1.24.0",
    "voluptuous>=0.13.1",
]

setup(
    name="room_occupancy",
    version="1.0.0",
    description="Home Assistant integration for intelligent room occupancy detection",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ha-room-occupancy",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=REQUIRES,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Home Automation",
    ],
)
