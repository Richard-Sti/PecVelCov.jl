from setuptools import find_packages, setup  # noqa

BUILD_REQ = ["numpy"]
INSTALL_REQ = BUILD_REQ
INSTALL_REQ += ["julia"
                ]


# TODO  EDIT ALL THIS
setup(
    name="pecvelcov",
    version="0.1",
    description="Covariance matrix of radial peculiar velocities",
    url="https://github.com/Richard-Sti/PecVelCov.jl",
    author="Richard Stiskalek",
    author_email="richard.stiskalek@protonmail.com",
    license="GPL-3.0",
    packages=find_packages(),
    python_requires=">=3.10",
    build_requires=BUILD_REQ,
    setup_requires=BUILD_REQ,
    install_requires=INSTALL_REQ,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9"
        ]
)
