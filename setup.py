from setuptools import setup, find_packages, Extension
import sys
import numpy
import os
import os.path as path
import multiprocessing

use_cython = True
force = False
profile = False
install_rates = True

if "--skip-cython" in sys.argv:
    use_cython = False
    del sys.argv[sys.argv.index("--skip-cython")]

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

if "--skip-rates-install" in sys.argv:
    install_rates = False
    del sys.argv[sys.argv.index("--skip-rates-install")]

source_paths = ['cherab', 'demos']
compilation_includes = [".", numpy.get_include()]
compilation_args = []
cython_directives = {
    'language_level': 3
}
setup_path = path.dirname(path.abspath(__file__))

if use_cython:

    from Cython.Build import cythonize

    # build .pyx extension list
    extensions = []
    for package in source_paths:
        for root, dirs, files in os.walk(path.join(setup_path, package)):
            for file in files:
                if path.splitext(file)[1] == ".pyx":
                    pyx_file = path.relpath(path.join(root, file), setup_path)
                    module = path.splitext(pyx_file)[0].replace("/", ".")
                    extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes, extra_compile_args=compilation_args),)

    if profile:
        cython_directives["profile"] = True

    # generate .c files from .pyx
    extensions = cythonize(extensions, nthreads=multiprocessing.cpu_count(), force=force, compiler_directives=cython_directives)

else:

    # build .c extension list
    extensions = []
    for package in source_paths:
        for root, dirs, files in os.walk(path.join(setup_path, package)):
            for file in files:
                if path.splitext(file)[1] == ".c":
                    c_file = path.relpath(path.join(root, file), setup_path)
                    module = path.splitext(c_file)[0].replace("/", ".")
                    extensions.append(Extension(module, [c_file], include_dirs=compilation_includes, extra_compile_args=compilation_args),)

# parse the package version number
with open(path.join(path.dirname(__file__), 'cherab/openadas/VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name="cherab-openadas",
    version=version,
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    description='Cherab spectroscopy framework: OpenADAS atomic data source package',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    install_requires=['cherab', 'numpy', 'cython>=0.28'],
    packages=find_packages(),
    include_package_data=True,
    zip_safe= False,
    ext_modules=extensions
)

# setup a rate repository with common rates
if install_rates:
    try:
        from cherab.openadas import repository
        repository.populate()
    except ImportError:
        pass

