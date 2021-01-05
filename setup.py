#!/usr/bin/env python
"""Setup script for the zerospeech2021 Python package"""

import codecs
import setuptools
import setuptools.command.build_ext
import zerospeech2021


class build_ext(setuptools.command.build_ext.build_ext):
    def finalize_options(self):
        setuptools.command.build_ext.build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


setuptools.setup(
    # general description
    name='zerospeech2021',
    description="Evaluation and validation tools for ZeroSpeech2021",
    version=zerospeech2021.__version__,

    # python package dependencies
    setup_requires=['cython', 'numpy'],

    # include Python code
    packages=setuptools.find_packages(),

    # build cython extension
    ext_modules=[setuptools.Extension(
        'libri_light_dtw',
        sources=['zerospeech2021/phonetic_eval/ABX_src/dtw.pyx'],
        extra_compile_args=['-O3'])],

    # needed for cython/setuptools, see
    # http://docs.cython.org/en/latest/src/quickstart/build.html
    zip_safe=False,

    # the command-line scripts to export
    entry_points={
        'console_scripts': [
            'zerospeech2021-validate = zerospeech2021.cli.validate:validate',
            'zerospeech2021-evaluate = zerospeech2021.cli.evaluate:evaluate']},

    # metadata
    author='CoML team',
    author_email='zerospeech2021@gmail.com',
    license='GPL3',
    url='https://zerospeech.com/2021',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
)
