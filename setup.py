from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='SpectralExtractor',
      version='0.1',
      description='Python Tool for extracting 1D spectrum from 2D image',
      long_description = readme(),
      classifiers=[
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      keywords='Spectral Extraction Astronomy Data Reduction',
      url='https://github.com/indiajoe/SpectralExtractor',
      author='Joe Ninan',
      author_email='indiajoe@gmail.com',
      license='LGPLv3+',
      packages=['SpectralExtractor'],
      entry_points = {
          'console_scripts': ['spectral_extractor=SpectralExtractor.spectral_extractor:main'],
      },
      install_requires=[
          'numpy',
          'astropy',
          'scikit-image',
          'matplotlib',
          'scipy',
          'ccdproc',
          'RVEstimator',
          'WavelengthCalibrationTool'
      ],
      include_package_data=True,
      zip_safe=False)
