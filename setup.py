from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='SpectrumExtractor',
      version='0.2',
      description='Python Tool for extracting 1D spectrum from 2D image',
      long_description = readme(),
      classifiers=[
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Programming Language :: Python :: 3.6+',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      keywords='Spectrum Extraction Astronomy Data Reduction',
      url='https://github.com/indiajoe/SpectrumExtractor',
      author='Joe Ninan',
      author_email='indiajoe@gmail.com',
      license='LGPLv3+',
      packages=['SpectrumExtractor'],
      entry_points = {
          'console_scripts': ['spectrum_extractor=SpectrumExtractor.spectrum_extractor:main'],
      },
      install_requires=[
          'numpy',
          'astropy',
          'scikit-image',
          'matplotlib',
          'scipy',
          'ccdproc',
          'WavelengthCalibrationTool @ git+https://github.com/indiajoe/WavelengthCalibrationTool.git@master'
      ],
      extras_require={
          'Rectification':['RVEstimator']
          },
      include_package_data=True,
      zip_safe=False)
