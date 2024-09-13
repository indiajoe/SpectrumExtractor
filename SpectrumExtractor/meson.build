
add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# laplace_source = custom_target('laplace_filtermodule.c',
#   input : ['laplace_filter.f95'],  # .f so no F90 wrappers
#   output : ['laplace_filtermodule.c', 'laplace_filter-f2pywrappers.f'],
#   command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'laplace_filter', '--lower'],
#   install_dir : 'SpectrumExtractor'
# )

laplace_source = custom_target('laplace_filter',
   input : ['laplace_filter.f95'],  # .f so no F90 wrappers
   output : ['laplace_filtermodule.c', 'laplace_filter-f2pywrappers.f'],
   command : [py, '-m', 'numpy.f2py', '-c', '@INPUT@', '-m', 'laplace_filter', '--lower'],
#   install_dir : 'SpectrumExtractor'
 )


inc_np = include_directories(incdir_numpy, incdir_f2py)

py.extension_module('laplace_filter',
  ['laplace_filter.f95', laplace_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)