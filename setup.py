from setuptools import setup, find_packages
from pydoc import ModuleScanner 


modules = []
def callback(path, modname, desc, modules=modules):
    if modname and modname[-9:] == '.__init__':
        modname = modname[:-9] + ' (package)'
    if modname.find('.') < 0:
        modules.append(modname)
def onerror(modname):
    callback(None, modname, None)

ModuleScanner().run(callback, onerror=onerror)

print(modules)

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="PyHa_test", 
        version="0.0.14",
        author="Jacob Ayers, Sean Perry, Sam Prestrelski, Vannessa Salgado",
        #author_email="<youremail@email.com>",
        description="A python package for automatically detecting species and comparing to ground truth",
        #long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=modules, # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)