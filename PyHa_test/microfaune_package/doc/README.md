# Automatic documentation generation

## Documentation

Once build, the documentation is available by opening the file `build/html/index.html`.


## Build documentation

To generate the documentation, the packages *sphinx* and *sphinx-rtd-theme* need to be 
installed. 

All the commands must be launched in console from the folder `doc/`.

To remove the previous generated documentation:
```bash
make clean
```

To generate the html documentation:
```bash
make html
```
