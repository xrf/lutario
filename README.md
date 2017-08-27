# `lutario`

## External dependencies

By default, the program links to an *external* BLAS and LAPACK implementation, supplied by a library named `blas`.  This library is expected to provide *both* BLAS and LAPACK through the C interface.

To link to external libraries that are not named `blas`, you will need to configure the appropriate symbolic links or linker scripts to redirect `blas` to whatever library of your choice.

To link to the Apple Accelerate framework on OS X, use `--no-default-features --features=accelerate`.

If you do not want to rely on an external library, you can have Cargo automatically build OpenBLAS from source using `--no-default-features --features=openblas`.

For more information on features, see:

  - [Choosing features of libraries](http://doc.crates.io/specifying-dependencies.html#choosing-features)
  - [Choosing features of end products](http://doc.crates.io/manifest.html#usage-in-end-products)
