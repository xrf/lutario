# Lutario: In-medium similarity renormalization group method

Lutario is an implementation of the in-medium similarity renormalization group method used for solving quantum many-body problems.

## Building

Get the following dependencies:

  - A modern C++11 compiler, e.g. [Clang](http://clang.llvm.org/), [GCC](https://gcc.gnu.org/)
  - [GNU Make](https://www.gnu.org/software/make/)
  - [CBLAS](http://www.netlib.org/blas/faq.html#_5_a_id_are_optimized_blas_libraries_available_where_can_i_find_vendor_supplied_blas_a_are_optimized_blas_libraries_available_where_can_i_find_optimized_blas_libraries)
  - [sg-ode](https://github.com/xrf/sg-ode)

Within the project directory, copy [`etc/local.mk.example`](etc/local.mk.example) to `etc/local.mk` and customize the Makefile variables appropriately to reflect your configuration.

To build, run:

~~~sh
make
~~~

The pairing model demo can be executed via `bin/main`.

To run the tests, do:

~~~sh
make check
~~~
