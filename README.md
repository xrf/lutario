# Lutario: In-medium similarity renormalization group method

Lutario is an implementation of the in-medium similarity renormalization group method used for solving quantum many-body problems.

## Building

A C++11 compiler is required.  [GNU Make](https://www.gnu.org/software/make/) is also needed.

Get the following dependencies:

  - [CBLAS](http://www.netlib.org/blas/faq.html#_5_a_id_are_optimized_blas_libraries_available_where_can_i_find_vendor_supplied_blas_a_are_optimized_blas_libraries_available_where_can_i_find_optimized_blas_libraries)
  - [sg-ode](https://github.com/xrf/sg-ode)

Within the project directory, create an empty file `etc/local.mk` to customize the Makefile variables.  For example:

~~~mk
# etc/local.mk
CC=clang
CXX=clang++
CPPFLAGS=-g -O2
CFLAGS=-std=c99
CXXFLAGS=-std=c++11
CBLAS_LIBS=-lcblas
ODE_LIBS=-lsgode
~~~

To build, run:

~~~sh
make
~~~

The pairing model demo executed by calling `bin/main`.

To run the tests, do:

~~~sh
make check
~~~
