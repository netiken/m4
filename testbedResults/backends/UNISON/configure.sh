#!/bin/bash
# Configure ns-3 with UNISON multi-threaded support and examples enabled
export CC=gcc-9
export CXX=g++-9
CXXFLAGS=-w ./ns3 configure --enable-mtp --enable-examples --build-profile=optimized --disable-tests --enable-python-bindings --disable-werror --disable-warnings
