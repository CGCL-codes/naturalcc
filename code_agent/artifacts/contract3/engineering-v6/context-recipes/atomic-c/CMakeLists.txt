cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(contract3_atomic_c_context)
target_sources(app PRIVATE main.c)
