# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
# 3.18+ for CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(spmv LANGUAGES CXX CUDA VERSION 0.1.0.0)

include(CheckLanguage)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

if (MPI_FOUND)
message(STATUS "MPI_CXX_COMPILER:      ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_INCLUDE_DIRS:  ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LIBRARIES:     ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI_CUDA_INCLUDE_DIRS: ${MPI_CUDA_INCLUDE_DIRS}")
message(STATUS "MPI_CUDA_LIBRARIES:    ${MPI_CUDA_LIBRARIES}")
message(STATUS "MPIEXEC_EXECUTABLE:    ${MPIEXEC_EXECUTABLE}")
message(STATUS "MPIEXEC_NUMPROC_FLAG:  ${MPIEXEC_NUMPROC_FLAG}")
message(STATUS "MPIEXEC_MAX_NUMPROCS:  ${MPIEXEC_MAX_NUMPROCS}")
message(STATUS "MPIEXEC_PREFLAGS:      ${MPIEXEC_PREFLAGS}")
message(STATUS "MPIEXEC_POSTFLAGS:     ${MPIEXEC_POSTFLAGS}")

endif()

function(set_cxx_options target)
target_compile_options(${target} PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:
 -Wall
 -Wextra
 -Wcast-align;
 -Wdisabled-optimization;
 -Wformat=2;
 -Winit-self;
 -Wlogical-op;
 -Wmissing-include-dirs;
 -Woverloaded-virtual;
 -Wpointer-arith;
 -Wshadow;
 -Wstrict-aliasing;
 -Wswitch-enum;
 -Wvla;
 >
)
endfunction()

function(set_cuda_options target)
target_compile_options(${target} PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:
--Wno-deprecated-gpu-targets;
--expt-extended-lambda;
-Xcompiler=-Wall;
-Xcompiler=-Wextra;
-Xcompiler=-Wcast-align;
-Xcompiler=-Wdisabled-optimization;
-Xcompiler=-Wformat=2;
-Xcompiler=-Winit-self;
-Xcompiler=-Wlogical-op;
-Xcompiler=-Wmissing-include-dirs;
-Xcompiler=-Woverloaded-virtual;
-Xcompiler=-Wpointer-arith;
-Xcompiler=-Wshadow;
-Xcompiler=-Wstrict-aliasing;
-Xcompiler=-Wswitch-enum;
-Xcompiler=-Wvla;
-Xptxas=-v;
>
)
endfunction()

function(set_cxx_standard target)
set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${target} PROPERTY CUDA_STANDARD 11)
set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
endfunction()

# copy run-all.sh to build directory
#configure_file(${CMAKE_CURRENT_LIST_DIR}/run-all.sh ${CMAKE_CURRENT_BINARY_DIR}/run-all.sh COPYONLY)

if (MPI_FOUND)
  add_executable(main main.cu)
  target_include_directories(main PRIVATE SYSTEM ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(main ${MPI_CXX_LIBRARIES})
  target_link_libraries(main CUDA::nvToolsExt)
  # target_include_directories(main PRIVATE ${MPI_CXX_INCLUDE_PATH})
  # target_compile_options(main PRIVATE ${MPI_CXX_COMPILE_FLAGS})
  # target_link_libraries(main ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  set_cxx_options(main)
  set_cxx_standard(main)
endif()

