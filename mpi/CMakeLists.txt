cmake_minimum_required(VERSION 3.22)
project(mpi)

set(CMAKE_CXX_STANDARD 20)

add_executable(mpi main.cpp)

########################################################################
# link library

find_package(MPI REQUIRED) #make it REQUIRED, if you want

# for "C" language
#target_include_directories(mpi PRIVATE ${MPI_C_INCLUDE_PATH})

# for "C++" language
target_link_libraries(mpi PRIVATE MPI::MPI_CXX)

set_target_properties(mpi
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

########################################################################

# script

#set(SOURCE_FILES main.c)
#
#add_custom_command(
#        OUTPUT "out"
#        COMMAND mpicc ${SOURCE_FILES}
##        COMMAND mpic++ ${SOURCE_FILES}
#        COMMAND mpiexec -np 4 ./a.out
##        COMMAND sbatch ./run_sbatch_config.sh
#        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
#        COMMENT "Compile mpi"
#        VERBATIM
#)
#
#add_custom_target(mpi_custom_command DEPENDS "out")

########################################################################
