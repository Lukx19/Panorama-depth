cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(dataset_modifications)

find_package(TIFF REQUIRED)
find_package(PNG REQUIRED)

find_package(PCL 1.2 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wno-deprecated-declarations -std=c++11 -Wall -O3")
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/../bin/")

include_directories(${PCL_INCLUDE_DIRS}
                    ${TIFF_INCLUDE_DIRS}
                    ${PNG_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS}
                ${PNG_DEFINITIONS})


add_executable (segmentation segmentation.cpp)
target_link_libraries (segmentation
                        ${PCL_LIBRARIES}
                        ${TIFF_LIBRARIES}
                        ${PNG_LIBRARIES}
                        )


set(BINARIES segmentation)

install(TARGETS ${BINARIES} DESTINATION bin)