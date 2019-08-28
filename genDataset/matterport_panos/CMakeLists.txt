#-----------------------------------------------------------------------
cmake_minimum_required(VERSION 3.0)
project(MatterportPanos)
#set character set. unicode
#add_definitions(-DUNICODE -D_UNICODE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wno-deprecated-declarations -std=c++11 -Wall -O3")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/../bin/")

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(TIFF REQUIRED)

#------------------------------------------------------------------------
#cpp files
SET(Project_SRCS
	main.cpp
   )


INCLUDE_DIRECTORIES(
   ${EIGEN_INCLUDE_DIR}
   ${Boost_INCLUDE_DIRS}
            )
link_directories(${Boost_LIBRARY_DIRS}
                  ${TIFF_INCLUDE_DIRS}
                )

ADD_EXECUTABLE(convert2pano
                ${Project_SRCS}
              )

target_link_libraries(convert2pano
                  ${OpenCV_LIBRARIES}
                  ${Boost_LIBRARIES}
                  ${TIFF_LIBRARIES}
                  # Eigen3::Eigen
                  )
