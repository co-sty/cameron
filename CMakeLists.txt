cmake_minimum_required(VERSION 2.8)

project( cameron )

set(ENV{OpenCV_DIR} /usr/local)

find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "OpenCV ${OpenCV_DIR}") 
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#file(GLOB my_SRC
#	    "h/*.hpp"
	    #	    "src/*.cpp"
	    #	    "src/db/*.cpp"
	    #)

include_directories(
	/usr/local/include
	/usr/local
	h
)

add_executable( 
	cameron 
	main.cpp 
	#	h/db.hpp
	h/db.hpp
	h/glob_img.hpp
	src/glob_img.cpp
	src/db/db.cpp
	src/db/read_write.cpp
	src/db/svm.cpp
	#${my_SRC}
)

target_link_libraries( 
	cameron 
	${OpenCV_LIBS} 
)

target_compile_features(
	cameron 
	PRIVATE 
	cxx_range_for
)
