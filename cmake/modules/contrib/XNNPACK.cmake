# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# XNNPACK support for TVM
# XNNPACK provides highly optimized neural network operators for ARM/x86/WASM
# Useful for mobile and embedded deployments

if(USE_XNNPACK STREQUAL "OFF")
  # XNNPACK disabled
  return()
endif()

# Add XNNPACK codegen source files to compiler
tvm_file_glob(GLOB XNNPACK_CONTRIB_SRC src/contrib/cochl/framework/xnnpack/*.cc)
list(APPEND COMPILER_SRCS ${XNNPACK_CONTRIB_SRC})

if(USE_XNNPACK STREQUAL "ON")
  # Find XNNPACK library in system paths
  find_library(EXTERN_LIBRARY_XNNPACK
    NAMES XNNPACK libXNNPACK xnnpack libxnnpack
  )

  if(EXTERN_LIBRARY_XNNPACK STREQUAL "EXTERN_LIBRARY_XNNPACK-NOTFOUND")
    # Try to find in 3rdparty
    set(XNNPACK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/XNNPACK)
    find_library(EXTERN_LIBRARY_XNNPACK
      NAMES XNNPACK libXNNPACK xnnpack libxnnpack
      HINTS "${XNNPACK_PATH}/build" "${XNNPACK_PATH}/lib"
    )

    if(EXTERN_LIBRARY_XNNPACK STREQUAL "EXTERN_LIBRARY_XNNPACK-NOTFOUND")
      message(WARNING "Cannot find XNNPACK library. Please build XNNPACK first or specify path with USE_XNNPACK=/path/to/xnnpack")
      return()
    endif()

    set(XNNPACK_INCLUDE_DIRS ${XNNPACK_PATH}/include)
  else()
    # System-wide installation
    set(XNNPACK_INCLUDE_DIRS /usr/include /usr/local/include)
  endif()

elseif(IS_DIRECTORY ${USE_XNNPACK})
  # Use pre-built XNNPACK from specified path
  set(XNNPACK_PATH ${USE_XNNPACK})
  set(XNNPACK_INCLUDE_DIRS ${XNNPACK_PATH}/include)

  find_library(EXTERN_LIBRARY_XNNPACK
    NAMES XNNPACK libXNNPACK xnnpack libxnnpack
    HINTS "${XNNPACK_PATH}" "${XNNPACK_PATH}/lib" "${XNNPACK_PATH}/build"
  )

  if(EXTERN_LIBRARY_XNNPACK STREQUAL "EXTERN_LIBRARY_XNNPACK-NOTFOUND")
    message(FATAL_ERROR "Cannot find XNNPACK library at ${XNNPACK_PATH}. "
                        "Please build XNNPACK first:\n"
                        "  cd ${XNNPACK_PATH}\n"
                        "  mkdir build && cd build\n"
                        "  cmake -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF ..\n"
                        "  make -j4")
  endif()

else()
  message(FATAL_ERROR "Invalid option: USE_XNNPACK=${USE_XNNPACK}. "
                      "Valid options: ON, OFF, or /path/to/xnnpack")
endif()

# Set include directories
include_directories(${XNNPACK_INCLUDE_DIRS})

# Link XNNPACK library to TVM runtime
list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_XNNPACK})

# Add preprocessor definition
add_definitions(-DUSE_XNNPACK=1)

message(STATUS "Build with XNNPACK support: ${EXTERN_LIBRARY_XNNPACK}")
