# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/seba/esp/v5.1.2/esp-idf/components/bootloader/subproject"
  "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader"
  "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader-prefix"
  "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader-prefix/tmp"
  "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader-prefix/src/bootloader-stamp"
  "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader-prefix/src"
  "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/seba/Code/SistEmb/proyecto/bottle_detector_dispenser/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
