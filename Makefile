# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.21.3_1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.21.3_1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/gracesun/Desktop/645/project/FastCode_Image_Filter

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/gracesun/Desktop/645/project/FastCode_Image_Filter

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/local/Cellar/cmake/3.21.3_1/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/Cellar/cmake/3.21.3_1/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/gracesun/Desktop/645/project/FastCode_Image_Filter/CMakeFiles /Users/gracesun/Desktop/645/project/FastCode_Image_Filter//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/gracesun/Desktop/645/project/FastCode_Image_Filter/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named emboss

# Build rule for target.
emboss: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 emboss
.PHONY : emboss

# fast build rule for target.
emboss/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/emboss.dir/build.make CMakeFiles/emboss.dir/build
.PHONY : emboss/fast

#=============================================================================
# Target rules for targets named brightness

# Build rule for target.
brightness: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 brightness
.PHONY : brightness

# fast build rule for target.
brightness/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness.dir/build.make CMakeFiles/brightness.dir/build
.PHONY : brightness/fast

#=============================================================================
# Target rules for targets named sepia

# Build rule for target.
sepia: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 sepia
.PHONY : sepia

# fast build rule for target.
sepia/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sepia.dir/build.make CMakeFiles/sepia.dir/build
.PHONY : sepia/fast

#=============================================================================
# Target rules for targets named 60s_TV

# Build rule for target.
60s_TV: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 60s_TV
.PHONY : 60s_TV

# fast build rule for target.
60s_TV/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV.dir/build.make CMakeFiles/60s_TV.dir/build
.PHONY : 60s_TV/fast

#=============================================================================
# Target rules for targets named brightness_ref

# Build rule for target.
brightness_ref: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 brightness_ref
.PHONY : brightness_ref

# fast build rule for target.
brightness_ref/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness_ref.dir/build.make CMakeFiles/brightness_ref.dir/build
.PHONY : brightness_ref/fast

#=============================================================================
# Target rules for targets named duo_tone

# Build rule for target.
duo_tone: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 duo_tone
.PHONY : duo_tone

# fast build rule for target.
duo_tone/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/duo_tone.dir/build.make CMakeFiles/duo_tone.dir/build
.PHONY : duo_tone/fast

60s_TV.o: 60s_TV.cpp.o
.PHONY : 60s_TV.o

# target to build an object file
60s_TV.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV.dir/build.make CMakeFiles/60s_TV.dir/60s_TV.cpp.o
.PHONY : 60s_TV.cpp.o

60s_TV.i: 60s_TV.cpp.i
.PHONY : 60s_TV.i

# target to preprocess a source file
60s_TV.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV.dir/build.make CMakeFiles/60s_TV.dir/60s_TV.cpp.i
.PHONY : 60s_TV.cpp.i

60s_TV.s: 60s_TV.cpp.s
.PHONY : 60s_TV.s

# target to generate assembly for a file
60s_TV.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV.dir/build.make CMakeFiles/60s_TV.dir/60s_TV.cpp.s
.PHONY : 60s_TV.cpp.s

brightness.o: brightness.cpp.o
.PHONY : brightness.o

# target to build an object file
brightness.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness.dir/build.make CMakeFiles/brightness.dir/brightness.cpp.o
.PHONY : brightness.cpp.o

brightness.i: brightness.cpp.i
.PHONY : brightness.i

# target to preprocess a source file
brightness.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness.dir/build.make CMakeFiles/brightness.dir/brightness.cpp.i
.PHONY : brightness.cpp.i

brightness.s: brightness.cpp.s
.PHONY : brightness.s

# target to generate assembly for a file
brightness.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness.dir/build.make CMakeFiles/brightness.dir/brightness.cpp.s
.PHONY : brightness.cpp.s

brightness_ref.o: brightness_ref.cpp.o
.PHONY : brightness_ref.o

# target to build an object file
brightness_ref.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness_ref.dir/build.make CMakeFiles/brightness_ref.dir/brightness_ref.cpp.o
.PHONY : brightness_ref.cpp.o

brightness_ref.i: brightness_ref.cpp.i
.PHONY : brightness_ref.i

# target to preprocess a source file
brightness_ref.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness_ref.dir/build.make CMakeFiles/brightness_ref.dir/brightness_ref.cpp.i
.PHONY : brightness_ref.cpp.i

brightness_ref.s: brightness_ref.cpp.s
.PHONY : brightness_ref.s

# target to generate assembly for a file
brightness_ref.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/brightness_ref.dir/build.make CMakeFiles/brightness_ref.dir/brightness_ref.cpp.s
.PHONY : brightness_ref.cpp.s

duo_tone.o: duo_tone.cpp.o
.PHONY : duo_tone.o

# target to build an object file
duo_tone.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/duo_tone.dir/build.make CMakeFiles/duo_tone.dir/duo_tone.cpp.o
.PHONY : duo_tone.cpp.o

duo_tone.i: duo_tone.cpp.i
.PHONY : duo_tone.i

# target to preprocess a source file
duo_tone.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/duo_tone.dir/build.make CMakeFiles/duo_tone.dir/duo_tone.cpp.i
.PHONY : duo_tone.cpp.i

duo_tone.s: duo_tone.cpp.s
.PHONY : duo_tone.s

# target to generate assembly for a file
duo_tone.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/duo_tone.dir/build.make CMakeFiles/duo_tone.dir/duo_tone.cpp.s
.PHONY : duo_tone.cpp.s

emboss.o: emboss.cpp.o
.PHONY : emboss.o

# target to build an object file
emboss.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/emboss.dir/build.make CMakeFiles/emboss.dir/emboss.cpp.o
.PHONY : emboss.cpp.o

emboss.i: emboss.cpp.i
.PHONY : emboss.i

# target to preprocess a source file
emboss.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/emboss.dir/build.make CMakeFiles/emboss.dir/emboss.cpp.i
.PHONY : emboss.cpp.i

emboss.s: emboss.cpp.s
.PHONY : emboss.s

# target to generate assembly for a file
emboss.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/emboss.dir/build.make CMakeFiles/emboss.dir/emboss.cpp.s
.PHONY : emboss.cpp.s

sepia.o: sepia.cpp.o
.PHONY : sepia.o

# target to build an object file
sepia.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sepia.dir/build.make CMakeFiles/sepia.dir/sepia.cpp.o
.PHONY : sepia.cpp.o

sepia.i: sepia.cpp.i
.PHONY : sepia.i

# target to preprocess a source file
sepia.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sepia.dir/build.make CMakeFiles/sepia.dir/sepia.cpp.i
.PHONY : sepia.cpp.i

sepia.s: sepia.cpp.s
.PHONY : sepia.s

# target to generate assembly for a file
sepia.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sepia.dir/build.make CMakeFiles/sepia.dir/sepia.cpp.s
.PHONY : sepia.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... 60s_TV"
	@echo "... brightness"
	@echo "... brightness_ref"
	@echo "... duo_tone"
	@echo "... emboss"
	@echo "... sepia"
	@echo "... 60s_TV.o"
	@echo "... 60s_TV.i"
	@echo "... 60s_TV.s"
	@echo "... brightness.o"
	@echo "... brightness.i"
	@echo "... brightness.s"
	@echo "... brightness_ref.o"
	@echo "... brightness_ref.i"
	@echo "... brightness_ref.s"
	@echo "... duo_tone.o"
	@echo "... duo_tone.i"
	@echo "... duo_tone.s"
	@echo "... emboss.o"
	@echo "... emboss.i"
	@echo "... emboss.s"
	@echo "... sepia.o"
	@echo "... sepia.i"
	@echo "... sepia.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

