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

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/Cellar/cmake/3.21.3_1/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/local/Cellar/cmake/3.21.3_1/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

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
# Target rules for targets named 60s_TV_ref

# Build rule for target.
60s_TV_ref: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 60s_TV_ref
.PHONY : 60s_TV_ref

# fast build rule for target.
60s_TV_ref/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV_ref.dir/build.make CMakeFiles/60s_TV_ref.dir/build
.PHONY : 60s_TV_ref/fast

60s_TV_ref.o: 60s_TV_ref.cpp.o
.PHONY : 60s_TV_ref.o

# target to build an object file
60s_TV_ref.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV_ref.dir/build.make CMakeFiles/60s_TV_ref.dir/60s_TV_ref.cpp.o
.PHONY : 60s_TV_ref.cpp.o

60s_TV_ref.i: 60s_TV_ref.cpp.i
.PHONY : 60s_TV_ref.i

# target to preprocess a source file
60s_TV_ref.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV_ref.dir/build.make CMakeFiles/60s_TV_ref.dir/60s_TV_ref.cpp.i
.PHONY : 60s_TV_ref.cpp.i

60s_TV_ref.s: 60s_TV_ref.cpp.s
.PHONY : 60s_TV_ref.s

# target to generate assembly for a file
60s_TV_ref.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/60s_TV_ref.dir/build.make CMakeFiles/60s_TV_ref.dir/60s_TV_ref.cpp.s
.PHONY : 60s_TV_ref.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... 60s_TV_ref"
	@echo "... 60s_TV_ref.o"
	@echo "... 60s_TV_ref.i"
	@echo "... 60s_TV_ref.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

