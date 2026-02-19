# Shortcuts for building the project; the true build system is CMake, but this records common commands.

TRITON_DIR ?= $(shell ci/pick-local-artifact.py triton)
TRITON_BUILD_DIR ?= $(shell find $(TRITON_DIR)/build -maxdepth 1 -name "cmake.*" -type d | head -1)
LLVM_DIR ?= $(shell ci/pick-local-artifact.py llvm)
$(if $(and $(TRITON_DIR),$(TRITON_BUILD_DIR),$(LLVM_DIR)),,$(error Missing artifact directories))
BUILD_DIR ?= build
EXTRA_CMAKE_ARGS ?=

default: build

.PHONY: configure
configure:
	mkdir -p ${BUILD_DIR}
	LLVM_DIR="$(LLVM_DIR)" \
	TRITON_DIR="$(TRITON_DIR)" \
	TRITON_BUILD_DIR="$(TRITON_BUILD_DIR)" \
		cmake -S . -B ${BUILD_DIR} -G Ninja ${EXTRA_CMAKE_ARGS}

.PHONY: build
build: configure
	cmake --build ${BUILD_DIR}

.PHONY: test
test:
	ninja -C ${BUILD_DIR} check-lit-tests

.PHONY: clean
clean:
	rm -rf ${BUILD_DIR}

.PHONY: clean-all
clean-all: clean
	rm -rf triton-* llvm-*
