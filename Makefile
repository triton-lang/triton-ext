# Shortcuts for building the project; the true build system is CMake, but this records common commands.

TRITON_INSTALL_DIR ?= $(shell ci/pick-local-artifact.py triton)
LLVM_INSTALL_DIR ?= $(shell ci/pick-local-artifact.py llvm)
$(if $(and $(TRITON_INSTALL_DIR),$(LLVM_INSTALL_DIR)),,$(error Missing artifact directories))
BUILD_DIR ?= build
EXTRA_CMAKE_ARGS ?=

default: build

.PHONY: configure
configure:
	mkdir -p ${BUILD_DIR}
	LLVM_INSTALL_DIR="$(LLVM_INSTALL_DIR)" \
	TRITON_INSTALL_DIR="$(TRITON_INSTALL_DIR)" \
		cmake -S . -B ${BUILD_DIR} -G Ninja ${EXTRA_CMAKE_ARGS}

.PHONY: build
build: configure
	cmake --build ${BUILD_DIR}

.PHONY: test
test: check-lit-tests check-pytest-tests

.PHONY: check-lit-tests
check-lit-tests:
	ninja -C ${BUILD_DIR} check-lit-tests

# Plugin-import smoke tests: load each lib<name>.so via TRITON_PLUGIN_PATHS and
# `import triton`. Catches regressions in the plugin's static-init path that
# lit/FileCheck tests can't see (e.g. PluginInfo fields read by libtriton's loader).
.PHONY: check-pytest-tests
check-pytest-tests:
	TRITON_EXT_BUILD_DIR=${BUILD_DIR} python3 -m pytest testing/

.PHONY: clean
clean:
	rm -rf ${BUILD_DIR}

.PHONY: clean-all
clean-all: clean
	rm -rf triton-* llvm-*
