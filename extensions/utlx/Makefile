WHEEL_DIR ?= dist

.PHONY: build
build:
	pip wheel . --no-build-isolation --no-deps --wheel-dir $(WHEEL_DIR)

.PHONY: install
install:
	pip install $(WHEEL_DIR)/*.whl --force-reinstall

.PHONY: test
test:
	pytest test/ || [ $$? -eq 5 ]
