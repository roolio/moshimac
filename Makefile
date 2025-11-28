# MoshiMac Makefile

.PHONY: build run clean test help setup

help:
	@echo "MoshiMac - Build & Run Commands"
	@echo ""
	@echo "  make setup    - Initialize submodules and dependencies"
	@echo "  make build    - Build the project"
	@echo "  make run      - Build and run the app"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make test     - Run tests"
	@echo "  make release  - Build release version"
	@echo ""

setup:
	@echo "Initializing submodules..."
	git submodule update --init --recursive
	@echo "Resolving Swift package dependencies..."
	swift package resolve
	@echo "Setup complete!"

build:
	@echo "Building MoshiMac..."
	swift build

run: build
	@echo "Running MoshiMac..."
	.build/debug/MoshiMac

release:
	@echo "Building release version..."
	swift build -c release
	@echo "Release binary: .build/release/MoshiMac"

clean:
	@echo "Cleaning build artifacts..."
	swift package clean
	rm -rf .build

test:
	@echo "Running tests..."
	swift test

xcode:
	@echo "Generating Xcode project..."
	swift package generate-xcodeproj
	@echo "Opening Xcode..."
	open MoshiMac.xcodeproj

format:
	@echo "Formatting Swift code..."
	swift-format format -i -r Sources/

lint:
	@echo "Linting Swift code..."
	swift-format lint -r Sources/
