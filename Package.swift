// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MoshiMac",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "MoshiMac",
            targets: ["MoshiMac"]
        ),
    ],
    dependencies: [
        // MLX Swift for the model
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0"),

        // Global keyboard shortcuts
        .package(url: "https://github.com/sindresorhus/KeyboardShortcuts", from: "2.0.0"),

        // Launch at login
        .package(url: "https://github.com/sindresorhus/LaunchAtLogin", from: "5.0.0"),

        // HuggingFace Transformers (includes Hub and Tokenizers)
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .executableTarget(
            name: "MoshiMac",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                "KeyboardShortcuts",
                "LaunchAtLogin",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MoshiMac",
            resources: [
                .process("Resources")
            ],
            linkerSettings: [
                .unsafeFlags(["-L", "Sources/MoshiMac/Resources/lib"]),
                .linkedLibrary("rustymimi_c")
            ]
        ),
        .testTarget(
            name: "MoshiMacTests",
            dependencies: ["MoshiMac"]
        ),
    ]
)
