{
  description = "Nockchain development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Match the exact toolchain from rust-toolchain.toml
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # Add additional extensions for development
        rustToolchainWithExtensions = rustToolchain.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Build inputs needed for the project
        buildInputs = with pkgs; [
          # Rust toolchain
          rustToolchainWithExtensions

          # Build tools
          pkg-config
          protobuf
          cmake

          # Libraries that might be needed
          openssl

          # For bindgen (murmur3-sys build)
          libclang.lib
          llvmPackages.clang
          stdenv.cc.cc.lib
        ];

        nativeBuildInputs = with pkgs; [
          pkg-config
          protobuf
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          # Environment variables
          RUST_SRC_PATH = "${rustToolchainWithExtensions}/lib/rustlib/src/rust/library";
          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

          # Ensure cargo and rustc use the correct toolchain
          shellHook = ''
            echo "Nockchain development environment"
            echo "Rust toolchain: nightly-2025-02-14"
            rustc --version
            cargo --version
          '';
        };

        # Expose the Rust toolchain for convenience
        packages.rust = rustToolchainWithExtensions;
      }
    );
}
