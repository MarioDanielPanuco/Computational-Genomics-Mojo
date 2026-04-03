{
  description = "A computational genomics library in Mojo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pixi.url = "github:prefix-dev/pixi";
  };

  outputs = { self, nixpkgs, flake-utils, pixi }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # CUDA packages — Linux only; evaluated lazily so macOS is unaffected
        cudaPkgs = if pkgs.stdenv.isLinux then
          (import nixpkgs {
            inherit system;
            config = { allowUnfree = true; cudaSupport = true; };
          }).cudaPackages
        else
          null;

        baseInputs = [
          pixi.packages.${system}.default
          pkgs.git
          pkgs.curl
          pkgs.which
          pkgs.gnumake
          pkgs.clang
        ];

        gpuInputs = pkgs.lib.optionals (cudaPkgs != null) [
          cudaPkgs.cudatoolkit
          cudaPkgs.cudnn
          pkgs.nvtopPackages.nvidia
        ];

      in {
        # ── Default shell: CPU-only, works on macOS and Linux ────────────────
        devShells.default = pkgs.mkShell {
          buildInputs = baseInputs;

          shellHook = ''
            echo "Mojo + Pixi dev environment (CPU)"
            echo "Run 'pixi install' to sync the MAX/Mojo environment."
            echo "For GPU support: nix develop .#gpu"
          '';
        };

        # ── GPU shell: CUDA + full toolchain, Linux only ─────────────────────
        devShells.gpu = pkgs.mkShell {
          buildInputs = if cudaPkgs != null then baseInputs ++ gpuInputs
                        else builtins.throw "GPU shell requires Linux (CUDA not available on ${system})";

          shellHook = ''
            echo "Mojo + Pixi + CUDA dev environment"

            export CUDA_PATH=${if cudaPkgs != null then "${cudaPkgs.cudatoolkit}" else ""}
            export LD_LIBRARY_PATH=${if cudaPkgs != null then "${cudaPkgs.cudatoolkit}/lib:${cudaPkgs.cudnn}/lib" else ""}:$LD_LIBRARY_PATH

            echo "Run 'pixi install' to sync the MAX/Mojo environment."
          '';
        };
      });
}
