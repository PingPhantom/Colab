{
  description = "Strata Studio devshell — llama.cpp tooling, huggingface-hub, uv, ruff";

  # ปักหมุด nixpkgs เพื่อให้ env เดิมทุกครั้ง (อัปเดตเองได้โดยแก้ค่านี้)
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/@NIXPKGS_REV@";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pyPkg = pkgs.@PY_VER@ or pkgs.python3;

      # คำสั่งโหลด GGUF จาก Hugging Face (ใช้ HF_REPO_ID override repo ได้)
      downloadDeepSeek = pkgs.writeShellScriptBin "download-deepseek" ''
        set -e
        REPO="''${HF_REPO_ID:-@CUSTOM_HF_REPO@}"
        if [[ "''${1:-}" == -* ]] || [ -z "''${1:-}" ]; then
          MODEL_DIR="@MODEL_DIR@"
        else
          MODEL_DIR="$1"; shift
        fi
        mkdir -p "$MODEL_DIR"
        ${pkgs.python3Packages.huggingface-hub}/bin/hf download "$REPO" --local-dir "$MODEL_DIR" "$@"
        echo "✅ ดาวน์โหลดเสร็จที่: $MODEL_DIR"
      '';
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pyPkg
          pkgs.python3Packages.huggingface-hub
          pkgs.uv
          pkgs.ruff
          pkgs.jq
          pkgs.ripgrep
          pkgs.curl
          pkgs.gnused
          pkgs.aria2
          downloadDeepSeek
        ];
        env = {
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${pyPkg}/bin/python";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.glib ];
        };
      };
    };
}
