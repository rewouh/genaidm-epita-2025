{ pkgs ? import <nixpkgs> {
    config = {
      cudaSupport = true;
      allowUnfreePredicate = pkg:
        builtins.elem (pkgs.lib.getName pkg) [
          "cuda_cudart"
          "libcublas"
          "cuda_cccl"
          "cuda_nvcc"
        ];
    };
  }
}:

pkgs.mkShell {
  packages = with pkgs; [
    (ollama.override { acceleration = "cuda"; })
    python311
    python311Packages.numpy
    ollama
    glibc.bin
    uv
    prismlauncher
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
