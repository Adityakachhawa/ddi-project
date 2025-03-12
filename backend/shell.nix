{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "ddi-env";
  buildInputs = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.gcc
    pkgs.zlib.dev
    pkgs.linuxPackages.kernel.dev
  ];
  
  shellHook = ''
    export PYTHONPATH="${pkgs.python310}/lib/python3.10/site-packages:$PYTHONPATH"
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.glibc}/lib:$LD_LIBRARY_PATH"
  '';
}