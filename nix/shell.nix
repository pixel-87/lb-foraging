{
  mkShell,
  callPackage,
  stdenv,

  # python tooling
  python3,
  uv,
  pyright,
  ruff,

  # graphics 
  xvfb-run,
  zlib,
  libGL,
  libGLU,
  xorg,
  freetype,
  fontconfig,
}:

let
  defaultPackage = callPackage ./default.nix { };
in
mkShell {
  inputsFrom = [ defaultPackage ];

  packages = [
    python3
    uv
    pyright
    ruff
    xvfb-run
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:${zlib}/lib:${libGL}/lib:${libGLU}/lib:${xorg.libX11}/lib:${xorg.libXcursor}/lib:${xorg.libXi}/lib:${xorg.libXinerama}/lib:${freetype}/lib:${fontconfig.lib}/lib:$LD_LIBRARY_PATH

    if [ ! -d ".venv" ]; then
      uv venv
    fi
    source .venv/bin/activate

    uv pip install -e .
  '';

}
