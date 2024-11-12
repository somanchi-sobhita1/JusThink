{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils?ref=main";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system: {
    packages.default = with nixpkgs.legacyPackages.${system}; python3Packages.buildPythonPackage {
      pname = "JusThink";
      version = "1.0";

      src = ./.;

      # List all required Python packages
      buildInputs = [
        python312
        python312Packages.setuptools
        python312Packages.openai
        python312Packages.tiktoken
        python312Packages.networkx
        python312Packages.matplotlib
        python312Packages.numpy
        python312Packages.scikit-learn
        python312Packages.tenacity
        python312Packages.tqdm
        python312Packages.flask
        python312Packages.boto3 
      ];

      installPhase = ''
        mkdir -p $out/JusThink
        cp *.py $out/JusThink/
        cp *.json $out/JusThink/
        cp *.txt $out/JusThink/
        chmod +x $out/JusThink/*.py
      '';

    };
     JusThink = {
      options = {
        scriptPath = nixpkgs.lib.mkOption {
          type = nixpkgs.lib.types.str;
          default = "./JusThink.py";
          description = "Path to the Python script.";
        };
      };
     };
    
  });
}