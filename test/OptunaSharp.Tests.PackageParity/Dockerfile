# Isolated environment for the OptunaSharp package-parity gate.
#
# Build context is THIS project directory only, so src/ is not present anywhere in the image — the
# tests can bind to nothing but the PACKAGED OptunaSharp (dropped into ./feed by the pipeline) and its
# declared dependency graph, exactly as a downstream NuGet consumer would. Real ML packages are
# pip-installed so the parity oracles are produced by real Optuna + real xgboost/lightgbm/sklearn.
#
#   docker build -t optunasharp-pkgparity --build-arg OPTUNASHARP_VERSION=0.1.0 \
#                -f test/OptunaSharp.Tests.PackageParity/Dockerfile test/OptunaSharp.Tests.PackageParity
#   docker run --rm optunasharp-pkgparity
FROM mcr.microsoft.com/dotnet/sdk:10.0

# libgomp1 is xgboost/lightgbm's OpenMP runtime; python3 + pip drive the real-package oracles.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages \
      optuna xgboost lightgbm scikit-learn numpy

WORKDIR /work
ARG OPTUNASHARP_VERSION=0.1.0
ENV OPTUNASHARP_VERSION=${OPTUNASHARP_VERSION} \
    PACKAGE_PARITY_ORACLE_DIR=/work/oracle \
    PACKAGE_PARITY_REQUIRE=1

# ONLY the isolated test project (incl. its ./feed with the packed .nupkg) enters the image.
COPY . ./

# Restore the PACKAGED OptunaSharp from the local feed. Its NumSharp / NumSharp.Interop.OpenBLAS deps
# ALSO come from the local feed (journey3 build, pinned by NuGet.config source-mapping) — nuget.org has
# only an older NumSharp 0.60.0 and no OpenBLAS package; everything else still restores from nuget.org.
RUN dotnet restore OptunaSharp.Tests.PackageParity.csproj -p:OptunaSharpPackageVersion=${OPTUNASHARP_VERSION}

# 1) generate oracles from REAL optuna + xgboost/lightgbm/sklearn, then 2) replay them against the
#    packaged OptunaSharp. PACKAGE_PARITY_REQUIRE=1 makes a missing oracle a hard failure.
CMD python3 gen_oracle.py \
 && dotnet test OptunaSharp.Tests.PackageParity.csproj -c Release -f net10.0 --no-restore \
      -p:OptunaSharpPackageVersion=${OPTUNASHARP_VERSION}
