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
 && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 git \
 && rm -rf /var/lib/apt/lists/*
# Install the VENDORED optuna — the EXACT submodule commit (the v5.0.0 release @ 01ddd17) OptunaSharp's
# samplers track. An older PyPI optuna (4.x) is a different TPE implementation and its trajectory diverges
# wildly (the SvmSklearn_Tpe oracle went 0.034 -> 4.73), so the commit is pinned rather than left to
# whatever PyPI resolves. numpy is pinned to 2.5.4, the port's byte-exact target since NumSharp 0.81.0
# (its OpenBLAS natives are numpy 2.5.4's scipy-openblas build, so the BLAS-backed oracles stay
# byte-identical); the erf-tolerant TPE parity then holds at 1e-9, exactly as the live interop suite does.
# Move this pin with the repo's $(NumSharpVersion), like the interop job's.
# The container has no access to src/ (isolation), so it fetches the commit from GitHub instead.
RUN pip3 install --no-cache-dir --break-system-packages "numpy==2.5.4" \
 && pip3 install --no-cache-dir --break-system-packages \
      "optuna @ git+https://github.com/optuna/optuna.git@01ddd17ab9f36c3d2817aed9afe376e1cb047d35" \
      xgboost lightgbm scikit-learn \
 && pip3 install --no-cache-dir --break-system-packages "numpy==2.5.4"

WORKDIR /work
ARG OPTUNASHARP_VERSION=0.1.0
ENV OPTUNASHARP_VERSION=${OPTUNASHARP_VERSION} \
    PACKAGE_PARITY_ORACLE_DIR=/work/oracle \
    PACKAGE_PARITY_REQUIRE=1

# ONLY the isolated test project (incl. its ./feed with the packed .nupkg) enters the image.
COPY . ./

# Restore the PACKAGED OptunaSharp from the local feed (pinned by NuGet.config source-mapping). Its
# NumSharp / NumSharp.Interop.OpenBLAS deps — and everything else — restore from nuget.org, exactly as
# a downstream consumer installing OptunaSharp would.
RUN dotnet restore OptunaSharp.Tests.PackageParity.csproj -p:OptunaSharpPackageVersion=${OPTUNASHARP_VERSION}

# 1) generate oracles from REAL optuna + xgboost/lightgbm/sklearn, then 2) replay them against the
#    packaged OptunaSharp. PACKAGE_PARITY_REQUIRE=1 makes a missing oracle a hard failure.
CMD python3 gen_oracle.py \
 && dotnet test OptunaSharp.Tests.PackageParity.csproj -c Release -f net10.0 --no-restore \
      -p:OptunaSharpPackageVersion=${OPTUNASHARP_VERSION}
