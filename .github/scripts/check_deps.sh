#!/usr/bin/env bash

set -euo pipefail

contains_dependency_all=true

while read -r package version; do
    # Skip the interpreter and build tooling: these are needed to build/install the
    # package in the pixi environment but are not runtime dependencies, so they
    # intentionally do not appear in pyproject.toml.
    if [[ $package == "python" || $package == "pip" || $package == "hatchling" ]]; then
        continue
    fi

    dependency="${package} ${version}"
    contains_dependency=$(yq -r ".project.dependencies | map(. == \"${dependency}\") | any" pyproject.toml)
    if [[ $contains_dependency == "false" ]]; then
        echo "${dependency} not found in pyproject.toml"
        contains_dependency_all=false
    fi
done < <(yq -r '.dependencies | to_entries | .[] | "\(.key) \(.value)"' pixi.toml)

if [[ $contains_dependency_all == "false" ]]; then
    exit 1
fi
