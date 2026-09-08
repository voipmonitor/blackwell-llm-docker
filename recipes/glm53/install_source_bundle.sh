#!/usr/bin/env bash
set -euo pipefail

# Install a complete committed source tree while retaining ignored runtime
# artifacts, such as compiled extensions and separately packaged dependencies.
# The caller supplies an exact commit and tree from the serving source lock.
if (($# != 4)); then
    echo "usage: $0 <bundle> <commit> <tree> <repository-directory>" >&2
    exit 2
fi
readonly bundle=$(realpath -e "$1")
readonly commit=$2 tree=$3 destination=$(realpath -m "$4")
if [[ ! $commit =~ ^[0-9a-f]{40}$ || ! $tree =~ ^[0-9a-f]{40}$ ]]; then
    echo "Source installation requires full Git commit and tree identifiers." >&2
    exit 2
fi
if [[ $destination == / || $destination == /root || $destination == /opt ]]; then
    echo "The destination must be a dedicated source repository." >&2
    exit 2
fi
if ! git bundle list-heads "$bundle" | awk -v commit="$commit" '
    $1 == commit { found = 1 } END { exit !found }'; then
    echo "The bundle does not advertise the locked source commit." >&2
    exit 1
fi
if [[ ! -d $destination/.git ]]; then
    if [[ -e $destination && -n $(ls -A "$destination") ]]; then
        echo "Refusing to initialize a nonempty non-repository destination." >&2
        exit 1
    fi
    git init --quiet "$destination"
fi
if [[ $(git -C "$destination" rev-parse --show-toplevel) != "$destination" ]]; then
    echo "The destination is not its own Git working tree." >&2
    exit 1
fi
git -C "$destination" fetch --quiet --no-tags "$bundle" "$commit"
if [[ $(git -C "$destination" rev-parse "${commit}^{tree}") != "$tree" ]]; then
    echo "The bundle's source tree differs from the source lock." >&2
    exit 1
fi

# Deletions are limited to paths tracked by the image's existing index and
# absent from the locked tree. Git objects retain their original contents.
readonly previous_tree=$(git -C "$destination" write-tree)
while IFS= read -r -d '' path; do
    rm -f -- "$destination/$path"
done < <(git -C "$destination" diff --name-only --diff-filter=D -z \
    "$previous_tree" "$commit")
git -C "$destination" update-ref refs/heads/image-source "$commit"
git -C "$destination" symbolic-ref HEAD refs/heads/image-source
git -C "$destination" read-tree "$commit"
git -C "$destination" checkout-index --all --force
git -C "$destination" diff --quiet HEAD
test "$(git -C "$destination" write-tree)" = "$tree"
