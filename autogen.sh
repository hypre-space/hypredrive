#!/bin/sh

set -e
libtoolize --force --copy
#autoscan
#autoheader
autoreconf -i
automake --add-missing --copy

echo "Bootstrap done! Now, run './configure'"
