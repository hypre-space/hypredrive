#!/bin/sh

set -e
libtoolize --force --copy
autoscan
aclocal
autoheader
autoconf
automake --add-missing --copy

echo "Bootstrap done! Now, run './configure'"
