#!/bin/sh

set -e
autoscan
aclocal
autoheader
autoconf
automake --add-missing --copy

echo "Bootstrap done! Now, run './configure'"
