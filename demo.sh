#!/bin/sh

set -e

./libmf convert smalldata smalldata.tr.bin
./libmf convert smalldata smalldata.te.bin
./libmf train --tr-rmse --obj -k 40 -p 0.05 -q 0.05 -g 0.003 -ub -1 -ib -1 --no-use-avg --rand-shuffle -v smalldata.te.bin smalldata.tr.bin model
./libmf predict smalldata.te.bin model output
