#!/bin/sh

set -e

./libmf convert movielens.sub movielens.sub.tr.bin
./libmf convert movielens.sub movielens.sub.te.bin
./libmf train --tr-rmse --obj -k 40 -p 0.05 -q 0.05 -g 0.003 -v movielens.sub.te.bin movielens.sub.tr.bin model
./libmf predict movielens.sub.te.bin model output
