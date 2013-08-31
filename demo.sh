#!/bin/sh

set -e

./libmf convert movielens movielens.tr.bin
./libmf convert movielens movielens.te.bin
./libmf train --tr-rmse --obj -k 40 -p 0.05 -q 0.05 -g 0.003 -v movielens.te.bin movielens.tr.bin model
./libmf predict movielens.te.bin model output
