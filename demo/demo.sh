#!/bin/sh
train=../mf-train
predict=../mf-predict

if [ ! -s $train ] || [ ! -s $predict ]
then
    (cd .. && make)
fi

##########################################################################
# Real-valued matrix factorization
##########################################################################
echo "--------------------------------"
echo "Real-valued matrix factorization"
echo "--------------------------------"
# In-memory training with holdout valudation
$train -f 0 -l2 0.05 -k 100 -t 10 -p bigdata.te.txt bigdata.tr.txt model.txt
# Do prediction and show MAE
$predict -e 1 bigdata.te.txt model.txt output.txt

##########################################################################
# Binary matrix factorization
##########################################################################
echo "---------------------------"
echo "binary matrix factorization"
echo "---------------------------"
# In-memory training with holdout valudation
$train -f 5 -l2 0.01 -k 64 -p bigdata_bin.te.txt bigdata_bin.tr.txt model.txt
# Do prediction and show accuracy
$predict -e 6 bigdata_bin.te.txt model.txt output.txt

##########################################################################
# One-class matrix factorization
##########################################################################
echo "------------------------------"
echo "one-class matrix factorization"
echo "------------------------------"
# In-memory training
$train -f 10 -l2 0.01 -k 32 -p bigdata_one.te.txt bigdata_one.tr.txt model.txt
# Do prediction and show row-oriented MPR
$predict -e 10 bigdata_one.te.txt model.txt output.txt
# Do prediction and show row-oriented AUC
$predict -e 12 bigdata_one.te.txt model.txt output.txt
