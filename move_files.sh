#!/bin/bash

[ "$1" == "" ] && echo "Please Enter a name" && exit

echo "Move files to $1"
mkdir -p back/$1

mv -v logs/*.log back/$1/
mv -v models/model_*.ns back/$1/
cp -v models/*.npy back/$1/
