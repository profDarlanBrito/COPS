#!/bin/sh

echo 'Starting script tabu search ...'

cd $2
echo $(pwd)

command="$1 tabu_search.py --path $3"
echo $command

eval "$command"