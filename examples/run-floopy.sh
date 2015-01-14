#! /bin/sh

ARG="$1"
shift

python $(which floopy) --target=cl:0,0 --lang=floopy "$ARG" - "$@"
