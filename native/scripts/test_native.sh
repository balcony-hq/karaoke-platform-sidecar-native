#!/bin/sh
set -eu

binary=${1:?native executable is required}
model=${2:?native model is required}
test -s "$model"

output=$(
  printf '%s\n' \
    '{"id":1,"type":"ping"}' \
    '{"id":2,"type":"load"}' \
    '{"id":3,"type":"shutdown"}' |
  VOCALARC_NATIVE_MODEL="${binary}.missing-model.f32" "$binary" --threads 1
)

printf '%s\n' "$output" | grep -q '"type":"pong"'
printf '%s\n' "$output" | grep -q '"type":"loaded"'
printf '%s\n' "$output" | grep -q '"type":"shutdown"'
echo "native protocol smoke test passed"
