#!/usr/bin/env sh

TOOLS=/opt/caffe/build/tools

$TOOLS/caffe train --solver=ocr_solver.prototxt
