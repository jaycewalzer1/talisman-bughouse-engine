#!/bin/bash
# Lists what Phoenix offers for the build. Free: run it on a login node. Output also goes to cluster_survey.txt.
{
  for name in gcc cuda tensorrt onnxruntime cmake ninja python uv; do
    echo "=== module spider $name"
    module spider "$name" 2>&1
  done
  echo "=== partitions and GPUs"
  sinfo -o "%P %G" | sort -u
  echo "=== QoS"
  sacctmgr -n show qos format=Name%20,MaxWall 2>&1 | head -20
  echo "=== balance"
  pace-quota 2>&1 | grep -A3 "Job Charge Account"
} | tee cluster_survey.txt
