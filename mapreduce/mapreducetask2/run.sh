#! /usr/bin/env bash

OUT_DIR1="vashkevich_mapreduce_task2_1"
OUT_DIR2="vashkevich_mapreduce_task2_2"
NUM_REDUCERS=8

hdfs dfs -rm -r -skipTrash "${OUT_DIR1}" > /dev/null
hdfs dfs -rm -r -skipTrash "${OUT_DIR2}" > /dev/null

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapred.job.name="vashkevich_mapreduce_task2_1" \
    -D mapreduce.job.reduces="${NUM_REDUCERS}" \
    -files mapper_1.py,reducer_1.py \
    -mapper mapper_1.py \
    -reducer reducer_1.py \
    -input /data/wiki/en_articles \
    -output "${OUT_DIR1}" > /dev/null

NUM_REDUCERS=1

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapred.job.name="vashkevich_mapreduce_task2_2" \
    -D mapreduce.job.reduces="${NUM_REDUCERS}" \
    -D mapreduce.job.output.key.comparator.class=org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator \
    -D mapreduce.partition.keycomparator.options=-nr \
    -files mapper_2.py,reducer_2.py \
    -mapper mapper_2.py \
    -reducer reducer_2.py \
    -input "${OUT_DIR1}" \
    -output "${OUT_DIR2}" > /dev/null

hdfs dfs -cat "${OUT_DIR2}/part-00000" | head -n 10
