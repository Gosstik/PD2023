#! /usr/bin/env bash

hive -f 0_create_tables.sql
hive -f queries.sql

