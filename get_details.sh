#!/bin/bash
tail -1000 logs/ga.log | egrep 'mean|fitness_value'| tail -2
cat logs/ga.log |grep fitness_value | awk '{print $9}'| awk -F ',' '{printf "%s,%s,%s,%s,%s\n", $2,$3,$4, $5, $6}'
grep mean logs/ga.log | grep -v nan| cut -d ']' -f3 | awk 'BEGIN{mn = 100000; mx=0}{s = $3;  if (s < mn) mn = s; if(s > mx) mx = s } END {print mn,mx}'
