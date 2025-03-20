#!/usr/bin/env bash

set -e

# Clean, build test with code coverage, launch tests
cd ./../
./project_scripts.py -c -gc -gd -bd -td -j 8

# Generate name for test coverage
report_name="dopt|commit:`git rev-parse HEAD`/date:`git log -n1 --date short --pretty=format:%cD`"

# Collect coverage information
lcov --capture --directory ./ --output-file cov_all.info

# Remove several data from tracefile
lcov --remove cov_all.info  "/usr/include/*" "*/3rdparty/*" "/usr/lib/gcc/*" --output-file cov_strip.info

# Generate coverage report
genhtml cov_strip.info --output-directory ./coverage_report -t "${report_name}" --frames --show-details --legend --function-coverage --demangle-cpp

# Compress coverage report
cd ./coverage_report
tar -zcvf ./../coverage_report.tar.gz ./
cd -

# Open coverage report
xdg-open ./coverage_report/index.html &

# Remove artifacts
rm cov_strip.info
rm cov_all.info

echo "Script $0 has been finished [OK]"
