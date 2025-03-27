#!/usr/bin/env bash

apt-get install -y cpufrequtils
apt-get install -y linux-tools-common
apt-get install -y linux-tools-generic
apt-get install -y linux-tools-$(uname -r)

echo "Step - 1"
# https://askubuntu.com/questions/619875/disabling-intel-turbo-boost-in-ubuntu
echo "Disabling Intel Turbo Boost in Ubuntu"
echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

echo ""

# https://serverfault.com/questions/235825/disable-hyperthreading-from-within-linux-no-access-to-bios
echo "Step - 2"
echo "Disable Simultaneous Multithreading (SMT) allows multiple execution threads to be executed on a single physical CPU core"
echo off | sudo tee /sys/devices/system/cpu/smt/control

echo ""

# https://nixcp.com/disable-cpu-frecuency-scaling/
# https://askubuntu.com/questions/1021748/set-cpu-governor-to-performance-in-18-04
echo "Step - 3"
echo "Disable CPU Frecuency Scaling on Linux and run CPU at Full Speed."
sudo cpupower frequency-set -g performance
echo ""

echo "Step - 4"
echo "Turn off various services"
sudo systemctl stop falcon-sensor.service
sudo systemctl stop containerd.service
sudo service gdm3 stop
sudo service docker stop

sudo systemctl stop amagent
sudo systemctl stop snapd
sudo systemctl stop rsyslog.service
sudo systemctl stop ModemManager.service
sudo systemctl stop NetworkManager.service
sudo systemctl stop rsyslog.service
sudo systemctl stop avahi-daemon.socket
sudo systemctl stop avahi-daemon.service

sudo systemctl stop teamviewerd.service

echo "Step - 5"
echo "Kill any python scripts"
sudo kill -9 `pidof python3`

echo ""
echo "PREPARE ENVIRONMENT FINISHED [OK]"
