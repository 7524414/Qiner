#!/bin/bash
u="$USER"
sudo apt update
yes | sudo apt install git
sudo git clone https://www.github.com/forci0ne/Qiner.git /home/$u/Qiner
yes | sudo apt install screen
yes | sudo apt install htop
yes | sudo apt install g++
sleep 2
/usr/bin/g++ -Wall -march=native -Ofast -funroll-loops -pthread /home/$u/Qiner/QinerLinux.cpp -o /home/$u/Qiner/Qiner
sleep 2
cd /home/$u/Qiner
sudo chmod -R 777 Qiner
./Qiner 141.94.135.157 30