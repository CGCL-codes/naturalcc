cd Repos/Python/
find . -maxdepth 2 -type d -empty | xargs -i sudo rm -rf {}
find . -maxdepth 1 -type d -empty | xargs -i sudo rm -rf {}

cd ../../Code/Python/
find . -maxdepth 2 -type d -empty | xargs -i sudo rm -rf {}
find . -maxdepth 1 -type d -empty | xargs -i sudo rm -rf {}

ls ../../Repos/Python/ | xargs -i sudo rm -rf {}
sudo rm -rf ../../Repos/Python/*
