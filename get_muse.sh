dir_name=MUSE
rm -rf $dir_name
mkdir $dir_name
mkdir $dir_name/trainA
mkdir $dir_name/trainB
mkdir $dir_name/testA
mkdir $dir_name/testB

wget -q https://storage.cloud.google.com/muse_data/MUSE/testA/Stack00{90..99}.jpg -P $dir_name/testA
wget -q https://storage.cloud.google.com/muse_data/MUSE/testB/Stack00{90..99}.jpg -P $dir_name/testB
echo "Done downloading testing files"
wget -q https://storage.cloud.google.com/muse_data/MUSE/trainA/Stack00{00..90}.jpg -P $dir_name/trainA
wget -q https://storage.cloud.google.com/muse_data/MUSE/trainB/Stack00{00..90}.jpg -P $dir_name/trainB
echo "Done donwloading training files"

