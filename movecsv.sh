cd /demo-image-processing
for f in *.json
do 
   cp -v "$f" /data/"${f%.csv}".json
done