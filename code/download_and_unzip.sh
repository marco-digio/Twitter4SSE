YEAR=2020
MONTH=11
DAY=01
URL=https://archive.org/download/archiveteam-twitter-stream-${YEAR}-${MONTH}/twitter-stream-${YEAR}-${MONTH}-${DAY}.zip 
wget -P ./Data/twitter-stream ${URL}
unzip ./Data/twitter-stream/twitter-stream-${YEAR}-${MONTH}-${DAY}.zip -d Data/twitter-stream
