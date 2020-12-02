# Check that the labels are correct
CSV=$1

cat $CSV | while read line; 
do 
    echo "${line%% *} : $(ls *.jpg | fgrep -c ${line%% *})"; 
done
