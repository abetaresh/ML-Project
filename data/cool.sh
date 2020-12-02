# Validation split
for f in $(find training/*.jpg -type f -printf "%f\n" | shuf -n $(bc <<< "$(\ls training | wc -l) / 10")); do mv training/$f validation/$f; done
