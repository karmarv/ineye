IFS=$'\n'

# Install FFMPEG - https://ffmpeg.org/download.html

# Convert AVI videos to MP4 format in the current folder
convert_to_mp4(){
  list="`find . -type f -name '*.avi'`"
  for f in $list;
    do name=`echo "$f"`
    echo "$name"
    ffmpeg -nostdin -i "$f" -c:v copy -c:a copy -y  "${name%.*}.mp4"
  done
}


extract_frames_1fps(){
  ffmpeg -i $1 -r 1 frame%d.png
}

