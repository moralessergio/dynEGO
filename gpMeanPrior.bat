for ((  i = 0 ;  i <= 1024;  i++  ))
do
  echo $i completed
  ./gpMovPeaks_1d $i
done
