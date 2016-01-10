for ((  i = $1 ;  i <= $2;  i++  ))
do
  echo $i completed
  ./gpMovPeaks_2d $i
done

