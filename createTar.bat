a=$(more main.h | grep 'define D ' | tr -d '\t ')
b=${a:8:1}
exe1="gpMovPeaks_$b"
exe2="d"
exe="$exe1$exe2"
tarName1="gpMovPeaks_fMean_code_$b"
tarName2="d.tar"
tarName="$tarName1$tarName2"
rm $tarName
tar -cvf $tarName *.cpp *.c *.h $exe makefile

