for i in 10000 30000 50000
do
    for j in 0.01 0.03 0.05
    do
        python plot.py "problem_5/hc_b${i}_r"* --prefix $i
    done
done