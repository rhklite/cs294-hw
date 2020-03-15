for i in $(seq 500 500 5000)
do
    python plot.py "data/problem_5/hc_b${i}_r"* --prefix $i
done