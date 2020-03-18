echo "Plotting Problem 4"
for i in lb sb
do
    python plot.py "data/problem_4/${i}"* --prefix "p4_${i}"
done

echo "Plotting Problem 5"
for i in 10000 30000 50000
do
    python plot.py "data/problem_5/hc_b${i}_r"* --prefix "p5_${i}"
done

echo "Plotting Problem 7"

python plot.py "data/problem_7/"* --prefix "p7"

echo "Plotting Problem 8 Param Search Result"
for i in 10000 30000 50000
do
    python plot.py "data/problem_8/hc_b${i}_r"* --prefix "p8_param_${i}"
done

echo "Plotting Problem 8 Optimal Param"

echo "Plotting Problem 8 Param Search Result"
python plot.py "data/problem_8_part2/hc"* --prefix "p8_optim"
