# python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3-l 2 -s 64 -b <b*> -lr <r*> -rtg --exp_name hc_b<b*>_r<r*> --dirIP

python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 300 -lr 0.01 -rtg --exp_name "hc_b300_r0.01" --dir problem_5_sol

python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 10000 -lr 0.05 -rtg --exp_name "hc_b10000_r0.05" --dir problem_5_sol

for i in 10000 30000 50000
do
	for j in 0.01 0.03 0.05
    do
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $i -lr $j -rtg --exp_name "hc_b${i}_r${j}" --dir problem_5
    done
done