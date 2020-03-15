# python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3-l 2 -s 64 -b <b*> -lr <r*> -rtg --exp_name hc_b<b*>_r<r*> --dirIP

for i in $(seq 500 500 5000)
do
	for j in $(seq 1e-4 4e-3 20e-3)
    do
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $i -lr $j -rtg --exp_name "hc_b${i}_r${j}" --dir problem_5
    done
done