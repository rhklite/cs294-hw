# python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3-l 2 -s 64 -b <b*> -lr <r*> -rtg --exp_name hc_b<b*>_r<r*> --dirIP

for i in 10000 30000 50000
do
	for j in 0.005 0.01 0.02
    do
        python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $i -lr $j -rtg --nn_baseline --exp_name "hc_b${i}_r${j}" --dir problem_8
    done
done
