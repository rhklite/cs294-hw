for i in 10000 30000 50000
do
	for j in 0.01 0.03 0.05
    do
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $i -lr $j -rtg --exp_name "hc_b${i}_r${j}" --dir problem_5
    done
done

notify-send "cs294" "Problem 5 Training Complete"