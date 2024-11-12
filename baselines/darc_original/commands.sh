rm -r ~/darc/reacher
python train_eval.py \
    --root_dir=~/darc/reacher \
    --gin_bindings='train_eval.environment_name="reacher"' \
    --gin_bindings='train_eval.delta_r_warmup=100000' \
    --gin_bindings='train_eval.eval_interval=5000' \
    --gin_bindings='critic_loss.delta_r_warmup=100000' \
    --gin_bindings='train_eval.run_name="darc_reacher_3"'

rm -r ~/darc/half_cheetah
python train_eval.py \
    --root_dir=~/darc/half_cheetah \
    --gin_bindings='train_eval.environment_name="half_cheetah"' \
    --gin_bindings='train_eval.delta_r_warmup=100000' \
    --gin_bindings='train_eval.eval_interval=5000' \
    --gin_bindings='critic_loss.delta_r_warmup=100000' \
    --gin_bindings='train_eval.collect_steps_per_iteration=1' \
    --gin_bindings='train_eval.train_steps_per_iteration=1' \
    --gin_bindings='train_eval.run_name="darc_half_cheetah_5"'

