{
    logdir: "logdir/spider_glove_run_no_join_cond",
    model_config: "configs/spider/nl2code-glove.jsonnet",
    model_config_args: {
        seed: 2,
        join_cond: false,
        cv_link: true,
        clause_order: null, # strings like "SWGOIF"
        enumerate_order: false,
    },

    eval_name: "glove_run_no_Join_cond_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [39200],
    // eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
}