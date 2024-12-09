{
    logdir: "logdir/spider-kor-nng_translate",
    model_config: "configs/spider/nl2code-kor_nng_translate.jsonnet",
    model_config_args: {
        data_path: 'data/spider-kor/',
        bs: 6,
        num_batch_accumulated: 4,
        bert_version: "bert-base-multilingual-uncased",
        use_kor_nng_translate: true,

        summarize_header: "avg",
        use_column_type: false,
        max_steps: 81000,
        num_layers: 8,
        lr: 7.44e-4,
        bert_lr: 3e-6,
        att: 4,
        end_lr: 0,
        sc_link: true,
        cv_link: true,
        use_align_mat: true,
        use_align_loss: true,
        bert_token_type: true,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true
        clause_order: null, # strings like "SWGOIF", it will be prioriotized over end_with_from 
        join_cond: false,
        seed: 1,
    },

    eval_name: "spider-kor-nng_translate",
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
    loss_type: "label_smooth",
}
