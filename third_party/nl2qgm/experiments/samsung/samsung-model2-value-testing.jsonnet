local exp_name = "samsung_model2-value-testing";
{
    logdir: "logdir/" + exp_name,
    model_config: "configs/samsung/nl2code-bert.jsonnet",
    model_config_args: {
        data_path: 'data/samsung_model2-value-testing/',
        bert_version: "markussagen/xlm-roberta-longformer-base-4096",
        bs: 1,
        num_batch_accumulated: 8,
        seed: 0,
        preprocess_save_path: exp_name,
        use_kor_nng_translate: false,
        db_type: "postgresql",
        db_path: "user=postgres password=postgres host=localhost port=5435",
        use_column_description: true,
        use_column_desc_emb: true,
        include_literals: true,
        
        summarize_header: "avg",
        use_column_type: false,
        max_steps: 16000,
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
    },

    eval_name: exp_name,
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [55750],
    eval_section: "val",
    loss_type: "label_smooth",
    use_original_eval: false,
}
