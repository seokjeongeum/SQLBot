{
    logdir: "logdir/kaggle-coldesc-fewshot-div/%s/" % [self.model_config_args.db_name],
    model_config: "configs/kaggle/nl2code-coldesc.jsonnet",
    model_config_args: {
        // 1. change here to test another db
        // 2. preprocess the dataset
        // 3. change dec_vocab.json, grammar_rules.json, observed_productions.json file to spider's ones
        // 4. try to evaluate or fine-tune, so that it make logdir
        // 5. link the best pretrained model
        // 6. evaluate or fine-tune
        db_name: 'WorldSoccerDataBase',

        data_path: 'data/kaggle-dbqa-div/%s/' % [self.db_name],
        join_cond: false,
        bs: 1,
        num_batch_accumulated: 24,
        seed: 0,
        use_column_description: true,
        
        bert_version: "bert-large-uncased-whole-word-masking",
        summarize_header: "avg",
        use_column_type: false,
        max_steps: 24020,
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
    },

    eval_name: "kaggle_coldesc_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [24000 + 100 * x for x in std.range(0, 11)],
    eval_section: "val",
    loss_type: "label_smooth",
    use_original_eval: true,
}
