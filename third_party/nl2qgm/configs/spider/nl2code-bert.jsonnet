local _base = import 'nl2code-base.libsonnet';
local _output_from = true;
local _fs = 2;

function(args) _base(join_cond=args.join_cond, output_from=_output_from, data_path=args.data_path) + {
    local data_path = args.data_path,
    local join_cond = args.join_cond,
    
    local lr_s = '%0.1e' % args.lr,
    local bert_lr_s = '%0.1e' % args.bert_lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    local enc_size = if std.length(std.findSubstr('large', args.bert_version)) > 0 then 1024 else 768,
    local lm_name = 'bert',
    local lm_type = if std.length(std.findSubstr('squad', args.bert_version)) > 0 then 'squad' else '',

    // XXX: find better way to set default values
    local use_column_description = if std.objectHas(args, 'use_column_description') then args.use_column_description else false,
    local use_column_desc_emb = if std.objectHas(args, 'use_column_desc_emb') then args.use_column_desc_emb else false,
    local use_kor_full_translate = if std.objectHas(args, 'use_kor_full_translate') then args.use_kor_full_translate else false,
    local use_kor_nng_translate = if std.objectHas(args, 'use_kor_nng_translate') then args.use_kor_nng_translate else false,
    local preprocess_save_path = if std.objectHas(args, 'preprocess_save_path') then args.preprocess_save_path 
                                 else 'nl2code,join_cond=%s,output_from=%s,fs=%d,emb=%s%s,cvlink' % [join_cond, _output_from, _fs, lm_name, lm_type],
    local db_path = if std.objectHas(args, 'db_path') then args.db_path
                    else data_path + "database",
    local db_type = if std.objectHas(args, 'db_type') then args.db_type
                    else 'sqlite',
    local grammar_name = if std.objectHas(args, 'grammar_name') then args.grammar_name else "spider",

    local plm_lr = if std.objectHas(args, 'plm_lr') then args.plm_lr
                    else '',
    local plm_data_path = if std.objectHas(args, 'plm_data_path') then args.plm_data_path
                    else '',
    local plm_batch_size = if std.objectHas(args, 'plm_batch_size') then args.plm_batch_size
                    else '',
    local plm_num_batch_accumulated = if std.objectHas(args, 'plm_num_batch_accumulated') then args.plm_num_batch_accumulated
                    else '',
    local plm_max_train_steps = if std.objectHas(args, 'plm_max_train_steps') then args.plm_max_train_steps
                    else '',

    // XXX: make sure this model_name doesn't overlap with other models or find another way to identify each model
    model_name: 'bs=%(bs)d,lr=%(lr)s,bert_lr=%(bert_lr)s,end_lr=%(end_lr)s,seed=%(seed)d,join_cond=%(join_cond)s' % (args + {
        lr: lr_s,
        join_cond: join_cond,
        bert_lr: bert_lr_s,
        end_lr: end_lr_s,
    }),

    plm:{
        lr: plm_lr,
        data_path: plm_data_path,
        batch_size: plm_batch_size,
        num_batch_accumulated: plm_num_batch_accumulated,
        max_train_steps: plm_max_train_steps,
    },

    data+: {
        train+: {
            use_column_description: use_column_description,
            db_path: db_path,
            db_type: db_type,
            grammar: grammar_name,
        },
        val+: {
            use_column_description: use_column_description,
            db_path: db_path,
            db_type: db_type,
            grammar: grammar_name,
        },
    },

    model+: {
        encoder+: {
            name: 'spider-bert',
            batch_encs_update:: null,
            question_encoder:: null,
            column_encoder:: null,
            table_encoder:: null,
            dropout:: null,
            update_config+:  {
                name: 'relational_transformer',
                num_layers: args.num_layers,
                num_heads: 8,
                sc_link: args.sc_link,
                cv_link: args.cv_link,
            },
            summarize_header: args.summarize_header,
            use_column_type: args.use_column_type,
            bert_version: args.bert_version,
            bert_token_type: args.bert_token_type,
            top_k_learnable:: null,
            word_emb_size:: null,
        },
        encoder_preproc+: {
            use_kor_full_translate: use_kor_full_translate,
            use_kor_nng_translate: use_kor_nng_translate,

            use_column_description: use_column_description,
            /* JHCHO - 21.11.02: Add col-desc emb */
            use_column_desc_emb: use_column_desc_emb,

            word_emb:: null,
            min_freq:: null,
            max_count:: null,
            db_path: db_path,
            compute_sc_link: args.sc_link,
            compute_cv_link: args.cv_link,
            fix_issue_16_primary_keys: true,
            bert_version: args.bert_version,
            count_tokens_in_word_emb_for_vocab:: null,
            /* 
            XXX: save_path overlaps after adding several new arguments. 
            we need to find another way to identify preprocessed data files
            */
            save_path: data_path + preprocess_save_path,
        },
        decoder_preproc+: {
            grammar+: {
                name: if grammar_name == "postgresql" then "postgres" else "spider",
                end_with_from: args.end_with_from,
                clause_order: args.clause_order,
                infer_from_conditions: join_cond,
                factorize_sketch: _fs,
                include_literals: args.include_literals,
            },
            /* 
            XXX: save_path overlaps after adding several new arguments. 
            we need to find another way to identify preprocessed data files
            */
            save_path: data_path + preprocess_save_path,
            db_path: db_path,
            db_type: db_type,

            compute_sc_link:: null,
            compute_cv_link:: null,
            fix_issue_16_primary_keys:: null,
            bert_version:: null,
            use_kor_full_translate:: null,
            use_kor_nng_translate:: null,
            use_column_desc_emb:: null,
        },
        decoder+: {
            name: 'NL2Code',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            enc_recurrent_size: enc_size,
            recurrent_size : args.decoder_hidden_size,
            loss_type: 'label_smooth',
            use_align_mat: args.use_align_mat,
            use_align_loss: args.use_align_loss,
        }
    },

    train+: {
        batch_size: args.bs,
        num_batch_accumulated: args.num_batch_accumulated,
        clip_grad: 1,

        model_seed: args.seed,
        data_seed:  args.seed,
        init_seed:  args.seed,
        
        max_steps: args.max_steps,
    },

    optimizer: {
        name: 'bertAdamw',
        lr: 0.0,
        bert_lr: 0.0,
        freeze_bert: false,
    },

    lr_scheduler+: {
        name: 'bert_warmup_polynomial_group',
        start_lrs: [args.lr, args.bert_lr],
        end_lr: args.end_lr,
        num_warmup_steps: $.train.max_steps / 8,
    },

    log: {
        reopen_to_flush: true,
    }
}
