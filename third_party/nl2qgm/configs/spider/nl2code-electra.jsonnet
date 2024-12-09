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
    local lm_name= 'electra',
    local lm_type = if std.length(std.findSubstr('squad', args.bert_version)) > 0 then 'squad' else '',

    model_name: 'bs=%(bs)d,lr=%(lr)s,bert_lr=%(bert_lr)s,end_lr=%(end_lr)s,seed=%(seed)d,join_cond=%(join_cond)s' % (args + {
        lr: lr_s,
        join_cond: join_cond,
        bert_lr: bert_lr_s,
        end_lr: end_lr_s,
    }),

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
                like_t5: args.like_t5,
            },
            summarize_header: args.summarize_header,
            use_column_type: args.use_column_type,
            bert_version: args.bert_version,
            bert_token_type: args.bert_token_type,
            top_k_learnable:: null,
            word_emb_size:: null,
        },
        encoder_preproc+: {
            word_emb:: null,
            min_freq:: null,
            max_count:: null,
            db_path: data_path + "database",
            compute_sc_link: args.sc_link,
            compute_cv_link: args.cv_link,
            fix_issue_16_primary_keys: true,
            bert_version: args.bert_version,
            count_tokens_in_word_emb_for_vocab:: null,
            save_path: data_path + 'nl2code,join_cond=%s,output_from=%s,fs=%d,emb=%s%s,cvlink' % [join_cond, _output_from, _fs, lm_name, lm_type],
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: args.end_with_from,
                clause_order: args.clause_order,
                infer_from_conditions: join_cond,
                factorize_sketch: _fs,
            },
            save_path: data_path + 'nl2code,join_cond=%s,output_from=%s,fs=%d,emb=%s%s,cvlink' % [join_cond, _output_from, _fs, lm_name, lm_type],
            db_path: data_path + "database",

            compute_sc_link:: null,
            compute_cv_link:: null,
            fix_issue_16_primary_keys:: null,
            bert_version:: null,
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
