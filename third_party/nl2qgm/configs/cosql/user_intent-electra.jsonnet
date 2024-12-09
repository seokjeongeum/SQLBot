local _base = import 'user_intent-base.libsonnet';
local _output_from = true;
local _fs = 2;

function(args) _base(output_from=_output_from, data_path=args.data_path) + {
    local data_path = args.data_path,
    
    local lr_s = '%0.1e' % args.lr,
    local bert_lr_s = '%0.1e' % args.bert_lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    local enc_size = if std.length(std.findSubstr('large', args.bert_version)) > 0 then 1024 else 768,
    local lm_name= 'electra',
    local lm_type = if std.length(std.findSubstr('squad', args.bert_version)) > 0 then 'squad' else '',
    local like_t5 = if std.objectHas(args, 'like_t5') then args.like_t5 else false,

    model_name: 'bs=%(bs)d,lr=%(lr)s,bert_lr=%(bert_lr)s,end_lr=%(end_lr)s,seed=%(seed)d' % (args + {
        lr: lr_s,
        bert_lr: bert_lr_s,
        end_lr: end_lr_s,
        end: args.seed
    }),

    model+: {
        encoder+: {
            name: 'cosql-bert',
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
                like_t5: like_t5,
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
            save_path: data_path + 'user_intent_prediction/nl2intent,output_from=%s,fs=%d,emb=%s%s,cvlink' % [_output_from, _fs, lm_name, lm_type],
        },
        decoder_preproc+: {
            save_path: data_path + 'user_intent_prediction/nl2intent,output_from=%s,fs=%d,emb=%s%s,cvlink' % [_output_from, _fs, lm_name, lm_type],
            db_path: data_path + "database/",

            compute_sc_link:: null,
            compute_cv_link:: null,
            fix_issue_16_primary_keys:: null,
            bert_version:: null,
        },
        decoder+: {
            name: 'NL2Intent',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            enc_recurrent_size: enc_size,
            recurrent_size : args.decoder_hidden_size,
            loss_type: 'label_smooth',
            use_align_mat: args.use_align_mat,
            use_align_loss: args.use_align_loss,
            output_classes: ["ambiguous", "infer_sql", "addtion", "sorry", "cannot_answer", "not_related", "greeting", "good_bye", "cannot_understand", "inform_sql", "affirm", "negate", "thank_you", "inform_sql"]
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
