local _base = import 'nl2code-base.libsonnet';
local _data_path = 'data/wikitablequestions/';

function(args, data_path=_data_path) _base(join_cond=args.join_cond, output_from=true, data_path=data_path) + {
    local lr = 0.000743552663260837,
    local end_lr = 0,
    local bs = 20,
    local seed = args.seed,
    local join_cond = args.join_cond,

    local lr_s = '%0.1e' % lr,
    local end_lr_s = '0e0',
    model_name: 'bs=%(bs)d,lr=%(lr)s,end_lr=%(end_lr)s,seed=%(seed)d,join_cond=%(join_cond)s' % ({
        bs: bs,
        lr: lr_s,
        end_lr: end_lr_s,
        seed: seed,
        join_cond: join_cond,
    }),

    model+: {
        encoder+: {
            batch_encs_update: false,
            question_encoder: ['emb', 'bilstm'],
            column_encoder: ['emb', 'bilstm-summarize'],
            table_encoder: ['emb', 'bilstm-summarize'],
            update_config+:  {
                name: 'relational_transformer',
                num_layers: 8,
                num_heads: 8,
                sc_link: true,
                cv_link: args.cv_link,
            },
            top_k_learnable: 50,
        },
        encoder_preproc+: {
            word_emb+: {
                name: 'glove',
                kind: '42B',
                lemmatize: true,
            },
            min_freq: 4,
            max_count: 5000,
            db_path: _data_path + "database",
            compute_sc_link: true,
            compute_cv_link: args.cv_link,
            fix_issue_16_primary_keys: true,
            count_tokens_in_word_emb_for_vocab: true,
            save_path: _data_path + 'nl2code-glove,cv_link=%s,join_cond=%s' % [args.cv_link, join_cond],
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: true,
                infer_from_conditions: join_cond,
                clause_order: args.clause_order,
                factorize_sketch: 2,
            },
            save_path: _data_path + 'nl2code-glove,cv_link=%s,join_cond=%s' % [args.cv_link, join_cond],

            compute_sc_link :: null,
            compute_cv_link :: null,
            fix_issue_16_primary_keys:: null,
            db_path:: null,
        },
        decoder+: {
            name: 'NL2Code',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            recurrent_size : 512,
            loss_type: "softmax",
            use_align_mat: true,
            use_align_loss: true,
            enumerate_order: args.enumerate_order,
        },
    },

    train+: {
        batch_size: bs,
        num_batch_accumulated: 1,
        clip_grad: null,

        model_seed: seed,
        data_seed:  seed,
        init_seed:  seed,
    },

    lr_scheduler+: {
        start_lr: lr,
        end_lr: end_lr,
    },

}