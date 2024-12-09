function(output_from, data_path='data/cosql/') {
    local PREFIX = data_path,

    data: {
        train: {
            name: 'cosql', 
            paths: [PREFIX + 'user_intent_prediction/cosql_train.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },
        val: {
            name: 'cosql', 
            paths: [PREFIX + 'user_intent_prediction/cosql_dev.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'cosql-user_intent',
            dropout: 0.2,
            word_emb_size: 300,
            question_encoder: ['emb', 'bilstm'],
            column_encoder: ['emb', 'bilstm-summarize'],
            table_encoder: ['emb', 'bilstm-summarize'],
            update_config:  {
                name: 'relational_transformer',
                num_layers: 4,
                num_heads: 8,
            },
        },   
        decoder: {
            name: 'NL2Intent',
            dropout: 0.2,
            desc_attn: 'mha',
        },
        encoder_preproc: {
            word_emb: {
                name: 'glove',
                kind: '42B',
            },
            count_tokens_in_word_emb_for_vocab: false,
            min_freq: 50,
            max_count: 5000,
            include_table_name_in_column: false,

            save_path: PREFIX + 'emb=glove-42B,min_freq=%d/' % [50],
        },
        decoder_preproc: self.encoder_preproc {
            word_emb:: null,
            include_table_name_in_column:: null,
            count_tokens_in_word_emb_for_vocab:: null,
        },
    },

    train: {
        batch_size: 10,
        eval_batch_size: 50,

        keep_every_n: 1000,
        eval_every_n: 1000,
        save_every_n: 1000,
        report_every_n: 10,

        max_steps: 40000,
        num_eval_items: 50,
    },
    optimizer: {
        name: 'adam',
        lr: 0.0,
    },
    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: 1e-3,
        end_lr: 0,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    }
}
