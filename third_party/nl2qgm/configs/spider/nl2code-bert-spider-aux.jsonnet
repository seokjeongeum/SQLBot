local _base = import 'nl2code-bert.jsonnet';

function(args) _base(args) + {
    local aux_data_path = args.aux_data_path,
    local aux_db_path = args.aux_db_path,
    local aux_db_type = args.aux_db_type,

    data+: {
        train+: {
            grammar: 'postgres',
        },
        val+: {
            grammar: 'postgres',
        },
        // TODO: it must be tested
        aux: {
            name: 'spider',
            paths: [aux_data_path + 'all.json'],
            tables_paths: [aux_data_path + 'tables.json'],
            db_path: aux_db_path,
            db_type: aux_db_type,
            grammar: 'postgres',
        },
    },    

    model+: {
        decoder_preproc+: {
            grammar+: {
                name: 'postgres'
            },
        },
    },
}
