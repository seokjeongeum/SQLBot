local _base = import 'nl2code-bert.jsonnet';

function(args) _base(args) + {
    data+: {
        train+: {
            grammar: 'spider'
        },
        val+: {
            grammar: 'spider'
        },
    },    

    model+: {
        decoder_preproc+: {
            grammar+: {
                name: 'spider'
            },
        },
    },
}