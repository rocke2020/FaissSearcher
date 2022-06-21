from dataclasses import dataclass, field


DISEASE = 'DISEASE'
GENE = 'GENE'


@dataclass
class StanzaArguments:
    """
    Arguments for stanza
    """
    ner_model_type: str = field(
        default='stanza',
        metadata={"help": "bert, stanza"},
    )    
    batch_size: int = field(
        default=16,
        metadata={"help": ""},
    )
    tokenizer_model: str = field(
        default='spacy',
        metadata={"help": "spacy tokenizer_model is faster"},
    )


@dataclass
class LinkerArguments:
    """
    Arguments for linking process
    """
    top_num: int = field(
        default=1,
        metadata={"help": "the similiarest number to retrieve"},
    )
    search_batch_size: int = field(
        default=500,
        metadata={"help": "the similiarest number to retrieve"},
    )
    drop_collection: bool = field(
        default=False,
        metadata={
            "help": "drop milvus collection to start a new one, 1 true 0 false.",
        },
    )
    cache_vector_df: bool = field(
        default=False, metadata={"help": ""}
    )
    release_after_initialization: bool = field(
        default=False, metadata={"help": "release_after_initialization"}
    )
    link_lower_case: bool = field(
        default=True, metadata={"help": "Always uses lower case"}
    )
    dictionary_type: str = field(
        default='csv', 
        metadata={"help": """ dictionary file type, relative with corpus_type in CorpusArguments
            csv, tsv
            DiseaseForMetanovas  # a special-case .xlsx format 
            """}
    )
    extra_out_fields: str = field(
        default='metanovas_id', 
        metadata={"help": """ split by ' ', relative with corpus_type in CorpusArguments
            metanovas_id
            fullname categories
            """}
    )
    overwrite_ner_output: bool = field(
        default=True,
        metadata={
            "help": "overwrite cached NER output"
        },
    )
    save_orig_linked_output: bool = field(
        default=False,
        metadata={
            "help": "save the orig_linked_output before filtering which can be also called link result analysis"
        },
    )
    link_result_file_prefix: str = field(
        default='',
        metadata={
            "help": "link_result_file_prefix"
        },
    )
    linker_type: str = field(
        default='milvus',
        metadata={
            "help": "milvus, faiss"
        },
    )

@dataclass
class CorpusArguments:
    """
    Arguments for corpus relative, including cache
    """
    corpus_root: str = field(
        default='/home/qcdong/corpus/NER',
        metadata={"help": "If corpus_root is empty string, the corpus will not be auto read to save time, useful to predict sentences which are not from corpus"},
    )
    corpus_type: str = field(
        default='DISEASE',  # GENE DISEASE 'DiseaseForMetanovas'
        metadata={"help": "This value is also the milvus collection name"},
    )
    corpus_subtype: str = field(
        default='Disease',
        metadata={
            "help": """ disease: [BC5CDR, NCBI, merged]
            gene: [JNLPBA, ]
            """
        },
    )
    corpus_split_mode: str = field(
        default='test', 
        metadata={"help": "splitted parts: test, dev, and train"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "overwrite cached preprocess NER dataset"
        },
    )
    enable_start_index_forward_one_sentence: bool = field(
        default=False,
        metadata={
            "help": "If enable, there is one sentence forward overlap by the last comma or dot when spliting long texts"
        },
    )


@dataclass
class BertArguments:
    """
    Arguments for bert, only for predict in pipeline
    """
    ner_model_type: str = field(
        default='bert',
        metadata={"help": "bert, stanza, autoner"},
    )        
    model_type: str = field(
        default='bert',
        metadata={"help": "bert, robert, xlnet etc"},
    )
    model_subtype: str = field(
        default='bert',
        metadata={"help": "bert, biobert, bioelectra etc"},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": ""
        },
    )
    enable_word_piece_tokenizer: int = field(
        default=True,
        metadata={
            "help": "when predict from sentences, enable_word_piece_tokenizer True, keep deault True for simple"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={
            "help": "Set this flag 1 if you are using an uncased(lower case) model."
        },
    )
    per_gpu_eval_batch_size: int = field(
        default=32,
        metadata={
            "help": "Batch size per GPU/CPU for evaluation."
        },
    )
    no_cuda: bool = field(
        default=0,
        metadata={
            "help": "Avoid using CUDA when available"
        },
    )
    prefix_ner_output_cache: str = field(
        default='',
        metadata={"help": "prefix_ner_output_cache"},
    )
    enable_cache_ner_output: bool = field(
        default=0,
        metadata={"help": "enable_cache_ner_output"},
    )
    local_rank: int = field(
        default=-1,
        metadata={
            "help": """KEEP it as corpus_data_processor.py needs this argument! For distributed training, -1 is local.
            if args.local_rank in [-1, 0]:
                torch.save(features, cached_features_file)
            else:
                need to config not [-1, 0] when don't need to cache featrues.
            """
        },
    )


@dataclass
class PubmedArguments:
    """
    Arguments for Pubmed
    """
    total_parts_num: int = field(
        default=2,
        metadata={"help": "the total split parts of files to run in different server"},
    )    
    part_seq: int = field(
        default=1,
        metadata={"help": "The current splt part sequence, start from 1 to total_parts"},
    )
    start_index: int = field(
        default=-1,
        metadata={"help": "from start_index to the end"},
    )
    overwrite_pubmed_out_result: bool = field(
        default=0,
        metadata={"help": "overwrite_pubmed_out_result"},
    )
    fix_seq_len: bool = field(
        default=0,
        metadata={"help": "overwrite_pubmed_out_result"},
    )
    file_sequence_reverse: bool = field(
        default=0,
        metadata={"help": "overwrite_pubmed_out_result"},
    )
